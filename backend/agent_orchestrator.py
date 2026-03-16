from __future__ import annotations

import asyncio
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from dotenv import dotenv_values, load_dotenv

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except Exception:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI

BACKEND_DIR = os.path.dirname(__file__)
PROJECT_ROOT_DIR = os.path.dirname(BACKEND_DIR)
BACKEND_ENV_PATH = os.path.join(BACKEND_DIR, ".env")
PROJECT_ROOT_ENV_PATH = os.path.join(PROJECT_ROOT_DIR, ".env")

load_dotenv(dotenv_path=BACKEND_ENV_PATH)
load_dotenv(dotenv_path=PROJECT_ROOT_ENV_PATH)


def _resolve_gemini_api_key() -> str | None:
    """Resolve Gemini API key from process env or dotenv files."""
    key_from_env = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key_from_env:
        return key_from_env

    for env_path in (BACKEND_ENV_PATH, PROJECT_ROOT_ENV_PATH):
        if not os.path.exists(env_path):
            continue
        values = dotenv_values(env_path)
        key_from_file = values.get("GOOGLE_API_KEY") or values.get("GEMINI_API_KEY")
        if isinstance(key_from_file, str) and key_from_file.strip():
            return key_from_file.strip()

    return None


SYSTEM_PROMPT = (
    "You are a drone-orchestration agent for search-and-rescue. "
    "Always discover drones with list_active_drones before issuing drone-specific commands. "
    "Use tools as needed and perform multiple tool calls in a single response when required. "
    "Use shared memory from prior thermal scans to avoid re-scanning known cells unless explicitly requested. "
    "When thermal_scan reports a suspect hit, do not immediately confirm survivor status. "
    "First triage using battery and mission priority: compare available active drones, check battery levels, "
    "and decide whether to dispatch a second drone for multimodal verification or continue sweep. "
    "Only use verify_survivor after this triage decision is justified. "
    "When a survivor is verified, transition to Human Rescue Routing: call plan_human_rescue_route "
    "to compute an A* path from base (20,20) to the verified location."
)


@dataclass
class SharedMemoryContext:
    """Persistent in-process context built from thermal scan tool outputs."""

    # (x, y) -> {'status': 'clear'|'suspect', 'source_drone_id': int|None}
    scanned_cells: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    # (x, y) -> {'source_drone_id': int|None, 'verified': bool}
    pending_verification: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    # (x, y) verified locations discovered by multimodal checks.
    confirmed_survivors: set[tuple[int, int]] = field(default_factory=set)
    # (x, y) -> {'path_length': int, 'total_cost': int}
    rescue_routes: dict[tuple[int, int], dict[str, int]] = field(default_factory=dict)

    def update_from_thermal_scan(self, observation: Any) -> None:
        """Ingest a thermal_scan observation payload into shared memory."""
        if not isinstance(observation, dict):
            return

        position = observation.get("position")
        status = observation.get("status")
        drone_id = observation.get("drone_id")

        if not isinstance(position, list) or len(position) != 2:
            return
        if status not in {"clear", "suspect"}:
            return

        x, y = int(position[0]), int(position[1])
        self.scanned_cells[(x, y)] = {
            "status": status,
            "source_drone_id": int(drone_id) if isinstance(drone_id, int) else None,
        }
        if status == "suspect":
            self.pending_verification[(x, y)] = {
                "source_drone_id": int(drone_id) if isinstance(drone_id, int) else None,
                "verified": False,
            }

    def update_from_verify_survivor(self, observation: Any) -> None:
        """Ingest verify_survivor output and resolve pending suspect status."""
        if not isinstance(observation, dict):
            return

        position = observation.get("position")
        status = observation.get("status")
        if not isinstance(position, list) or len(position) != 2:
            return

        x, y = int(position[0]), int(position[1])
        pos_key = (x, y)
        if pos_key not in self.pending_verification:
            return

        if status == "Confirmed Survivor":
            self.pending_verification[pos_key]["verified"] = True
            self.confirmed_survivors.add(pos_key)
        else:
            # Verification attempted and not confirmed; remove from pending queue.
            self.pending_verification.pop(pos_key, None)

    def update_from_rescue_route(self, observation: Any) -> None:
        """Ingest route-planning output for a confirmed survivor."""
        if not isinstance(observation, dict):
            return

        status = observation.get("status")
        destination = observation.get("to")
        path = observation.get("path")
        total_cost = observation.get("total_cost")
        if status != "Human Rescue Route Ready":
            return
        if not isinstance(destination, list) or len(destination) != 2:
            return
        if not isinstance(path, list) or not isinstance(total_cost, int):
            return

        x, y = int(destination[0]), int(destination[1])
        self.rescue_routes[(x, y)] = {
            "path_length": len(path),
            "total_cost": int(total_cost),
        }

    def to_prompt_context(self) -> str:
        """Render compact shared-memory context for agent planning."""
        if not self.scanned_cells:
            return "No scanned cells yet."

        clear_cells = [pos for pos, data in self.scanned_cells.items() if data["status"] == "clear"]
        suspect_cells = [pos for pos, data in self.scanned_cells.items() if data["status"] == "suspect"]
        unresolved = [
            pos for pos, data in self.pending_verification.items() if not bool(data.get("verified", False))
        ]
        confirmed = list(self.confirmed_survivors)
        routed = list(self.rescue_routes.keys())

        return (
            f"Known clear cells ({len(clear_cells)}): {clear_cells[:20]}\n"
            f"Known suspect cells ({len(suspect_cells)}): {suspect_cells[:20]}\n"
            f"Pending suspect verification cells ({len(unresolved)}): {unresolved[:20]}\n"
            f"Confirmed survivors ({len(confirmed)}): {confirmed[:20]}\n"
            f"Confirmed survivors with planned rescue routes ({len(routed)}): {routed[:20]}\n"
            "Planning rule: avoid thermal_scan on known clear/suspect cells unless recheck is explicitly required."
        )

    def to_global_plan(self) -> str:
        """Return a concise global planning directive based on shared memory."""
        suspect_count = sum(1 for data in self.scanned_cells.values() if data["status"] == "suspect")
        clear_count = sum(1 for data in self.scanned_cells.values() if data["status"] == "clear")
        unresolved_count = sum(
            1 for data in self.pending_verification.values() if not bool(data.get("verified", False))
        )
        confirmed_count = len(self.confirmed_survivors)
        routed_count = len(self.rescue_routes)
        return (
            "Global plan: prioritize unscanned cells for exploration, dispatch secondary verification for suspect "
            f"cells, and skip redundant scans for {clear_count + suspect_count} already-scanned cells. "
            f"Unresolved suspect cells awaiting triage/verification: {unresolved_count}. "
            f"Verified survivors: {confirmed_count}. Rescue routes prepared: {routed_count}."
        )


SUSPECT_TRIAGE_POLICY = (
    "Suspect triage policy:\n"
    "1) After suspect thermal hit, do not confirm survivor immediately.\n"
    "2) Call list_active_drones and get_battery_status for candidate drones before deciding.\n"
    "3) If a second drone has sufficient battery margin for travel + verification + return, dispatch it to verify.\n"
    "4) If battery margins are poor or mission priority favors broader search, continue sweep and queue suspect for later verification.\n"
    "5) Only call verify_survivor after documenting why verification is now justified."
)


HUMAN_RESCUE_ROUTING_POLICY = (
    "Human Rescue Routing policy:\n"
    "1) If verify_survivor returns Confirmed Survivor, do not end the turn immediately.\n"
    "2) Call plan_human_rescue_route(x, y) for the confirmed location in the same decision cycle.\n"
    "3) Ensure the route is A*-based and originates from base (20,20).\n"
    "4) Report route readiness, path cost, and handoff instructions for human rescue teams."
)


async def _load_mcp_tools(server_script: str = "mcp_server.py") -> list[Any]:
    """Load all tools exposed by the MCP server via the LangChain MCP adapter."""
    script_path = (
        server_script
        if os.path.isabs(server_script)
        else os.path.join(os.path.dirname(__file__), server_script)
    )

    client = MultiServerMCPClient(
        {
            "swarm-sim": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [script_path],
            }
        }
    )
    return await client.get_tools()


def _run_coroutine_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, Exception] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - passthrough
            error["value"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")


def _build_llm() -> ChatGoogleGenerativeAI:
    """Initialize an LLM backend for the tool-calling agent.

    Default model is Gemini and can be overridden with AGENT_MODEL.
    """
    model_name = os.getenv("AGENT_MODEL", "gemini-2.0-flash")
    temperature = float(os.getenv("AGENT_TEMPERATURE", "0"))
    api_key = _resolve_gemini_api_key()
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in environment, "
            f"{BACKEND_ENV_PATH}, or {PROJECT_ROOT_ENV_PATH}."
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def create_orchestrator(server_script: str = "mcp_server.py") -> AgentExecutor:
    """Create a LangChain agent executor bound to all MCP tools."""
    tools = _run_coroutine_sync(_load_mcp_tools(server_script=server_script))
    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + "\n\nShared Memory Context:\n{shared_memory_context}\n\n"
                + "Global Search Plan:\n{global_search_plan}\n\n"
                + "Decision Policy:\n{suspect_triage_policy}\n\n"
                + "Mission State Policy:\n{human_rescue_routing_policy}",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    # max_iterations>1 allows chained/multiple tool calls in one turn.
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10,
        return_intermediate_steps=True,
        verbose=True,
    )


class CommandAgentOrchestrator:
    """Stateful orchestrator that persists shared scan memory across turns."""

    def __init__(self, server_script: str = "mcp_server.py"):
        self.executor = create_orchestrator(server_script=server_script)
        self.shared_memory = SharedMemoryContext()

    def _ingest_tool_results(self, intermediate_steps: list[Any]) -> None:
        for step in intermediate_steps:
            if not isinstance(step, tuple) or len(step) != 2:
                continue

            action, observation = step
            tool_name = getattr(action, "tool", None)
            if tool_name == "thermal_scan":
                self.shared_memory.update_from_thermal_scan(observation)
            elif tool_name == "verify_survivor":
                self.shared_memory.update_from_verify_survivor(observation)
            elif tool_name == "plan_human_rescue_route":
                self.shared_memory.update_from_rescue_route(observation)

    def invoke(self, user_input: str) -> dict[str, Any]:
        payload = {
            "input": user_input,
            "shared_memory_context": self.shared_memory.to_prompt_context(),
            "global_search_plan": self.shared_memory.to_global_plan(),
            "suspect_triage_policy": SUSPECT_TRIAGE_POLICY,
            "human_rescue_routing_policy": HUMAN_RESCUE_ROUTING_POLICY,
        }
        result = self.executor.invoke(payload)
        self._ingest_tool_results(result.get("intermediate_steps", []))
        result["shared_memory"] = {
            "scanned_cells": {
                f"{x},{y}": data for (x, y), data in self.shared_memory.scanned_cells.items()
            },
            "confirmed_survivors": [f"{x},{y}" for (x, y) in sorted(self.shared_memory.confirmed_survivors)],
            "rescue_routes": {
                f"{x},{y}": data for (x, y), data in self.shared_memory.rescue_routes.items()
            },
        }
        return result

    def stream_mission_events(
        self,
        user_input: str,
        should_stop: Callable[[], bool] | None = None,
    ):
        """Yield mission events suitable for SSE transport.

        The stream emits THOUGHT-style summaries tied to tool actions and
        observations, then a final result message.
        """
        payload = {
            "input": user_input,
            "shared_memory_context": self.shared_memory.to_prompt_context(),
            "global_search_plan": self.shared_memory.to_global_plan(),
            "suspect_triage_policy": SUSPECT_TRIAGE_POLICY,
            "human_rescue_routing_policy": HUMAN_RESCUE_ROUTING_POLICY,
        }

        collected_steps: list[Any] = []
        final_output = ""

        for chunk in self.executor.iter(payload):
            if should_stop and should_stop():
                yield {
                    "event": "status",
                    "data": {
                        "message": "mission stopped by operator",
                    },
                }
                return

            if not isinstance(chunk, dict):
                continue

            if "actions" in chunk:
                for action in chunk["actions"]:
                    tool_name = getattr(action, "tool", "unknown_tool")
                    tool_input = getattr(action, "tool_input", {})
                    yield {
                        "event": "thought",
                        "data": {
                            "label": "THOUGHT",
                            "message": f"Planning next step: calling {tool_name} with {tool_input}.",
                        },
                    }

            if "steps" in chunk:
                for action, observation in chunk["steps"]:
                    collected_steps.append((action, observation))
                    tool_name = getattr(action, "tool", "unknown_tool")
                    yield {
                        "event": "observation",
                        "data": {
                            "tool": tool_name,
                            "observation": observation,
                        },
                    }

            if "output" in chunk:
                final_output = str(chunk["output"])

        self._ingest_tool_results(collected_steps)

        yield {
            "event": "final",
            "data": {
                "output": final_output,
                "shared_memory": {
                    "scanned_cells": {
                        f"{x},{y}": data for (x, y), data in self.shared_memory.scanned_cells.items()
                    },
                    "confirmed_survivors": [
                        f"{x},{y}" for (x, y) in sorted(self.shared_memory.confirmed_survivors)
                    ],
                    "rescue_routes": {
                        f"{x},{y}": data for (x, y), data in self.shared_memory.rescue_routes.items()
                    },
                },
            },
        }


def run_once(user_input: str, server_script: str = "mcp_server.py") -> dict[str, Any]:
    """Convenience helper to run one orchestration turn."""
    orchestrator = CommandAgentOrchestrator(server_script=server_script)
    return orchestrator.invoke(user_input)


if __name__ == "__main__":
    result = run_once("Discover active drones and report their status.")
    print(result["output"])
