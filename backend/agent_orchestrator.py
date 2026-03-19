from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except Exception:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = (
    "You are the SwarmPatrol mission commander for a drone-based urban search-and-rescue simulation. "
    "Primary mission objective: from base (20,20), move drones and thermal-scan cells until all cells in the 40x40 grid are scanned. "
    "You must operate by calling MCP tools for every mission action, observation, and decision. "
    "Do not invent telemetry or outcomes when a tool can provide the data. "
    "Start each cycle by calling list_active_drones before issuing drone-specific commands. "
    "To search a location, command movement with move_to and then call thermal_scan at the drone's current position. "
    "Do not assume movement or scanning happens automatically. "
    "Do not rely on autonomous simulation stepping for drone actions. "
    "Use chained MCP tool calls in the same turn when needed to complete a coherent decision cycle. "
    "Prioritize unscanned cells and use shared memory to avoid redundant thermal_scan calls unless explicitly asked to recheck. "
    "If thermal_scan returns an unconfirmed hit, do not mark it as a survivor immediately. "
    "Immediately assign a different active drone to move_to that same cell and run thermal_scan again for a second thermal pass. "
    "After this second-drone thermal re-scan, perform triage with MCP tools and call verify_survivor only when justified. "
    "Keep outputs concise, mission-focused, and grounded in MCP tool observations."
)


MISSION_DECOMPOSITION_POLICY = (
    "Autonomous mission planning policy:\n"
    "1) Decompose each high-level objective into sequenced tool actions before execution.\n"
    "2) For this mission, coverage objective is complete scan of all 40x40 cells starting from base (20,20).\n"
    "3) Typical sequence: discover fleet -> assign drone-task pairs -> move_to -> thermal_scan -> update coverage map.\n"
    "4) Distribute active drones across non-overlapping nearby cells to maximize coverage rate.\n"
    "5) Re-plan after each observation instead of committing to long fixed scripts."
)


FLEET_RESOURCE_POLICY = (
    "Strategic resource policy:\n"
    "1) Before dispatching a drone, check get_battery_status and estimate movement budget.\n"
    "2) Prefer assigning farther targets to higher-battery drones and nearer targets to lower-battery drones.\n"
    "3) Recall low-battery drones to base (20,20) with move_to before critical depletion risk.\n"
    "4) Keep at least one active drone available for verification when possible.\n"
    "5) Explain battery/coverage tradeoffs briefly before executing non-trivial assignments."
)


MCP_COMPLIANCE_POLICY = (
    "MCP-only compliance policy:\n"
    "1) All drone actions must be executed through MCP tools only.\n"
    "2) Do not assume implicit movement, implicit scanning, or hard-coded drone IDs.\n"
    "3) Always discover active drones via list_active_drones at the start of each planning cycle."
)


REASONING_SUMMARY_POLICY = (
    "Reasoning summary policy:\n"
    "Before important tool actions, provide a short rationale tied to mission coverage, risk, and battery status."
)


@dataclass
class SharedMemoryContext:
    """Persistent in-process context built from thermal scan tool outputs."""

    # (x, y) -> {'status': 'clear'|'unconfirmed', 'source_drone_id': int|None}
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
        if status not in {"clear", "unconfirmed"}:
            return

        x, y = int(position[0]), int(position[1])
        self.scanned_cells[(x, y)] = {
            "status": status,
            "source_drone_id": int(drone_id) if isinstance(drone_id, int) else None,
        }
        if status == "unconfirmed":
            self.pending_verification[(x, y)] = {
                "source_drone_id": int(drone_id) if isinstance(drone_id, int) else None,
                "verified": False,
            }

    def update_from_verify_survivor(self, observation: Any) -> None:
        """Ingest verify_survivor output and resolve pending unconfirmed status."""
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
        unconfirmed_cells = [
            pos for pos, data in self.scanned_cells.items() if data["status"] == "unconfirmed"
        ]
        unresolved = [
            pos for pos, data in self.pending_verification.items() if not bool(data.get("verified", False))
        ]
        confirmed = list(self.confirmed_survivors)
        routed = list(self.rescue_routes.keys())

        return (
            f"Known clear cells ({len(clear_cells)}): {clear_cells[:20]}\n"
            f"Known unconfirmed cells ({len(unconfirmed_cells)}): {unconfirmed_cells[:20]}\n"
            f"Pending unconfirmed verification cells ({len(unresolved)}): {unresolved[:20]}\n"
            f"Confirmed survivors ({len(confirmed)}): {confirmed[:20]}\n"
            f"Confirmed survivors with planned rescue routes ({len(routed)}): {routed[:20]}\n"
            "Planning rule: avoid thermal_scan on known clear/unconfirmed cells unless recheck is explicitly required."
        )

    def to_global_plan(self) -> str:
        """Return a concise global planning directive based on shared memory."""
        unconfirmed_count = sum(
            1 for data in self.scanned_cells.values() if data["status"] == "unconfirmed"
        )
        clear_count = sum(1 for data in self.scanned_cells.values() if data["status"] == "clear")
        unresolved_count = sum(
            1 for data in self.pending_verification.values() if not bool(data.get("verified", False))
        )
        confirmed_count = len(self.confirmed_survivors)
        routed_count = len(self.rescue_routes)
        return (
            "Global plan: prioritize unscanned cells for exploration, dispatch secondary verification for unconfirmed "
            f"cells, and skip redundant scans for {clear_count + unconfirmed_count} already-scanned cells. "
            f"Unresolved unconfirmed cells awaiting triage/verification: {unresolved_count}. "
            f"Verified survivors: {confirmed_count}. Rescue routes prepared: {routed_count}."
        )


UNCONFIRMED_TRIAGE_POLICY = (
    "Unconfirmed triage policy:\n"
    "1) After an unconfirmed thermal hit, do not confirm survivor immediately.\n"
    "2) Assign a different active drone (not the detecting drone) to move_to the same cell and run thermal_scan again.\n"
    "3) Call list_active_drones and get_battery_status for candidate drones before that reassignment.\n"
    "4) If a second drone has sufficient battery margin, prioritize this second thermal re-scan before verify_survivor.\n"
    "5) If battery margins are poor, queue the unconfirmed cell and revisit with another drone ASAP.\n"
    "6) Only call verify_survivor after documenting why verification is now justified."
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


def _build_llm() -> ChatOpenAI:
    """Initialize an LLM backend for the tool-calling agent.

    Default model is GPT-4o and can be overridden with AGENT_MODEL.
    """
    model_name = os.getenv("AGENT_MODEL", "gpt-4o")
    temperature = float(os.getenv("AGENT_TEMPERATURE", "0"))
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        model_kwargs={"parallel_tool_calls": True},
    )


def create_orchestrator(server_script: str = "mcp_server.py") -> AgentExecutor:
    """Create a LangChain agent executor bound to all MCP tools."""
    tools = _run_coroutine_sync(_load_mcp_tools(server_script=server_script))
    llm = _build_llm()
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "120"))
    early_stopping_method = os.getenv("AGENT_EARLY_STOPPING_METHOD", "force")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + "\n\nShared Memory Context:\n{shared_memory_context}\n\n"
                + "Global Search Plan:\n{global_search_plan}\n\n"
                + "Decision Policy:\n{unconfirmed_triage_policy}\n\n"
                + "Mission Decomposition Policy:\n{mission_decomposition_policy}\n\n"
                + "Fleet Resource Policy:\n{fleet_resource_policy}\n\n"
                + "MCP Compliance Policy:\n{mcp_compliance_policy}\n\n"
                + "Reasoning Summary Policy:\n{reasoning_summary_policy}",
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
        max_iterations=max_iterations,
        early_stopping_method=early_stopping_method,
        return_intermediate_steps=True,
        verbose=True,
    )


class CommandAgentOrchestrator:
    """Stateful orchestrator that persists shared scan memory across turns."""

    def __init__(self, server_script: str = "mcp_server.py"):
        self.executor = create_orchestrator(server_script=server_script)
        self.shared_memory = SharedMemoryContext()

    @staticmethod
    def _extract_tool_name(action: Any) -> str | None:
        tool_name = getattr(action, "tool", None)
        if isinstance(tool_name, str) and tool_name.strip():
            return tool_name.strip()

        if isinstance(action, dict):
            maybe = action.get("tool")
            if isinstance(maybe, str) and maybe.strip():
                return maybe.strip()

        if isinstance(action, (list, tuple)):
            for item in action:
                nested = CommandAgentOrchestrator._extract_tool_name(item)
                if nested:
                    return nested

        if isinstance(action, str) and action in {
            "list_active_drones",
            "get_battery_status",
            "move_to",
            "thermal_scan",
            "verify_survivor",
            "plan_human_rescue_route",
        }:
            return action

        return None

    @staticmethod
    def _normalize_observation(observation: Any) -> Any:
        if (
            isinstance(observation, tuple)
            and len(observation) == 2
            and isinstance(observation[0], str)
            and observation[0] in {"observation", "output", "result"}
        ):
            return observation[1]
        return observation

    @staticmethod
    def _summarize_tool_io(action: Any, observation: Any) -> str:
        tool_name = CommandAgentOrchestrator._extract_tool_name(action) or "unknown_tool"
        tool_input = getattr(action, "tool_input", {})
        tool_input_text = str(tool_input)
        normalized_observation = CommandAgentOrchestrator._normalize_observation(observation)
        observation_text = str(normalized_observation)
        if len(tool_input_text) > 100:
            tool_input_text = f"{tool_input_text[:100]}..."
        if len(observation_text) > 160:
            observation_text = f"{observation_text[:160]}..."
        return (
            f"openai called {tool_name} with {tool_input_text}; "
            f"result: {observation_text}"
        )

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

    @staticmethod
    def _is_hidden_mission_log_tool(tool_name: str | None) -> bool:
        """Return whether a tool event should be hidden from mission_log SSE output."""
        return tool_name in {"move_to"}

    @staticmethod
    def _is_allowed_observation_tool(tool_name: str | None) -> bool:
        """Return whether a tool should be shown in mission_log observation events."""
        return tool_name in {
            "list_active_drones",
            "get_battery_status",
            "thermal_scan",
            "verify_survivor",
            "plan_human_rescue_route",
        }

    @staticmethod
    def _extract_chunk_thoughts(chunk: dict[str, Any]) -> list[str]:
        """Extract model-authored thought text from streamed message chunks."""
        thoughts: list[str] = []
        messages = chunk.get("messages")
        if not isinstance(messages, list):
            return thoughts

        for message in messages:
            message_type = getattr(message, "type", None)
            if message_type not in {"ai", "assistant"}:
                continue

            content = getattr(message, "content", None)
            if isinstance(content, str):
                text = content.strip()
                if (
                    text
                    and not text.startswith("Invoking:")
                    and not text.startswith("{")
                    and not text.startswith("[")
                ):
                    thoughts.append(text)
                continue

            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            cleaned = text.strip()
                            if (
                                not cleaned.startswith("Invoking:")
                                and not cleaned.startswith("{")
                                and not cleaned.startswith("[")
                            ):
                                parts.append(cleaned)
                if parts:
                    thoughts.append("\n".join(parts))

        return thoughts

    def invoke(self, user_input: str) -> dict[str, Any]:
        payload = {
            "input": user_input,
            "shared_memory_context": self.shared_memory.to_prompt_context(),
            "global_search_plan": self.shared_memory.to_global_plan(),
            "unconfirmed_triage_policy": UNCONFIRMED_TRIAGE_POLICY,
            "mission_decomposition_policy": MISSION_DECOMPOSITION_POLICY,
            "fleet_resource_policy": FLEET_RESOURCE_POLICY,
            "mcp_compliance_policy": MCP_COMPLIANCE_POLICY,
            "reasoning_summary_policy": REASONING_SUMMARY_POLICY,
        }
        result = _run_coroutine_sync(self.executor.ainvoke(payload))
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
            "unconfirmed_triage_policy": UNCONFIRMED_TRIAGE_POLICY,
            "mission_decomposition_policy": MISSION_DECOMPOSITION_POLICY,
            "fleet_resource_policy": FLEET_RESOURCE_POLICY,
            "mcp_compliance_policy": MCP_COMPLIANCE_POLICY,
            "reasoning_summary_policy": REASONING_SUMMARY_POLICY,
        }

        chunk_queue: queue.Queue[Any] = queue.Queue()
        stream_done = object()
        stream_error: dict[str, Exception] = {}

        async def _pump_chunks() -> None:
            try:
                async for chunk in self.executor.astream(payload):
                    if isinstance(chunk, dict):
                        chunk_queue.put(chunk)
            except Exception as exc:  # pragma: no cover - passthrough
                stream_error["value"] = exc
            finally:
                chunk_queue.put(stream_done)

        def _pump_runner() -> None:
            asyncio.run(_pump_chunks())

        threading.Thread(target=_pump_runner, daemon=True).start()

        collected_steps: list[Any] = []
        pending_action_tools: list[str] = []
        final_output = ""

        while True:
            if should_stop and should_stop():
                yield {
                    "event": "status",
                    "data": {
                        "message": "mission stopped by operator",
                    },
                }
                return

            try:
                item = chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is stream_done:
                break

            chunk = item

            if not isinstance(chunk, dict):
                continue

            for thought_text in self._extract_chunk_thoughts(chunk):
                yield {
                    "event": "thought",
                    "data": {
                        "label": "THOUGHT",
                        "message": thought_text,
                    },
                }

            if "actions" in chunk and isinstance(chunk["actions"], list):
                for action_item in chunk["actions"]:
                    name = self._extract_tool_name(action_item)
                    if name:
                        pending_action_tools.append(name)

            if "steps" in chunk:
                for action, observation in chunk["steps"]:
                    collected_steps.append((action, observation))
                    tool_name = self._extract_tool_name(action)
                    if not tool_name and pending_action_tools:
                        tool_name = pending_action_tools.pop(0)
                    if not tool_name:
                        tool_name = "unknown_tool"

                    normalized_observation = self._normalize_observation(observation)
                    yield {
                        "event": "tool_call",
                        "data": {
                            "tool": tool_name,
                            "message": self._summarize_tool_io(action, normalized_observation),
                        },
                    }
                    if self._is_hidden_mission_log_tool(tool_name):
                        continue
                    if not self._is_allowed_observation_tool(tool_name):
                        continue
                    yield {
                        "event": "observation",
                        "data": {
                            "tool": tool_name,
                            "observation": normalized_observation,
                        },
                    }

            if "output" in chunk:
                final_output = str(chunk["output"])

        if "value" in stream_error:
            raise stream_error["value"]

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
