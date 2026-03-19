from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except Exception:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI
from engine import COMMAND_AGENT_ORIGIN


COMMAND_BASE_X = int(os.getenv("COMMAND_BASE_X", str(COMMAND_AGENT_ORIGIN[0])))
COMMAND_BASE_Y = int(os.getenv("COMMAND_BASE_Y", str(COMMAND_AGENT_ORIGIN[1])))


def _first_non_empty_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _normalize_manus_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return "https://api.manus.im/v1"
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


SYSTEM_PROMPT = (
        '''You are the Command Agent for the Swarmpatrol disaster-response simulation.

Your primary responsibility is to coordinate a drone swarm safely and efficiently, and to provide accurate operator-facing technical interface guidance when asked.

=== OPERATIONAL CONTEXT ===
- Simulation world: grid-based environment with base at (20,20).
- Typical active drones: 4 to 5 drones.
- Round model: one action step advances mission time and affects battery.
- Survivor workflow: thermal detections are probabilistic; unconfirmed cells require secondary verification.

=== SOURCE OF TRUTH FOR CONTROL INTERFACE ===
Use the following as the canonical interface details for this project.

1) HTTP/REST API (backend control plane)
- Base URL: value of SWARM_BACKEND_URL (default: http://127.0.0.1:8000).
- Authentication: X-API-Key header required for /agent/* routes.
    - Backend expects BACKEND_API_KEY.
    - MCP bridge sends SWARM_BACKEND_API_KEY as X-API-Key.
- Core endpoints:
    - GET /agent/list_active_drones
    - POST /agent/move_to?drone_id=<int>&x=<int>&y=<int>
    - GET /agent/get_battery_status?drone_id=<int>
    - POST /agent/thermal_scan?drone_id=<int>
    - POST /agent/verify_survivor?drone_id=<int>
    - GET /agent/plan_human_rescue_route?x=<int>&y=<int>
    - GET /agent/simulation_status
    - POST /agent/step_simulation?rounds=<int>
    - GET /agent/list_drones

2) Command-line interface
- Backend server: uvicorn main:app --host 0.0.0.0 --port 8000
- MCP tool server: python mcp_server.py
- Recommendation: prefer MCP/Python tool functions for mission control, not raw shell commands for per-drone actions.

3) Python module / tool interface
- Module: mcp_server.py exposes callable tool functions:
    - list_active_drones()
    - move_to(drone_id, x, y)
    - get_battery_status(drone_id)
    - thermal_scan(drone_id)
    - verify_survivor(drone_id)
    - plan_human_rescue_route(x, y)
    - simulation_status()
    - step_simulation(rounds=1)
    - list_drones()

4) Simulation initialization/configuration
- Defined in engine.py:
    - COMMAND_AGENT_ORIGIN = (20, 20)
    - GRID_WIDTH = 40, GRID_HEIGHT = 40
    - SwarmModel(num_drones=4) is default backend initialization
    - Drone battery starts at 100

=== REQUIRED ANSWERING RULES ===
When asked interface questions (API/auth, CLI, Python module, config, tool names, discovery, response format), answer concretely using the above details.

Your answer must explicitly include:
1. Exact API or tool interface names.
2. Whether to use shell commands, Python scripts, or dedicated API/tool functions (prefer dedicated functions for drone actions).
3. Drone ID discovery guidance (call list_active_drones() first; do not hardcode IDs).
4. Expected response formats (JSON objects/lists from API and tool calls).

=== RESPONSE FORMAT EXPECTATIONS ===
- Keep responses concise, technical, and action-oriented.
- Do not invent endpoints, function names, auth methods, or schema fields not defined above.
- If a value is environment-dependent, state both env var name and default value when available.

=== MISSION EXECUTION POLICY ===
- Before dispatching movement or scans, reason about battery safety and verification requirements.
- Prioritize: discover active drones -> inspect battery -> explore unvisited cells -> verify unconfirmed detections -> plan rescue routes for confirmed survivors.
'''
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


@dataclass
class ManusToolAction:
    tool: str
    tool_input: dict[str, Any]
    log: str


UNCONFIRMED_TRIAGE_POLICY = (
    "Unconfirmed triage policy:\n"
    "1) After an unconfirmed thermal hit, do not confirm survivor immediately.\n"
    "2) Call list_active_drones and get_battery_status for candidate drones before deciding.\n"
    "3) If a second drone has sufficient battery margin for travel + verification + return, dispatch it to verify.\n"
    "4) If battery margins are poor or mission priority favors broader search, continue sweep and queue unconfirmed cells for later verification.\n"
    "5) Only call verify_survivor after documenting why verification is now justified."
)

HUMAN_RESCUE_ROUTING_POLICY = (
    "Human rescue routing policy:\n"
    "1) Plan route only for confirmed survivors.\n"
    "2) Before planning, ensure target cell is confirmed and stable.\n"
    "3) Use plan_human_rescue_route(x, y) with confirmed coordinates.\n"
    "4) Avoid repeated route planning for the same destination unless mission state changed."
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
    raw_tools = await client.get_tools()
    wrapped_tools: list[Any] = []

    for raw_tool in raw_tools:
        name = str(getattr(raw_tool, "name", "tool"))
        description = str(getattr(raw_tool, "description", ""))
        args_schema = getattr(raw_tool, "args_schema", None)

        def _make_sync_runner(tool_ref: Any):
            def _sync_runner(**kwargs: Any) -> Any:
                return _run_coroutine_sync(tool_ref.ainvoke(kwargs))

            return _sync_runner

        def _make_async_runner(tool_ref: Any):
            async def _async_runner(**kwargs: Any) -> Any:
                return await tool_ref.ainvoke(kwargs)

            return _async_runner

        wrapped_tools.append(
            StructuredTool.from_function(
                func=_make_sync_runner(raw_tool),
                coroutine=_make_async_runner(raw_tool),
                name=name,
                description=description,
                args_schema=args_schema,
                infer_schema=args_schema is None,
            )
        )

    return wrapped_tools


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


def _build_llm() -> Any:
    """Initialize an LLM backend for the tool-calling agent.

    Provider selection is controlled by AGENT_LLM_PROVIDER:
      - gemini (default)
      - manus (OpenAI-compatible API)
    """
    provider = os.getenv("AGENT_LLM_PROVIDER", "gemini").strip().lower()
    temperature = float(os.getenv("AGENT_TEMPERATURE", "0"))

    if provider == "gemini":
        model_name = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
        llm_api_key = _first_non_empty_env("AGENT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")

        if not llm_api_key:
            raise RuntimeError(
                "Gemini API key is missing. Set AGENT_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY."
            )

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=llm_api_key,
            model_kwargs={"parallel_tool_calls": True},
        )

    if provider == "manus":
        manus_api_key = _first_non_empty_env("MANUS_API_KEY", "AGENT_API_KEY")
        manus_base_url = _normalize_manus_base_url(
            _first_non_empty_env("MANUS_BASE_URL") or "https://api.manus.im/v1"
        )
        manus_model = _first_non_empty_env("MANUS_MODEL", "AGENT_MODEL")

        if not manus_api_key:
            raise RuntimeError(
                "Manus API key is missing. Set MANUS_API_KEY (or AGENT_API_KEY when AGENT_LLM_PROVIDER=manus)."
            )
        if not manus_model:
            raise RuntimeError(
                "Manus model is missing. Set MANUS_MODEL (or AGENT_MODEL) when AGENT_LLM_PROVIDER=manus."
            )

        return ChatOpenAI(
            model=manus_model,
            temperature=temperature,
            api_key="manus-placeholder",
            base_url=manus_base_url,
            default_headers={"API_KEY": manus_api_key},
            use_responses_api=True,
            disable_streaming=True,
        )

    raise RuntimeError(
        "Unsupported AGENT_LLM_PROVIDER value. Use 'gemini' or 'manus'."
    )


def create_orchestrator(server_script: str = "mcp_server.py") -> AgentExecutor:
    """Create a LangChain agent executor bound to all MCP tools."""
    tools = _run_coroutine_sync(_load_mcp_tools(server_script=server_script))
    llm = _build_llm()
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "2"))
    if max_iterations < 1:
        max_iterations = 1

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + "\n\nShared Memory Context:\n{shared_memory_context}\n\n"
                + "Global Search Plan:\n{global_search_plan}\n\n"
                + "Decision Policy:\n{unconfirmed_triage_policy}\n\n"
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
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        verbose=True,
    )


class CommandAgentOrchestrator:
    """Stateful orchestrator that persists shared scan memory across turns."""

    def __init__(self, server_script: str = "mcp_server.py"):
        self.provider = os.getenv("AGENT_LLM_PROVIDER", "gemini").strip().lower()
        self.executor: AgentExecutor | None = None
        self.manus_client: OpenAI | None = None
        self.manus_model: str | None = None
        self.manus_tools: list[Any] = []
        self.manus_tool_map: dict[str, Any] = {}
        self.manus_poll_interval_seconds = float(os.getenv("MANUS_POLL_INTERVAL_SECONDS", "1.5"))
        self.manus_poll_timeout_seconds = float(os.getenv("MANUS_POLL_TIMEOUT_SECONDS", "30"))

        if self.provider == "manus":
            manus_api_key = _first_non_empty_env("MANUS_API_KEY", "AGENT_API_KEY")
            manus_base_url = _normalize_manus_base_url(
                _first_non_empty_env("MANUS_BASE_URL") or "https://api.manus.im/v1"
            )
            manus_model = _first_non_empty_env("MANUS_MODEL", "AGENT_MODEL")
            if not manus_api_key:
                raise RuntimeError(
                    "Manus API key is missing. Set MANUS_API_KEY (or AGENT_API_KEY when AGENT_LLM_PROVIDER=manus)."
                )
            if not manus_model:
                raise RuntimeError(
                    "Manus model is missing. Set MANUS_MODEL (or AGENT_MODEL) when AGENT_LLM_PROVIDER=manus."
                )

            self.manus_client = OpenAI(
                base_url=manus_base_url,
                api_key="manus-placeholder",
                default_headers={"API_KEY": manus_api_key},
            )
            self.manus_model = manus_model
            self.manus_tools = _run_coroutine_sync(_load_mcp_tools(server_script=server_script))
            self.manus_tool_map = {str(getattr(tool, "name", "")): tool for tool in self.manus_tools}
        else:
            self.executor = create_orchestrator(server_script=server_script)

        self.shared_memory = SharedMemoryContext()
        self.min_request_interval_seconds = float(
            os.getenv("AGENT_MIN_REQUEST_INTERVAL_SECONDS", "12.5")
        )
        self.max_retries = int(os.getenv("AGENT_MAX_RETRIES", "4"))
        self.backoff_initial_seconds = float(os.getenv("AGENT_BACKOFF_INITIAL_SECONDS", "1.0"))
        self.backoff_max_seconds = float(os.getenv("AGENT_BACKOFF_MAX_SECONDS", "16.0"))
        self.backoff_jitter_seconds = float(os.getenv("AGENT_BACKOFF_JITTER_SECONDS", "0.25"))
        self._llm_request_lock = threading.Lock()
        self._last_llm_request_at = 0.0

    def _is_rate_limit_error(self, exc: BaseException) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True

        message = str(exc).lower()
        return (
            "429" in message
            or "rate limit" in message
            or "ratelimit" in message
            or "too many requests" in message
        )

    def _throttle_before_request(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return

        with self._llm_request_lock:
            now = time.monotonic()
            elapsed = now - self._last_llm_request_at
            wait_for = self.min_request_interval_seconds - elapsed
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._last_llm_request_at = now

    def _compute_backoff_seconds(self, attempt: int) -> float:
        # attempt starts at 1 for first retry
        base = self.backoff_initial_seconds * (2 ** max(0, attempt - 1))
        capped = min(self.backoff_max_seconds, base)
        jitter = random.uniform(0.0, max(0.0, self.backoff_jitter_seconds))
        return capped + jitter

    def _invoke_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            self._throttle_before_request()
            try:
                if self.provider == "manus":
                    result = self._invoke_manus(payload)
                else:
                    if self.executor is None:
                        raise RuntimeError("Gemini executor is not initialized")
                    result = _run_coroutine_sync(self.executor.ainvoke(payload))
                if isinstance(result, dict):
                    return result
                raise RuntimeError("Agent returned non-dict invoke result")
            except Exception as exc:
                if not self._is_rate_limit_error(exc):
                    raise
                attempt += 1
                if attempt > self.max_retries:
                    raise
                time.sleep(self._compute_backoff_seconds(attempt))

    @staticmethod
    def _normalize_steps(steps: Any) -> list[Any]:
        if isinstance(steps, list):
            return steps
        if isinstance(steps, tuple):
            return list(steps)
        return []

    def _compose_manus_prompt(self, payload: dict[str, Any]) -> str:
        user_input = str(payload.get("input", ""))
        shared_memory_context = str(payload.get("shared_memory_context", ""))
        global_plan = str(payload.get("global_search_plan", ""))
        triage_policy = str(payload.get("unconfirmed_triage_policy", ""))
        routing_policy = str(payload.get("human_rescue_routing_policy", ""))
        available_tools: list[str] = []
        for tool_name, tool in self.manus_tool_map.items():
            description = str(getattr(tool, "description", "")).strip()
            schema = getattr(tool, "tool_call_schema", None)
            args_schema = schema if isinstance(schema, dict) else {}
            available_tools.append(
                f"- {tool_name}: {description} | args_schema={json.dumps(args_schema, ensure_ascii=False)}"
            )

        tools_text = "\n".join(available_tools) if available_tools else "- No tools available"

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Shared Memory Context:\n{shared_memory_context}\n\n"
            f"Global Search Plan:\n{global_plan}\n\n"
            f"Decision Policy:\n{triage_policy}\n\n"
            f"Mission State Policy:\n{routing_policy}\n\n"
            f"Operator Objective:\n{user_input}\n\n"
            "Runtime assumptions (must follow):\n"
            "- You are already connected to the Swarmpatrol runtime through callable tools.\n"
            "- Do NOT ask for project paths, file uploads, sandbox access, setup steps, or backend URL confirmation.\n"
            "- Do NOT claim you cannot locate files; filesystem access is irrelevant for this task.\n"
            "- Use only the provided tools as your interface to the swarm.\n\n"
            "You must autonomously choose actions (move/scan/verify/route) based on your own reasoning. "
            "Do not ask the operator any questions. Do not request more input.\n\n"
            "Available callable tools:\n"
            f"{tools_text}\n\n"
            "Return ONLY valid JSON with this exact schema (no markdown, no prose outside JSON):\n"
            "{\n"
            '  "thought": "short internal reasoning",\n'
            '  "actions": [\n'
            '    {"tool": "tool_name", "args": {"arg": "value"}}\n'
            "  ],\n"
            '  "final": "operator-facing summary"\n'
            "}\n"
            "Rules:\n"
            "- Execute mission progression autonomously using tools.\n"
            "- Prefer concrete movement and scanning actions when safe.\n"
            "- Max 4 actions.\n"
            "- Every action.tool must be from the available tool list.\n"
            "- Every action.args must match tool schema.\n"
            "- If no action is needed, return actions as [].\n"
            "- Never output questions.\n"
            "- Never output requests for paths/files/sandbox/backend confirmation."
        )

    def _parse_manus_action_payload(self, output_text: str) -> dict[str, Any]:
        raw = output_text.strip()
        if not raw:
            return {"thought": "", "actions": [], "final": ""}

        candidate = raw
        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            candidate = fenced_match.group(1)
        else:
            brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if brace_match:
                candidate = brace_match.group(1)

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {
            "thought": "",
            "actions": [],
            "final": raw,
        }

    @staticmethod
    def _looks_like_filesystem_refusal(text: str) -> bool:
        lowered = text.lower()
        signals = [
            "cannot locate",
            "project files",
            "sandbox",
            "provide the path",
            "initialize/download",
            "backend is already running",
            "once i have access",
        ]
        return any(signal in lowered for signal in signals)

    def _fallback_autonomous_actions(self) -> dict[str, Any]:
        return {
            "thought": "Starting mission autonomously via tool interface; discovering drones and validating battery before movement/scan.",
            "actions": [
                {"tool": "list_active_drones", "args": {}},
            ],
            "final": "Initialized autonomous control via tool API and started drone discovery.",
        }

    def _execute_tools_from_actions(
        self,
        actions: Any,
        thought: str,
    ) -> list[tuple[ManusToolAction, Any]]:
        """Execute tool actions produced by Manus JSON output."""
        intermediate_steps: list[tuple[ManusToolAction, Any]] = []
        if not isinstance(actions, list):
            return intermediate_steps

        for action_item in actions[:4]:
            if not isinstance(action_item, dict):
                continue

            tool_name = str(action_item.get("tool", "")).strip()
            args = action_item.get("args", {})
            if not isinstance(args, dict):
                args = {}

            action = ManusToolAction(
                tool=tool_name,
                tool_input=args,
                log=thought or f"Calling {tool_name} with {args}",
            )

            tool_ref = self.manus_tool_map.get(tool_name)
            if tool_ref is None:
                observation = {
                    "error": f"Unknown tool '{tool_name}'",
                    "available_tools": sorted(self.manus_tool_map.keys()),
                }
                intermediate_steps.append((action, observation))
                continue

            try:
                observation = _run_coroutine_sync(tool_ref.ainvoke(args))
            except Exception as exc:
                observation = {
                    "error": f"Tool execution failed for '{tool_name}': {type(exc).__name__}: {exc}"
                }

            intermediate_steps.append((action, observation))

        return intermediate_steps

    def _extract_manus_output_text(self, response_obj: Any) -> str:
        output = getattr(response_obj, "output", None)
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                item_dict = item.model_dump() if hasattr(item, "model_dump") else item
                if not isinstance(item_dict, dict):
                    continue
                if item_dict.get("type") != "message":
                    continue
                if item_dict.get("role") != "assistant":
                    continue
                content = item_dict.get("content", [])
                if isinstance(content, list):
                    for content_item in content:
                        if not isinstance(content_item, dict):
                            continue
                        text = content_item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
            if parts:
                return "\n".join(parts)

        output_text = getattr(response_obj, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        return ""

    def _invoke_manus(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.manus_client is None or self.manus_model is None:
            raise RuntimeError("Manus client is not initialized")

        prompt_text = self._compose_manus_prompt(payload)
        initial = self.manus_client.responses.create(
            model=self.manus_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt_text,
                        }
                    ],
                }
            ],
        )

        task_id = getattr(initial, "id", None)
        if not isinstance(task_id, str) or not task_id.strip():
            raise RuntimeError("Manus response did not return a task id")

        deadline = time.monotonic() + max(1.0, self.manus_poll_timeout_seconds)
        current = initial
        not_found_retries = 0
        while True:
            status = str(getattr(current, "status", "")).lower()
            if status in {"completed", "error", "failed", "cancelled", "canceled"}:
                break
            if status == "pending":
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(max(0.2, self.manus_poll_interval_seconds))
            try:
                current = self.manus_client.responses.retrieve(response_id=task_id)
                not_found_retries = 0
            except Exception as exc:
                message = str(exc).lower()
                if "task not found" in message or "404" in message:
                    not_found_retries += 1
                    if not_found_retries <= 6 and time.monotonic() < deadline:
                        continue
                    break
                raise

        status = str(getattr(current, "status", "")).lower()
        metadata = getattr(current, "metadata", None)
        task_url = None
        if isinstance(metadata, dict):
            maybe_url = metadata.get("task_url")
            if isinstance(maybe_url, str) and maybe_url.strip():
                task_url = maybe_url.strip()

        output_text = self._extract_manus_output_text(current)
        if not output_text:
            if status in {"running", "pending"}:
                output_text = (
                    "Manus task is still processing asynchronously. "
                    "Use mission_log again to fetch later updates."
                )
            elif status in {"error", "failed", "cancelled", "canceled"}:
                err = getattr(current, "error", None)
                output_text = f"Manus task failed with status={status}. error={err}"
            else:
                output_text = "Manus task completed, but no assistant text output was returned."

        if task_url:
            output_text = f"{output_text}\nTask URL: {task_url}"

        parsed = self._parse_manus_action_payload(output_text)
        if self._looks_like_filesystem_refusal(output_text):
            parsed = self._fallback_autonomous_actions()
        thought = str(parsed.get("thought", "")).strip()
        actions = parsed.get("actions", [])
        final = str(parsed.get("final", "")).strip()

        intermediate_steps = self._execute_tools_from_actions(actions=actions, thought=thought)
        output_summary = final or output_text or "Manus returned no final summary."

        return {
            "output": output_summary,
            "intermediate_steps": intermediate_steps,
            "manus_task_id": task_id,
            "manus_status": status,
            "manus_reasoning": thought,
            "manus_json": parsed,
        }

    def _iter_with_retry(self, payload: dict[str, Any]):
        result = self._invoke_with_retry(payload)
        intermediate_steps = self._normalize_steps(result.get("intermediate_steps"))
        actions = [
            action
            for step in intermediate_steps
            if isinstance(step, tuple) and len(step) == 2
            for action in [step[0]]
        ]
        yield {
            "actions": actions,
            "steps": intermediate_steps,
            "output": result.get("output", ""),
            "manus_reasoning": result.get("manus_reasoning", ""),  # Pass through Manus reasoning if available
            "manus_json": result.get("manus_json", {}),
        }

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

    def _extract_action_thought(self, action: Any) -> str | None:
        log_text = getattr(action, "log", None)
        if isinstance(log_text, str) and log_text.strip():
            return log_text.strip()

        message_log = getattr(action, "message_log", None)
        if isinstance(message_log, list):
            parts: list[str] = []
            for message in message_log:
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str) and text.strip():
                                parts.append(text.strip())
            if parts:
                return "\n".join(parts)

        return None

    def invoke(self, user_input: str) -> dict[str, Any]:
        payload = {
            "input": user_input,
            "shared_memory_context": self.shared_memory.to_prompt_context(),
            "global_search_plan": self.shared_memory.to_global_plan(),
            "unconfirmed_triage_policy": UNCONFIRMED_TRIAGE_POLICY,
            "human_rescue_routing_policy": HUMAN_RESCUE_ROUTING_POLICY,
        }
        result = self._invoke_with_retry(payload)
        normalized_steps = self._normalize_steps(result.get("intermediate_steps"))
        result["intermediate_steps"] = normalized_steps
        self._ingest_tool_results(normalized_steps)
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
            "human_rescue_routing_policy": HUMAN_RESCUE_ROUTING_POLICY,
        }

        collected_steps: list[Any] = []
        final_output = ""

        for chunk in self._iter_with_retry(payload):
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

            # Emit Manus reasoning as a response event (for Manus provider only)
            if "manus_reasoning" in chunk:
                yield {
                    "event": "manus_response",
                    "data": {
                        "reasoning": chunk["manus_reasoning"],
                        "provider": "manus",
                    },
                }

            if "manus_json" in chunk and isinstance(chunk["manus_json"], dict):
                yield {
                    "event": "manus_json",
                    "data": chunk["manus_json"],
                }

            if "actions" in chunk:
                for action in chunk["actions"]:
                    tool_name = getattr(action, "tool", "unknown_tool")
                    tool_input = getattr(action, "tool_input", {})
                    thought_message = self._extract_action_thought(action)
                    if thought_message is None:
                        thought_message = f"Calling {tool_name} with {tool_input}."
                    yield {
                        "event": "thought",
                        "data": {
                            "label": "THOUGHT",
                            "message": thought_message,
                        },
                    }

            if "steps" in chunk:
                steps = self._normalize_steps(chunk.get("steps"))
                for action, observation in steps:
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
