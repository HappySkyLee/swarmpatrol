from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque

from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load .env file so GEMINI_API_KEY and other secrets are available before AI init.
load_dotenv()

from engine import CELL_SIZE_METERS, COMMAND_AGENT_ORIGIN, SwarmModel as Model

try:
    from agent_orchestrator import CommandAgentOrchestrator

    ORCHESTRATOR_IMPORT_ERROR: str | None = None
except Exception as exc:
    CommandAgentOrchestrator = None
    ORCHESTRATOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


app = FastAPI(title="Swarm Intelligence API")

# Frontend origins for local Next.js development.
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=os.getenv(
        "ALLOWED_ORIGIN_REGEX",
        r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "").strip()


@app.middleware("http")
async def require_agent_api_key(request: Request, call_next):
    if request.url.path.startswith("/agent/"):
        if not BACKEND_API_KEY:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": (
                        "Command endpoint authentication is not configured. "
                        "Set BACKEND_API_KEY to enable /agent/* routes."
                    )
                },
            )

        provided = request.headers.get("X-API-Key", "").strip()
        if provided != BACKEND_API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

    return await call_next(request)


# Shared backend objects.
model = Model(num_drones=4)
orchestrator = None
if CommandAgentOrchestrator is not None:
    try:
        orchestrator = CommandAgentOrchestrator(server_script="mcp_server.py")
    except BaseException as exc:
        ORCHESTRATOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
simulation_lock = threading.Lock()
mission_control_lock = threading.Lock()
mission_stop_event = threading.Event()
stop_simulation_event = threading.Event()
simulation_thread: threading.Thread | None = None
SIMULATION_STEP_SECONDS = float(os.getenv("SIMULATION_STEP_SECONDS", "0.3"))
survivor_registry: dict[tuple[int, int], dict[str, int | str]] = {}
MISSION_PLAN_CACHE_TTL_SECONDS = float(os.getenv("MISSION_PLAN_CACHE_TTL_SECONDS", "45"))
mission_plan_cache_lock = threading.Lock()
mission_plan_cache: dict[str, Any] = {
    "key": None,
    "result": None,
    "expires_at": 0.0,
}
openai_usage_counters: dict[str, int] = {
    "start_mission_cache_hits": 0,
    "start_mission_cache_misses": 0,
    "start_mission_invoke_calls": 0,
    "mission_stream_sessions": 0,
    "mission_stream_thought_events": 0,
}
LLM_RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("LLM_RATE_LIMIT_WINDOW_SECONDS", "60"))
LLM_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("LLM_RATE_LIMIT_MAX_REQUESTS", "1"))
LLM_DAILY_MAX_REQUESTS = int(os.getenv("LLM_DAILY_MAX_REQUESTS", "20"))
llm_rate_limit_lock = threading.Lock()
llm_request_timestamps: deque[float] = deque()
llm_daily_day_utc = time.strftime("%Y-%m-%d", time.gmtime())
llm_daily_count = 0
AI_AGENT_COMMAND_MODE = os.getenv("AI_AGENT_COMMAND_MODE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_MISSION_OBJECTIVE = os.getenv(
    "MISSION_OBJECTIVE",
    (
        f"Coordinate active drones from base ({COMMAND_AGENT_ORIGIN[0]},{COMMAND_AGENT_ORIGIN[1]}) "
        "to search, triage, and verify survivors while minimizing battery usage."
    ),
)


def _advance_agent_rounds(rounds: int = 1) -> None:
    if rounds < 1:
        return

    model.round_count += int(rounds)
    model.elapsed_minutes += int(rounds)


def _get_drone_by_id(drone_id: int):
    if not 0 <= drone_id < len(model.drones):
        raise HTTPException(
            status_code=400,
            detail=f"drone_id must be between 0 and {len(model.drones) - 1}",
        )
    return model.drones[drone_id]


def _get_llm_rate_limit_snapshot() -> dict[str, float | int]:
    global llm_daily_day_utc
    global llm_daily_count

    now = time.time()
    with llm_rate_limit_lock:
        current_day_utc = time.strftime("%Y-%m-%d", time.gmtime(now))
        if current_day_utc != llm_daily_day_utc:
            llm_daily_day_utc = current_day_utc
            llm_daily_count = 0

        while llm_request_timestamps and (now - llm_request_timestamps[0]) >= LLM_RATE_LIMIT_WINDOW_SECONDS:
            llm_request_timestamps.popleft()

        used = len(llm_request_timestamps)
        remaining = max(0, LLM_RATE_LIMIT_MAX_REQUESTS - used)
        retry_after = 0.0
        if remaining == 0 and llm_request_timestamps:
            retry_after = max(
                0.0,
                LLM_RATE_LIMIT_WINDOW_SECONDS - (now - llm_request_timestamps[0]),
            )

        daily_remaining = max(0, LLM_DAILY_MAX_REQUESTS - llm_daily_count)

    return {
        "window_seconds": float(LLM_RATE_LIMIT_WINDOW_SECONDS),
        "max_requests": int(LLM_RATE_LIMIT_MAX_REQUESTS),
        "used_requests": int(used),
        "remaining_requests": int(remaining),
        "retry_after_seconds": float(retry_after),
        "daily_day_utc": llm_daily_day_utc,
        "daily_max_requests": int(LLM_DAILY_MAX_REQUESTS),
        "daily_used_requests": int(llm_daily_count),
        "daily_remaining_requests": int(daily_remaining),
    }


def _consume_llm_rate_limit_or_raise() -> None:
    global llm_daily_day_utc
    global llm_daily_count

    now = time.time()
    with llm_rate_limit_lock:
        current_day_utc = time.strftime("%Y-%m-%d", time.gmtime(now))
        if current_day_utc != llm_daily_day_utc:
            llm_daily_day_utc = current_day_utc
            llm_daily_count = 0

        while llm_request_timestamps and (now - llm_request_timestamps[0]) >= LLM_RATE_LIMIT_WINDOW_SECONDS:
            llm_request_timestamps.popleft()

        if llm_daily_count >= LLM_DAILY_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=(
                    "LLM daily rate limit reached. "
                    f"Max {LLM_DAILY_MAX_REQUESTS} requests per UTC day."
                ),
            )

        if len(llm_request_timestamps) >= LLM_RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(
                0.0,
                LLM_RATE_LIMIT_WINDOW_SECONDS - (now - llm_request_timestamps[0]),
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    "LLM rate limit reached. "
                    f"Max {LLM_RATE_LIMIT_MAX_REQUESTS} requests per {LLM_RATE_LIMIT_WINDOW_SECONDS:.0f}s. "
                    f"Retry after {retry_after:.1f}s."
                ),
            )

        llm_request_timestamps.append(now)
    llm_daily_count += 1


def _mission_state_fingerprint() -> str:
    with simulation_lock:
        round_count = int(model.round_count)
        elapsed_minutes = int(model.elapsed_minutes)
        mission_phase = str(getattr(model, "mission_phase", "searching"))
        mission_completed = bool(getattr(model, "mission_completed", False))
        shared_memory_size = int(len(getattr(model, "shared_memory", {})))
        confirmed_survivors = int(len(getattr(model, "confirmed_survivors", set())))

    return (
        f"round={round_count}|elapsed={elapsed_minutes}|phase={mission_phase}|"
        f"completed={mission_completed}|shared={shared_memory_size}|confirmed={confirmed_survivors}"
    )


def _build_mission_plan_cache_key(planning_prompt: str) -> str:
    normalized_prompt = " ".join(planning_prompt.strip().split())
    return f"{normalized_prompt}|state={_mission_state_fingerprint()}"


def _clear_mission_plan_cache() -> None:
    with mission_plan_cache_lock:
        mission_plan_cache["key"] = None
        mission_plan_cache["result"] = None
        mission_plan_cache["expires_at"] = 0.0


def _get_cached_mission_plan(cache_key: str) -> dict[str, Any] | None:
    now = time.time()
    with mission_plan_cache_lock:
        cached_key = mission_plan_cache.get("key")
        cached_result = mission_plan_cache.get("result")
        expires_at = float(mission_plan_cache.get("expires_at", 0.0))
        if cached_key == cache_key and now < expires_at and isinstance(cached_result, dict):
            openai_usage_counters["start_mission_cache_hits"] += 1
            return dict(cached_result)

        openai_usage_counters["start_mission_cache_misses"] += 1
        return None


def _store_mission_plan_cache(cache_key: str, result: dict[str, Any]) -> None:
    with mission_plan_cache_lock:
        mission_plan_cache["key"] = cache_key
        mission_plan_cache["result"] = dict(result)
        mission_plan_cache["expires_at"] = time.time() + MISSION_PLAN_CACHE_TTL_SECONDS


def _ensure_simulation_running() -> None:
    """Start the background simulation loop if it is not already running."""
    global simulation_thread

    if simulation_thread and simulation_thread.is_alive():
        return

    stop_simulation_event.clear()
    simulation_thread = threading.Thread(target=_simulation_loop, daemon=True)
    simulation_thread.start()


def _stop_simulation_loop() -> None:
    """Stop the background simulation loop and wait briefly for shutdown."""
    global simulation_thread

    stop_simulation_event.set()
    if simulation_thread and simulation_thread.is_alive():
        simulation_thread.join(timeout=1.0)
    simulation_thread = None


def _create_orchestrator_or_capture_error() -> tuple[Any, str | None]:
    if CommandAgentOrchestrator is None:
        return None, ORCHESTRATOR_IMPORT_ERROR

    try:
        return CommandAgentOrchestrator(server_script="mcp_server.py"), None
    except BaseException as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _reset_mission_state() -> None:
    """Reset model and orchestrator so a mission can restart from clean state."""
    global model
    global orchestrator
    global ORCHESTRATOR_IMPORT_ERROR

    with mission_control_lock:
        mission_stop_event.set()

    with simulation_lock:
        model = Model(num_drones=4)

    survivor_registry.clear()
    _clear_mission_plan_cache()

    new_orchestrator, orchestrator_error = _create_orchestrator_or_capture_error()
    orchestrator = new_orchestrator
    ORCHESTRATOR_IMPORT_ERROR = orchestrator_error

    with mission_control_lock:
        mission_stop_event.clear()


def _require_orchestrator() -> Any:
    if orchestrator is None:
        reason = ORCHESTRATOR_IMPORT_ERROR or "orchestrator unavailable"
        raise HTTPException(
            status_code=503,
            detail=(
                "Mission orchestrator is unavailable. Install backend AI dependencies "
                f"and restart the API. Root cause: {reason}"
            ),
        )
    return orchestrator


def _sync_survivor_registry_from_orchestrator() -> None:
    """Upgrade unconfirmed detections to confirmed survivors from orchestrator memory."""
    if orchestrator is None:
        return

    shared_memory = getattr(orchestrator, "shared_memory", None)
    if shared_memory is None:
        return

    confirmed = getattr(shared_memory, "confirmed_survivors", set())
    if not isinstance(confirmed, set):
        return

    current_round = int(model.round_count)
    for pos in confirmed:
        if not isinstance(pos, tuple) or len(pos) != 2:
            continue
        x, y = int(pos[0]), int(pos[1])
        key = (x, y)
        existing = survivor_registry.get(key)
        if existing is None:
            survivor_registry[key] = {
                "x": x,
                "y": y,
                "detected_round": current_round,
                "detected_elapsed_minutes": int(model.elapsed_minutes),
                "status": "confirmed",
            }
            continue

        existing["status"] = "confirmed"


def _sync_unconfirmed_registry_from_model() -> None:
    """Record unconfirmed/confirmed survivor cells from simulation shared memory."""
    current_round = int(model.round_count)
    elapsed = int(model.elapsed_minutes)

    for pos, cell_status in model.shared_memory.items():
        if cell_status not in {"unconfirmed", "survivor"}:
            continue

        x, y = int(pos[0]), int(pos[1])
        key = (x, y)
        existing = survivor_registry.get(key)
        if existing is None:
            survivor_registry[key] = {
                "x": x,
                "y": y,
                "detected_round": current_round,
                "detected_elapsed_minutes": elapsed,
                "status": "confirmed" if cell_status == "survivor" else "unconfirmed",
            }
            continue

        if cell_status == "survivor":
            existing["status"] = "confirmed"


def _simulation_loop() -> None:
    """Advance the Mesa model by one step every configured interval."""
    while not stop_simulation_event.wait(SIMULATION_STEP_SECONDS):
        if AI_AGENT_COMMAND_MODE:
            continue

        with simulation_lock:
            model.step()
            if bool(getattr(model, "mission_completed", False)):
                with mission_control_lock:
                    mission_stop_event.set()
                stop_simulation_event.set()


@app.on_event("startup")
def _startup_simulation_controller() -> None:
    _ensure_simulation_running()


@app.on_event("shutdown")
def _shutdown_simulation_controller() -> None:
    with mission_control_lock:
        mission_stop_event.set()
    _stop_simulation_loop()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Swarm Intelligence backend is running"}


@app.get("/health")
def health() -> dict[str, object]:
    rate_limit = _get_llm_rate_limit_snapshot()
    with simulation_lock:
        round_count = int(model.round_count)
        elapsed_minutes = int(model.elapsed_minutes)
        mission_phase = str(getattr(model, "mission_phase", "searching"))
        mission_completed = bool(getattr(model, "mission_completed", False))
        completed_round = getattr(model, "completed_round", None)
        completed_elapsed_minutes = getattr(model, "completed_elapsed_minutes", None)

    return {
        "status": "ok",
        "model": type(model).__name__,
        "orchestrator": type(orchestrator).__name__ if orchestrator else "unavailable",
        "orchestrator_ready": orchestrator is not None,
        "orchestrator_error": ORCHESTRATOR_IMPORT_ERROR,
        "simulation_running": bool(simulation_thread and simulation_thread.is_alive()),
        "mission_phase": mission_phase,
        "mission_completed": mission_completed,
        "completed_round": int(completed_round) if isinstance(completed_round, int) else None,
        "completed_elapsed_minutes": (
            int(completed_elapsed_minutes) if isinstance(completed_elapsed_minutes, int) else None
        ),
        "round": round_count,
        "elapsed_minutes": elapsed_minutes,
        "step_interval_seconds": SIMULATION_STEP_SECONDS,
        "openai_usage": {
            **openai_usage_counters,
            "mission_plan_cache_ttl_seconds": MISSION_PLAN_CACHE_TTL_SECONDS,
            "agent_model": os.getenv("AGENT_MODEL", "gemini-2.5-flash"),
        },
        "llm_rate_limit": rate_limit,
    }


@app.post("/start_mission")
def start_mission() -> dict[str, object]:
    """Start high-level mission planning for the command agent."""
    _ensure_simulation_running()
    active_orchestrator = _require_orchestrator()
    planning_prompt = DEFAULT_MISSION_OBJECTIVE
    cache_key = _build_mission_plan_cache_key(planning_prompt)
    planning_result = _get_cached_mission_plan(cache_key)
    if planning_result is None:
        _consume_llm_rate_limit_or_raise()
        with mission_plan_cache_lock:
            openai_usage_counters["start_mission_invoke_calls"] += 1
        planning_result = active_orchestrator.invoke(planning_prompt)
        _store_mission_plan_cache(cache_key, planning_result)

    return {
        "message": (
            f"Command Agent at ({COMMAND_AGENT_ORIGIN[0]},{COMMAND_AGENT_ORIGIN[1]}) "
            "has initiated the search-and-rescue operation."
        ),
        "mission_started": True,
        "agent_output": planning_result.get("output", ""),
    }


@app.post("/stop_mission")
def stop_mission() -> dict[str, object]:
    """Signal mission streams to stop and pause the simulation loop."""
    with mission_control_lock:
        mission_stop_event.set()

    _stop_simulation_loop()

    return {
        "mission_stopped": True,
        "simulation_running": False,
        "message": "Stop signal sent and simulation paused.",
    }


@app.post("/restart_mission")
def restart_mission() -> dict[str, object]:
    """Stop active mission streams and reset model/orchestrator state."""
    _stop_simulation_loop()
    _reset_mission_state()
    _ensure_simulation_running()
    return {
        "mission_restarted": True,
        "message": "Mission state reset completed.",
        "orchestrator_ready": orchestrator is not None,
        "orchestrator_error": ORCHESTRATOR_IMPORT_ERROR,
        "simulation_running": bool(simulation_thread and simulation_thread.is_alive()),
    }


@app.get("/survivors")
def survivors() -> dict[str, object]:
    """Return unconfirmed and confirmed survivor coordinates discovered so far."""
    with simulation_lock:
        _sync_unconfirmed_registry_from_model()
        _sync_survivor_registry_from_orchestrator()
        survivors_list = sorted(
            survivor_registry.values(),
            key=lambda item: (int(item["detected_round"]), int(item["x"]), int(item["y"])),
        )

        rescue_routes = getattr(getattr(orchestrator, "shared_memory", None), "rescue_routes", {})
        route_ready_positions = {
            (int(pos[0]), int(pos[1]))
            for pos in rescue_routes.keys()
            if isinstance(pos, tuple) and len(pos) == 2
        }

        enriched = [
            {
                **item,
                "route_ready": (int(item["x"]), int(item["y"])) in route_ready_positions,
            }
            for item in survivors_list
        ]

        confirmed_count = sum(1 for item in enriched if item.get("status") == "confirmed")
        unconfirmed_count = sum(1 for item in enriched if item.get("status") == "unconfirmed")

        return {
            "count": len(enriched),
            "confirmed_count": confirmed_count,
            "unconfirmed_count": unconfirmed_count,
            "survivors": enriched,
        }


@app.get("/grid_state")
def grid_state() -> dict[str, object]:
    """Return terrain weights and exploration status for every grid cell."""
    with simulation_lock:
        height, width = model.search_grid.shape
        status_grid = [["unvisited" for _ in range(width)] for _ in range(height)]

        for (x, y), status in model.shared_memory.items():
            if 0 <= x < width and 0 <= y < height and status in {"clear", "unconfirmed", "survivor"}:
                status_grid[y][x] = status

        terrain_weights = model.search_grid.tolist()
        hazard_types = model.hazard_grid.tolist()
        round_count = int(model.round_count)
        elapsed_minutes = int(model.elapsed_minutes)
        mission_phase = str(getattr(model, "mission_phase", "searching"))
        mission_completed = bool(getattr(model, "mission_completed", False))

    return {
        "width": int(width),
        "height": int(height),
        "cell_size_meters": CELL_SIZE_METERS,
        "round": round_count,
        "elapsed_minutes": elapsed_minutes,
        "terrain_legend": {
            "1": "clear_air",
            "4": "heavy_wind",
            "6": "heavy_smoke",
        },
        "hazard_legend": ["none", "heavy_wind", "heavy_smoke"],
        "status_legend": ["unvisited", "clear", "unconfirmed", "survivor"],
        "terrain_weights": terrain_weights,
        "hazard_types": hazard_types,
        "cell_status": status_grid,
        "mission_phase": mission_phase,
        "mission_completed": mission_completed,
    }


@app.get("/drone_telemetry")
def drone_telemetry() -> list[dict[str, object]]:
    """Return live telemetry for all drones currently in simulation."""
    telemetry: list[dict[str, object]] = []

    with simulation_lock:
        for index, drone in enumerate(model.drones):
            drone_id = getattr(drone, "unique_id", index)
            telemetry.append(
                {
                    "drone_id": int(drone_id),
                    "x": int(drone.x),
                    "y": int(drone.y),
                    "battery_percentage": int(drone.battery),
                    "mode": str(drone.mode),
                }
            )

    return telemetry


@app.get("/agent/list_active_drones")
def agent_list_active_drones() -> list[dict[str, object]]:
    with simulation_lock:
        active = []
        for index, drone in enumerate(model.drones):
            if drone.state != "active":
                continue

            active.append(
                {
                    "id": int(index),
                    "status": drone.state,
                    "position": [int(drone.x), int(drone.y)],
                    "battery": int(drone.battery),
                    "mode": str(drone.mode),
                }
            )

    return active


@app.post("/agent/move_to")
def agent_move_to(drone_id: int, x: int, y: int) -> dict[str, int]:
    with simulation_lock:
        width = model.search_grid.shape[1]
        height = model.search_grid.shape[0]
        if not (0 <= x < width and 0 <= y < height):
            raise HTTPException(status_code=400, detail="Target coordinates are outside the search grid")

        drone = _get_drone_by_id(drone_id)
        if drone.state == "failed" or drone.battery <= 0:
            raise HTTPException(status_code=409, detail="Hardware Failure")

        if drone.battery == 1:
            drone.battery = 0
            drone.state = "failed"
            raise HTTPException(status_code=409, detail="Hardware Failure")

        battery_before = int(drone.battery)
        new_x, new_y = drone.move_to(
            (int(x), int(y)),
            constrain_to_assigned_zone=False,
            avoid_unconfirmed=False,
        )
        _advance_agent_rounds(1)

        if battery_before - int(drone.battery) != 1:
            raise HTTPException(
                status_code=500,
                detail="Battery accounting error: expected exactly 1% per move",
            )

    return {"x": int(new_x), "y": int(new_y)}


@app.get("/agent/get_battery_status")
def agent_get_battery_status(drone_id: int) -> int:
    with simulation_lock:
        drone = _get_drone_by_id(drone_id)
        return int(drone.battery)


@app.post("/agent/thermal_scan")
def agent_thermal_scan(drone_id: int) -> dict[str, object]:
    with simulation_lock:
        drone = _get_drone_by_id(drone_id)
        drone.thermal_scan()

        real_signature_detected = bool(model.has_survivor_signature(drone.x, drone.y))
        false_alarm = (not real_signature_detected) and (random.random() < 0.30)
        thermal_signature_detected = real_signature_detected or false_alarm

        status = "unconfirmed" if thermal_signature_detected else "clear"
        drone.last_scan_result = status
        model.update_shared_memory((drone.x, drone.y), status)
        _advance_agent_rounds(1)

    result: dict[str, object] = {
        "drone_id": int(drone_id),
        "position": [int(drone.x), int(drone.y)],
        "status": status,
        "thermal_signature_detected": thermal_signature_detected,
        "false_alarm_probability": 0.30,
        "false_alarm_triggered": bool(false_alarm),
    }

    if thermal_signature_detected:
        result["verification_required"] = True
        result["verification_note"] = (
            "Thermal signature detected. Requires multimodal verification "
            "(sound/shape) from a second drone."
        )
    else:
        result["verification_required"] = False

    return result


@app.post("/agent/verify_survivor")
def agent_verify_survivor(drone_id: int) -> dict[str, object]:
    with simulation_lock:
        drone = _get_drone_by_id(drone_id)
        position = (int(drone.x), int(drone.y))

        cell_status = model.shared_memory.get(position, "clear")
        if cell_status != "unconfirmed":
            return {
                "drone_id": int(drone_id),
                "position": [int(drone.x), int(drone.y)],
                "status": "Not Confirmed",
                "reason": "Cell is not marked unconfirmed",
                "secondary_check_passed": False,
            }

        multimodal = {
            "temperature": random.random() < 0.75,
            "sound": random.random() < 0.60,
            "shape": random.random() < 0.65,
        }
        positive_count = sum(1 for detected in multimodal.values() if detected)
        secondary_check_passed = positive_count >= 2

        status = "Confirmed Survivor" if secondary_check_passed else "Not Confirmed"
        if status == "Confirmed Survivor":
            model.update_shared_memory(position, "survivor")
            model.confirmed_survivors.add(position)
        else:
            model.update_shared_memory(position, "clear")

        _advance_agent_rounds(1)

    return {
        "drone_id": int(drone_id),
        "position": [position[0], position[1]],
        "status": status,
        "secondary_check_passed": bool(secondary_check_passed),
        "multimodal": multimodal,
    }


@app.get("/agent/plan_human_rescue_route")
def agent_plan_human_rescue_route(x: int, y: int) -> dict[str, object]:
    with simulation_lock:
        width = model.search_grid.shape[1]
        height = model.search_grid.shape[0]
        if not (0 <= x < width and 0 <= y < height):
            raise HTTPException(status_code=400, detail="Survivor coordinates are outside the search grid")

        destination = (int(x), int(y))
        if destination not in model.confirmed_survivors:
            raise HTTPException(status_code=400, detail="Route denied: survivor at this location is not verified")

        path, total_cost = model.find_path(COMMAND_AGENT_ORIGIN, destination)
        _advance_agent_rounds(1)

    return {
        "status": "Human Rescue Route Ready",
        "from": [int(COMMAND_AGENT_ORIGIN[0]), int(COMMAND_AGENT_ORIGIN[1])],
        "to": [int(x), int(y)],
        "path": [[int(px), int(py)] for px, py in path],
        "total_cost": int(total_cost),
        "algorithm": "A*",
    }


@app.get("/agent/simulation_status")
def agent_simulation_status() -> dict[str, object]:
    with simulation_lock:
        return {
            "round": int(model.round_count),
            "elapsed_minutes": int(model.elapsed_minutes),
            "num_drones": int(len(model.drones)),
            "active_drones": int(sum(1 for d in model.drones if d.state == "active")),
        }


@app.post("/agent/step_simulation")
def agent_step_simulation(rounds: int = 1) -> dict[str, object]:
    if rounds < 1:
        raise HTTPException(status_code=400, detail="rounds must be >= 1")

    with simulation_lock:
        if AI_AGENT_COMMAND_MODE:
            _advance_agent_rounds(int(rounds))
        else:
            for _ in range(int(rounds)):
                model.step()

        return {
            "round": int(model.round_count),
            "elapsed_minutes": int(model.elapsed_minutes),
            "num_drones": int(len(model.drones)),
            "active_drones": int(sum(1 for d in model.drones if d.state == "active")),
        }


@app.get("/agent/list_drones")
def agent_list_drones() -> list[dict[str, object]]:
    with simulation_lock:
        drones = []
        for index, drone in enumerate(model.drones):
            drones.append(
                {
                    "id": int(index),
                    "position": [int(drone.x), int(drone.y)],
                    "battery": int(drone.battery),
                    "mode": str(drone.mode),
                    "state": str(drone.state),
                    "last_scan_result": drone.last_scan_result,
                }
            )
    return drones


def _format_sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/mission_log")
def mission_log(
    objective: str = DEFAULT_MISSION_OBJECTIVE,
) -> StreamingResponse:
    """Stream mission reasoning/events via Server-Sent Events (SSE)."""
    _ensure_simulation_running()
    active_orchestrator = _require_orchestrator()
    _consume_llm_rate_limit_or_raise()

    def event_stream():
        with mission_plan_cache_lock:
            openai_usage_counters["mission_stream_sessions"] += 1

        session_thought_events = 0
        with mission_control_lock:
            mission_stop_event.clear()

        yield _format_sse("status", {"message": "mission_log stream started"})
        try:
            for event in active_orchestrator.stream_mission_events(
                objective,
                should_stop=mission_stop_event.is_set,
            ):
                event_name = str(event.get("event", "message"))
                event_data = event.get("data", {})
                if not isinstance(event_data, dict):
                    event_data = {"message": str(event_data)}
                if event_name == "thought":
                    session_thought_events += 1
                yield _format_sse(event_name, event_data)

                if mission_stop_event.is_set():
                    break
        except BaseException as exc:
            yield _format_sse(
                "mission_error",
                {
                    "message": "mission_log stream failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
        finally:
            with mission_plan_cache_lock:
                openai_usage_counters["mission_stream_thought_events"] += session_thought_events

        yield _format_sse("status", {"message": "mission_log stream completed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
