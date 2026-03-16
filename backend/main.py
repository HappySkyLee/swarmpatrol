from __future__ import annotations

import json
import os
import threading

from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load .env file so OPENAI_API_KEY and other secrets are available before AI init.
load_dotenv()

from engine import CELL_SIZE_METERS, SwarmModel as Model

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
    """Upgrade suspect detections to confirmed survivors from orchestrator memory."""
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


def _sync_suspect_registry_from_model() -> None:
    """Record suspect/confirmed survivor cells from simulation shared memory."""
    current_round = int(model.round_count)
    elapsed = int(model.elapsed_minutes)

    for pos, cell_status in model.shared_memory.items():
        if cell_status not in {"suspect", "survivor"}:
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
                "status": "confirmed" if cell_status == "survivor" else "suspect",
            }
            continue

        if cell_status == "survivor":
            existing["status"] = "confirmed"


def _simulation_loop() -> None:
    """Advance the Mesa model by one step every configured interval."""
    while not stop_simulation_event.wait(SIMULATION_STEP_SECONDS):
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
    }


@app.post("/start_mission")
def start_mission() -> dict[str, object]:
    """Start high-level mission planning for the command agent."""
    _ensure_simulation_running()
    active_orchestrator = _require_orchestrator()
    planning_prompt = (
        "Start high-level mission planning from base (20,20). "
        "Discover active drones and produce the initial search-and-rescue plan."
    )
    planning_result = active_orchestrator.invoke(planning_prompt)

    return {
        "message": "Command Agent at (20,20) has initiated the search-and-rescue operation.",
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
    """Return suspect and confirmed survivor coordinates discovered so far."""
    with simulation_lock:
        _sync_suspect_registry_from_model()
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
        suspect_count = sum(1 for item in enriched if item.get("status") == "suspect")

        return {
            "count": len(enriched),
            "confirmed_count": confirmed_count,
            "suspect_count": suspect_count,
            "survivors": enriched,
        }


@app.get("/grid_state")
def grid_state() -> dict[str, object]:
    """Return terrain weights and exploration status for every grid cell."""
    with simulation_lock:
        height, width = model.search_grid.shape
        status_grid = [["unvisited" for _ in range(width)] for _ in range(height)]

        for (x, y), status in model.shared_memory.items():
            if 0 <= x < width and 0 <= y < height and status in {"clear", "suspect", "survivor"}:
                status_grid[y][x] = status

        terrain_weights = model.search_grid.tolist()
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
            "3": "heavy_smoke",
        },
        "status_legend": ["unvisited", "clear", "suspect", "survivor"],
        "terrain_weights": terrain_weights,
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


def _format_sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/mission_log")
def mission_log(
    objective: str = "Start high-level mission planning from base (20,20).",
) -> StreamingResponse:
    """Stream mission reasoning/events via Server-Sent Events (SSE)."""
    _ensure_simulation_running()
    active_orchestrator = _require_orchestrator()

    def event_stream():
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
