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
model = Model(num_drones=3)
orchestrator = None
if CommandAgentOrchestrator is not None:
    try:
        orchestrator = CommandAgentOrchestrator(server_script="mcp_server.py")
    except BaseException as exc:
        ORCHESTRATOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
simulation_lock = threading.Lock()
stop_simulation_event = threading.Event()
simulation_thread: threading.Thread | None = None
SIMULATION_STEP_SECONDS = float(os.getenv("SIMULATION_STEP_SECONDS", "3"))


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


def _simulation_loop() -> None:
    """Advance the Mesa model by one step every configured interval."""
    while not stop_simulation_event.wait(SIMULATION_STEP_SECONDS):
        with simulation_lock:
            model.step()


@app.on_event("startup")
def _startup_simulation_controller() -> None:
    global simulation_thread
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(target=_simulation_loop, daemon=True)
    simulation_thread.start()


@app.on_event("shutdown")
def _shutdown_simulation_controller() -> None:
    stop_simulation_event.set()
    if simulation_thread and simulation_thread.is_alive():
        simulation_thread.join(timeout=1.0)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Swarm Intelligence backend is running"}


@app.get("/health")
def health() -> dict[str, object]:
    with simulation_lock:
        round_count = int(model.round_count)
        elapsed_minutes = int(model.elapsed_minutes)

    return {
        "status": "ok",
        "model": type(model).__name__,
        "orchestrator": type(orchestrator).__name__ if orchestrator else "unavailable",
        "orchestrator_ready": orchestrator is not None,
        "orchestrator_error": ORCHESTRATOR_IMPORT_ERROR,
        "round": round_count,
        "elapsed_minutes": elapsed_minutes,
        "step_interval_seconds": SIMULATION_STEP_SECONDS,
    }


@app.post("/start_mission")
def start_mission() -> dict[str, object]:
    """Start high-level mission planning for the command agent."""
    active_orchestrator = _require_orchestrator()
    planning_prompt = (
        "Start high-level mission planning from base (10,10). "
        "Discover active drones and produce the initial search-and-rescue plan."
    )
    planning_result = active_orchestrator.invoke(planning_prompt)

    return {
        "message": "Command Agent at (10,10) has initiated the search-and-rescue operation.",
        "mission_started": True,
        "agent_output": planning_result.get("output", ""),
    }


@app.get("/grid_state")
def grid_state() -> dict[str, object]:
    """Return terrain weights and exploration status for every grid cell."""
    with simulation_lock:
        height, width = model.search_grid.shape
        status_grid = [["unvisited" for _ in range(width)] for _ in range(height)]

        for (x, y), status in model.shared_memory.items():
            if 0 <= x < width and 0 <= y < height and status in {"clear", "suspect"}:
                status_grid[y][x] = status

        terrain_weights = model.search_grid.tolist()
        round_count = int(model.round_count)
        elapsed_minutes = int(model.elapsed_minutes)

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
        "status_legend": ["unvisited", "clear", "suspect"],
        "terrain_weights": terrain_weights,
        "cell_status": status_grid,
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
    objective: str = "Start high-level mission planning from base (10,10).",
) -> StreamingResponse:
    """Stream mission reasoning/events via Server-Sent Events (SSE)."""
    active_orchestrator = _require_orchestrator()

    def event_stream():
        yield _format_sse("status", {"message": "mission_log stream started"})
        for event in active_orchestrator.stream_mission_events(objective):
            event_name = str(event.get("event", "message"))
            event_data = event.get("data", {})
            if not isinstance(event_data, dict):
                event_data = {"message": str(event_data)}
            yield _format_sse(event_name, event_data)
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
