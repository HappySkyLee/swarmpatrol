from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("swarm-intelligence")


def _backend_base_url() -> str:
    return os.getenv("SWARM_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _backend_api_key() -> str:
    return os.getenv("SWARM_BACKEND_API_KEY", "") or os.getenv("BACKEND_API_KEY", "")


def _backend_request(path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    url = f"{_backend_base_url()}{path}"
    body: bytes | None = None
    headers = {"Accept": "application/json"}
    api_key = _backend_api_key()
    if api_key:
        headers["X-API-Key"] = api_key

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8", "ignore")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore") if exc.fp else str(exc)
        raise RuntimeError(f"Backend request failed ({exc.code}) {path}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Backend request failed {path}: {exc}") from exc


@mcp.tool()
def list_active_drones() -> list[dict[str, Any]]:
    """Discover active drone objects and return IDs plus current status.

    This scans the live simulation state at call time, so clients must call this
    tool first to discover which drone IDs are currently available.
    """
    response = _backend_request("/agent/active_drones", method="GET")
    if not isinstance(response, list):
        raise RuntimeError("Unexpected backend response for /agent/active_drones")
    return response


@mcp.tool()
def move_to(drone_id: int, x: int, y: int) -> dict[str, int]:
    """Move a drone one step toward (x, y) and return its new coordinates."""
    response = _backend_request(
        "/agent/move_to",
        method="POST",
        payload={"drone_id": int(drone_id), "x": int(x), "y": int(y)},
    )
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected backend response for /agent/move_to")
    return {"x": int(response["x"]), "y": int(response["y"])}


@mcp.tool()
def get_battery_status(drone_id: int) -> int:
    """Return the battery level for a specific drone."""
    response = _backend_request(f"/agent/battery/{int(drone_id)}", method="GET")
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected backend response for /agent/battery")
    return int(response["battery"])


@mcp.tool()
def thermal_scan(drone_id: int) -> dict[str, Any]:
    """Scan the current cell with false-alarm uncertainty and verification metadata."""
    response = _backend_request(
        "/agent/thermal_scan",
        method="POST",
        payload={"drone_id": int(drone_id)},
    )
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected backend response for /agent/thermal_scan")
    return response


@mcp.tool()
def verify_survivor(drone_id: int) -> dict[str, Any]:
    """Run multimodal verification and confirm survivors in unconfirmed cells only."""
    response = _backend_request(
        "/agent/verify_survivor",
        method="POST",
        payload={"drone_id": int(drone_id)},
    )
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected backend response for /agent/verify_survivor")
    return response


@mcp.tool()
def plan_human_rescue_route(x: int, y: int) -> dict[str, Any]:
    """Plan A* rescue-team route from base (20,20) to a verified survivor."""
    response = _backend_request(
        "/agent/plan_human_rescue_route",
        method="POST",
        payload={"x": int(x), "y": int(y)},
    )
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected backend response for /agent/plan_human_rescue_route")
    return response


@mcp.tool()
def simulation_status() -> dict[str, Any]:
    """Return core state for the active simulation instance."""
    model = _backend_request("/health", method="GET")
    return {
        "round": int(model.get("round", 0)),
        "elapsed_minutes": int(model.get("elapsed_minutes", 0)),
        "num_drones": int(model.get("num_drones", 0)) if isinstance(model, dict) else 0,
        "active_drones": int(model.get("active_drones", 0)) if isinstance(model, dict) else 0,
    }


@mcp.tool()
def step_simulation(rounds: int = 1) -> dict[str, Any]:
    """Advance the active simulation by N rounds."""
    raise RuntimeError("step_simulation is disabled: drone actions must be MCP-commanded tool calls.")


@mcp.tool()
def list_drones() -> list[dict[str, Any]]:
    """Return per-drone snapshot data from the active model."""
    response = _backend_request("/drone_telemetry", method="GET")
    if not isinstance(response, list):
        raise RuntimeError("Unexpected backend response for /drone_telemetry")
    return [
        {
            "id": int(item.get("drone_id", 0)),
            "position": [int(item.get("x", 0)), int(item.get("y", 0))],
            "battery": int(item.get("battery_percentage", 0)),
            "mode": str(item.get("mode", "unknown")),
            "state": "active",
            "last_scan_result": None,
        }
        for item in response
        if isinstance(item, dict)
    ]


if __name__ == "__main__":
    mcp.run()
