from __future__ import annotations

import json
import os
from typing import Any
from urllib import parse, request

from fastmcp import FastMCP

mcp = FastMCP("swarm-intelligence")
BACKEND_BASE_URL = os.getenv("SWARM_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
BACKEND_API_KEY = os.getenv("SWARM_BACKEND_API_KEY", "").strip()


def _api_request(method: str, path: str, params: dict[str, Any] | None = None) -> Any:
    query = parse.urlencode(params or {})
    url = f"{BACKEND_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"

    req = request.Request(url=url, method=method.upper())
    req.add_header("Accept", "application/json")
    if BACKEND_API_KEY:
        req.add_header("X-API-Key", BACKEND_API_KEY)

    with request.urlopen(req, timeout=30) as response:
        payload = response.read().decode("utf-8")
        if not payload:
            return None
        return json.loads(payload)



@mcp.tool()
def list_active_drones() -> list[dict[str, Any]]:
    """Discover active drone objects and return IDs plus current status.

    This scans the live simulation state at call time, so clients must call this
    tool first to discover which drone IDs are currently available.
    """
    result = _api_request("GET", "/agent/list_active_drones")
    return result if isinstance(result, list) else []


@mcp.tool()
def move_to(drone_id: int, x: int, y: int) -> dict[str, int]:
    """Move a drone one step toward (x, y) and return its new coordinates."""
    result = _api_request("POST", "/agent/move_to", {"drone_id": drone_id, "x": x, "y": y})
    if not isinstance(result, dict):
        raise RuntimeError("Invalid response from backend /agent/move_to")
    return {"x": int(result.get("x", 0)), "y": int(result.get("y", 0))}


@mcp.tool()
def get_battery_status(drone_id: int) -> int:
    """Return the battery level for a specific drone."""
    result = _api_request("GET", "/agent/get_battery_status", {"drone_id": drone_id})
    return int(result)


@mcp.tool()
def thermal_scan(drone_id: int) -> dict[str, Any]:
    """Scan the current cell with false-alarm uncertainty and verification metadata."""
    result = _api_request("POST", "/agent/thermal_scan", {"drone_id": drone_id})
    if not isinstance(result, dict):
        raise RuntimeError("Invalid response from backend /agent/thermal_scan")
    return result


@mcp.tool()
def verify_survivor(drone_id: int) -> dict[str, Any]:
    """Run multimodal verification and confirm survivors in unconfirmed cells only."""
    result = _api_request("POST", "/agent/verify_survivor", {"drone_id": drone_id})
    if not isinstance(result, dict):
        raise RuntimeError("Invalid response from backend /agent/verify_survivor")
    return result


@mcp.tool()
def plan_human_rescue_route(x: int, y: int) -> dict[str, Any]:
    """Plan A* rescue-team route from base (20,20) to a verified survivor."""
    result = _api_request("GET", "/agent/plan_human_rescue_route", {"x": x, "y": y})
    if not isinstance(result, dict):
        raise RuntimeError("Invalid response from backend /agent/plan_human_rescue_route")
    return result


@mcp.tool()
def simulation_status() -> dict[str, Any]:
    """Return core state for the active simulation instance."""
    result = _api_request("GET", "/agent/simulation_status")
    if not isinstance(result, dict):
        raise RuntimeError("Invalid response from backend /agent/simulation_status")
    return result


@mcp.tool()
def step_simulation(rounds: int = 1) -> dict[str, Any]:
    """Advance the active simulation by N rounds."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    result = _api_request("POST", "/agent/step_simulation", {"rounds": rounds})
    if not isinstance(result, dict):
        raise RuntimeError("Invalid response from backend /agent/step_simulation")
    return result


@mcp.tool()
def list_drones() -> list[dict[str, Any]]:
    """Return per-drone snapshot data from the active model."""
    result = _api_request("GET", "/agent/list_drones")
    return result if isinstance(result, list) else []


if __name__ == "__main__":
    mcp.run()
