from __future__ import annotations

import random
from typing import Any

from fastmcp import FastMCP

from engine import COMMAND_AGENT_ORIGIN, Drone, SwarmModel as Model


mcp = FastMCP("swarm-intelligence")

# Active simulation instance shared by MCP tools.
_active_model: Model = Model(num_drones=3)
_verified_survivors: set[tuple[int, int]] = set()


def get_active_model() -> Model:
    """Return the active simulation model instance."""
    return _active_model


def set_active_model(model: Model) -> Model:
    """Replace the active simulation model instance."""
    global _active_model
    _active_model = model
    _verified_survivors.clear()
    return _active_model


def _get_drone_by_id(drone_id: int) -> Drone:
    model = get_active_model()
    if not 0 <= drone_id < len(model.drones):
        raise ValueError(f"drone_id must be between 0 and {len(model.drones) - 1}")
    return model.drones[drone_id]


@mcp.tool()
def list_active_drones() -> list[dict[str, Any]]:
    """Discover active drone objects and return IDs plus current status.

    This scans the live simulation state at call time, so clients must call this
    tool first to discover which drone IDs are currently available.
    """
    model = get_active_model()
    active = []

    for index, drone in enumerate(model.drones):
        if drone.state != "active":
            continue

        unique_id = getattr(drone, "unique_id", index)
        active.append(
            {
                "id": int(unique_id),
                "status": drone.state,
                "position": [drone.x, drone.y],
                "battery": drone.battery,
                "mode": drone.mode,
            }
        )

    return active


@mcp.tool()
def move_to(drone_id: int, x: int, y: int) -> dict[str, int]:
    """Move a drone one step toward (x, y) and return its new coordinates."""
    model = get_active_model()
    width = model.search_grid.shape[1]
    height = model.search_grid.shape[0]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Target coordinates are outside the search grid")

    drone = _get_drone_by_id(drone_id)
    if drone.state == "failed" or drone.battery <= 0:
        raise RuntimeError("Hardware Failure")

    # A move step always costs exactly 1% battery. If this step would deplete
    # battery to 0%, report hardware failure and do not execute the move.
    if drone.battery == 1:
        drone.battery = 0
        drone.state = "failed"
        raise RuntimeError("Hardware Failure")

    battery_before = drone.battery
    new_x, new_y = drone.move_to((x, y))

    if battery_before - drone.battery != 1:
        raise RuntimeError("Battery accounting error: expected exactly 1% per move")

    return {"x": new_x, "y": new_y}


@mcp.tool()
def get_battery_status(drone_id: int) -> int:
    """Return the battery level for a specific drone."""
    drone = _get_drone_by_id(drone_id)
    return drone.battery


@mcp.tool()
def thermal_scan(drone_id: int) -> dict[str, Any]:
    """Scan the current cell with false-alarm uncertainty and verification metadata."""
    model = get_active_model()
    drone = _get_drone_by_id(drone_id)

    # Keep the placeholder method invocation for simulation flow.
    drone.thermal_scan()

    cell_weight = int(model.search_grid[drone.y, drone.x])
    real_signature_detected = cell_weight >= 3
    false_alarm = (not real_signature_detected) and (random.random() < 0.30)
    thermal_signature_detected = real_signature_detected or false_alarm

    status = "suspect" if thermal_signature_detected else "clear"
    drone.last_scan_result = status
    model.update_shared_memory((drone.x, drone.y), status)

    result: dict[str, Any] = {
        "drone_id": drone_id,
        "position": [drone.x, drone.y],
        "status": status,
        "thermal_signature_detected": thermal_signature_detected,
        "false_alarm_probability": 0.30,
        "false_alarm_triggered": false_alarm,
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


@mcp.tool()
def verify_survivor(drone_id: int) -> dict[str, Any]:
    """Run multimodal verification and confirm survivors in suspect cells only."""
    model = get_active_model()
    drone = _get_drone_by_id(drone_id)
    position = (drone.x, drone.y)

    # Verification is only valid on cells currently marked as suspect.
    cell_status = model.shared_memory.get(position, "clear")
    if cell_status != "suspect":
        return {
            "drone_id": drone_id,
            "position": [drone.x, drone.y],
            "status": "Not Confirmed",
            "reason": "Cell is not marked suspect",
            "secondary_check_passed": False,
        }

    # Simulated multimodal signals from secondary verification.
    multimodal = {
        "temperature": random.random() < 0.75,
        "sound": random.random() < 0.60,
        "shape": random.random() < 0.65,
    }
    positive_count = sum(1 for detected in multimodal.values() if detected)
    secondary_check_passed = positive_count >= 2

    status = "Confirmed Survivor" if secondary_check_passed else "Not Confirmed"
    if status == "Confirmed Survivor":
        _verified_survivors.add(position)

    return {
        "drone_id": drone_id,
        "position": [drone.x, drone.y],
        "status": status,
        "secondary_check_passed": secondary_check_passed,
        "multimodal": multimodal,
    }


@mcp.tool()
def plan_human_rescue_route(x: int, y: int) -> dict[str, Any]:
    """Plan A* rescue-team route from base (10,10) to a verified survivor."""
    model = get_active_model()
    width = model.search_grid.shape[1]
    height = model.search_grid.shape[0]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Survivor coordinates are outside the search grid")

    destination = (x, y)
    if destination not in _verified_survivors:
        raise ValueError("Route denied: survivor at this location is not verified")

    path, total_cost = model.find_path(COMMAND_AGENT_ORIGIN, destination)
    return {
        "status": "Human Rescue Route Ready",
        "from": [COMMAND_AGENT_ORIGIN[0], COMMAND_AGENT_ORIGIN[1]],
        "to": [x, y],
        "path": [[px, py] for px, py in path],
        "total_cost": int(total_cost),
        "algorithm": "A*",
    }


@mcp.tool()
def simulation_status() -> dict[str, Any]:
    """Return core state for the active simulation instance."""
    model = get_active_model()
    return {
        "round": model.round_count,
        "elapsed_minutes": model.elapsed_minutes,
        "num_drones": len(model.drones),
        "active_drones": sum(1 for d in model.drones if d.state == "active"),
    }


@mcp.tool()
def step_simulation(rounds: int = 1) -> dict[str, Any]:
    """Advance the active simulation by N rounds."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    model = get_active_model()
    for _ in range(rounds):
        model.step()

    return simulation_status()


@mcp.tool()
def list_drones() -> list[dict[str, Any]]:
    """Return per-drone snapshot data from the active model."""
    model = get_active_model()
    drones: list[Drone] = model.drones
    return [
        {
            "id": index,
            "position": [drone.x, drone.y],
            "battery": drone.battery,
            "mode": drone.mode,
            "state": drone.state,
            "last_scan_result": drone.last_scan_result,
        }
        for index, drone in enumerate(drones)
    ]


if __name__ == "__main__":
    mcp.run()
