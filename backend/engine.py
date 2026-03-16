import numpy as np
import networkx as nx
from mesa import Agent, Model as MesaModel


CELL_SIZE_METERS = 100
GRID_WIDTH = 40
GRID_HEIGHT = 40
ZONE_SIZE = 20
COMMAND_AGENT_ORIGIN = (20, 20)
RECHARGE_RATE_PER_ROUND = 25
MIN_RETURN_RESERVE_BATTERY = 20
MIN_SURVIVOR_SIGNATURES = 40
MAX_SURVIVOR_SIGNATURES = 80
SECONDARY_CONFIRM_PROBABILITY = 0.5

ZoneBounds = tuple[int, int, int, int]


def _in_zone(position: tuple[int, int], zone_bounds: ZoneBounds | None) -> bool:
    if zone_bounds is None:
        return True
    x, y = position
    min_x, max_x, min_y, max_y = zone_bounds
    return min_x <= x <= max_x and min_y <= y <= max_y


class Drone(Agent):
    def __init__(
        self,
        model,
        x: int,
        y: int,
        mode: str = "exploring",
        assigned_zone: ZoneBounds | None = None,
    ):
        super().__init__(model)
        if mode not in {"exploring", "relay"}:
            raise ValueError("mode must be 'exploring' or 'relay'")

        self.x = x
        self.y = y
        self.battery = 100
        self.mode = mode
        self.state = "active"
        self.assigned_zone = assigned_zone
        self.target: tuple[int, int] | None = None
        self.last_scan_result: str | None = None

    def move_to(
        self,
        new_position: tuple[int, int],
        constrain_to_assigned_zone: bool = True,
        avoid_suspect: bool = True,
    ) -> tuple[int, int]:
        """Move one A*-planned step toward new_position and consume 1% battery."""
        if self.state == "failed":
            return self.x, self.y

        next_x, next_y = self.x, self.y
        try:
            # Drones spawn at the global origin. While outside assigned zone,
            # allow transit pathing so they can enter their sector first.
            use_zone_bounds = None
            if constrain_to_assigned_zone:
                use_zone_bounds = (
                    self.assigned_zone
                    if _in_zone((self.x, self.y), self.assigned_zone)
                    else None
                )
            path, _ = self.model.find_path(
                (self.x, self.y),
                new_position,
                zone_bounds=use_zone_bounds,
                avoid_suspect=avoid_suspect,
            )
            if len(path) > 1:
                next_x, next_y = path[1]
        except ValueError:
            # If no path is available this round, hover in place and only consume time.
            pass

        self.x, self.y = int(next_x), int(next_y)

        # One round elapses per call: deduct 1% battery, but never drop to 0%.
        self.battery = max(1, self.battery - 1)

        # Publish discovered terrain status to shared model memory.
        if hasattr(self.model, "update_shared_memory") and hasattr(self.model, "search_grid"):
            existing = self.model.shared_memory.get((self.x, self.y))
            if existing in {"survivor", "survivor_found", "confirmed"}:
                status = "survivor"
            else:
                cell_weight = int(self.model.search_grid[self.y, self.x])
                status = "suspect" if cell_weight >= 3 else "clear"
            self.model.update_shared_memory((self.x, self.y), status)

        return self.x, self.y

    def thermal_scan(self) -> str:
        """Placeholder thermal scan performed once per round."""
        result = "no_anomaly"
        self.last_scan_result = result
        return result



search_area = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=int)

# Randomly assign survivor-signature cells (weight 3) within 40-80 range.
rng = np.random.default_rng()
total_cells = GRID_WIDTH * GRID_HEIGHT
base_index = COMMAND_AGENT_ORIGIN[1] * GRID_WIDTH + COMMAND_AGENT_ORIGIN[0]
candidate_indices = np.array([idx for idx in range(total_cells) if idx != base_index])
signature_cells = int(rng.integers(MIN_SURVIVOR_SIGNATURES, MAX_SURVIVOR_SIGNATURES + 1))
signature_indices = rng.choice(candidate_indices, size=signature_cells, replace=False)
search_area.flat[signature_indices] = 3

# Keep deployment base clear.
search_area[COMMAND_AGENT_ORIGIN[1], COMMAND_AGENT_ORIGIN[0]] = 1


class SwarmModel(MesaModel):
    """Main model holding terrain and shared drone intelligence."""

    def __init__(self, grid: np.ndarray | None = None, num_drones: int = 4):
        super().__init__()
        if not 4 <= num_drones <= 5:
            raise ValueError("num_drones must be between 4 and 5")

        self.search_grid = np.array(grid, copy=True) if grid is not None else np.array(search_area, copy=True)
        self.shared_memory: dict[tuple[int, int], str] = {}
        self.round_count = 0
        self.elapsed_minutes = 0
        self.mission_phase = "searching"
        self.mission_completed = False
        self.completed_round: int | None = None
        self.completed_elapsed_minutes: int | None = None
        self.pending_secondary_checks: dict[tuple[int, int], int] = {}
        self.confirmed_survivors: set[tuple[int, int]] = set()
        self.drones: list[Drone] = []
        self.zone_assignments = self._build_zone_assignments()

        for index in range(num_drones):
            zone = self.zone_assignments[index % len(self.zone_assignments)]
            drone = Drone(
                self,
                COMMAND_AGENT_ORIGIN[0],
                COMMAND_AGENT_ORIGIN[1],
                mode="exploring",
                assigned_zone=zone,
            )
            drone.target = self._random_target(zone)
            self.drones.append(drone)

    def update_shared_memory(self, position: tuple[int, int], status: str) -> None:
        if status not in {"clear", "suspect", "survivor"}:
            raise ValueError("status must be 'clear', 'suspect', or 'survivor'")
        self.shared_memory[position] = status

    def _build_zone_assignments(self) -> list[ZoneBounds]:
        if self.search_grid.shape != (40, 40):
            raise ValueError("grid must be 40x40 to divide into four 20x20 zones")

        return [
            (0, 19, 0, 19),
            (20, 39, 0, 19),
            (0, 19, 20, 39),
            (20, 39, 20, 39),
        ]

    def _random_target(self, zone_bounds: ZoneBounds) -> tuple[int, int]:
        min_x, max_x, min_y, max_y = zone_bounds
        x = int(self.random.randrange(min_x, max_x + 1))
        y = int(self.random.randrange(min_y, max_y + 1))
        return x, y

    def _iter_zone_cells(self, zone_bounds: ZoneBounds):
        min_x, max_x, min_y, max_y = zone_bounds
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                yield (x, y)

    def _next_unvisited_zone_target(self, drone: Drone) -> tuple[int, int] | None:
        if drone.assigned_zone is None:
            return None

        unvisited = [
            pos for pos in self._iter_zone_cells(drone.assigned_zone) if pos not in self.shared_memory
        ]
        if not unvisited:
            return None

        # Prefer closest unvisited cell to improve battery efficiency.
        return min(unvisited, key=lambda pos: abs(pos[0] - drone.x) + abs(pos[1] - drone.y))

    def _steps_to_origin(self, drone: Drone) -> int:
        try:
            path, _ = self.find_path(
                (drone.x, drone.y),
                COMMAND_AGENT_ORIGIN,
                zone_bounds=None,
                avoid_suspect=False,
            )
            return max(0, len(path) - 1)
        except ValueError:
            return abs(drone.x - COMMAND_AGENT_ORIGIN[0]) + abs(drone.y - COMMAND_AGENT_ORIGIN[1])

    def _assign_next_target(self, drone: Drone) -> None:
        unvisited_target = self._next_unvisited_zone_target(drone)
        if unvisited_target is not None:
            drone.target = unvisited_target
            return

        if drone.assigned_zone is not None:
            drone.target = self._random_target(drone.assigned_zone)
        else:
            drone.target = (drone.x, drone.y)

    def _all_cells_scanned(self) -> bool:
        height, width = self.search_grid.shape
        if len(self.shared_memory) < width * height:
            return False

        # Mission search is complete only after all suspects are double-checked.
        return all(status != "suspect" for status in self.shared_memory.values())

    def _all_active_drones_at_origin(self) -> bool:
        active_drones = [drone for drone in self.drones if drone.state == "active"]
        if not active_drones:
            return True
        return all((drone.x, drone.y) == COMMAND_AGENT_ORIGIN for drone in active_drones)

    def _begin_return_to_base(self) -> None:
        self.mission_phase = "returning_to_base"
        for drone in self.drones:
            if drone.state != "active":
                continue
            drone.mode = "returning"
            drone.target = COMMAND_AGENT_ORIGIN

    def _mark_mission_complete(self) -> None:
        self.mission_phase = "completed"
        self.mission_completed = True
        self.completed_round = int(self.round_count)
        self.completed_elapsed_minutes = int(self.elapsed_minutes)

        for drone in self.drones:
            if drone.state != "active":
                continue
            drone.mode = "standby"
            drone.target = COMMAND_AGENT_ORIGIN

    def _nearest_available_second_drone(
        self,
        target: tuple[int, int],
        source_drone_index: int,
    ) -> int | None:
        candidates: list[tuple[int, int]] = []

        for index, drone in enumerate(self.drones):
            if index == source_drone_index:
                continue
            if drone.state != "active":
                continue
            if drone.mode not in {"exploring"}:
                continue

            distance = abs(drone.x - target[0]) + abs(drone.y - target[1])

            # Only select a verifier that can verify and still keep return reserve.
            steps_to_origin_after_verify = abs(target[0] - COMMAND_AGENT_ORIGIN[0]) + abs(
                target[1] - COMMAND_AGENT_ORIGIN[1]
            )
            required_battery = distance + steps_to_origin_after_verify + MIN_RETURN_RESERVE_BATTERY
            if drone.battery <= required_battery:
                continue

            candidates.append((distance, index))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _schedule_secondary_verification(
        self,
        suspect_position: tuple[int, int],
        source_drone_index: int,
    ) -> None:
        if self.mission_phase != "searching":
            return
        if self.shared_memory.get(suspect_position) != "suspect":
            return
        if suspect_position in self.pending_secondary_checks:
            return

        verifier_index = self._nearest_available_second_drone(suspect_position, source_drone_index)
        if verifier_index is None:
            return

        verifier = self.drones[verifier_index]
        verifier.mode = "verifying"
        verifier.target = suspect_position
        self.pending_secondary_checks[suspect_position] = verifier_index

    def _resolve_secondary_verification(
        self,
        verifier_index: int,
        position: tuple[int, int],
    ) -> None:
        verifier = self.drones[verifier_index]

        if self.shared_memory.get(position) != "suspect":
            self.pending_secondary_checks.pop(position, None)
            verifier.mode = "exploring"
            verifier.target = None
            return

        is_confirmed = bool(self.random.random() < SECONDARY_CONFIRM_PROBABILITY)
        if is_confirmed:
            self.update_shared_memory(position, "survivor")
            self.confirmed_survivors.add(position)
        else:
            self.update_shared_memory(position, "clear")

        self.pending_secondary_checks.pop(position, None)
        verifier.mode = "exploring"
        verifier.target = None

    def _schedule_outstanding_secondary_checks(self) -> None:
        if self.mission_phase != "searching":
            return

        suspect_positions = [
            pos for pos, status in self.shared_memory.items() if status == "suspect"
        ]
        for suspect_position in suspect_positions:
            if suspect_position in self.pending_secondary_checks:
                continue
            self._schedule_secondary_verification(suspect_position, source_drone_index=-1)

    def step(self) -> None:
        """Advance the simulation by 1 round (1 minute)."""
        if self.mission_completed:
            return

        self.round_count += 1
        self.elapsed_minutes += 1

        if self.mission_phase == "searching" and self._all_cells_scanned():
            self._begin_return_to_base()

        self._schedule_outstanding_secondary_checks()

        for drone in self.drones:
            if drone.state != "active":
                continue

            if self.mission_phase == "searching" and drone.mode != "recharging":
                required_to_base = self._steps_to_origin(drone) + MIN_RETURN_RESERVE_BATTERY
                if drone.battery <= required_to_base:
                    drone.mode = "returning"
                    drone.target = COMMAND_AGENT_ORIGIN

            if self.mission_phase == "returning_to_base":
                if (drone.x, drone.y) == COMMAND_AGENT_ORIGIN:
                    drone.mode = "standby"
                    drone.target = COMMAND_AGENT_ORIGIN
                    continue

                drone.mode = "returning"
                drone.target = COMMAND_AGENT_ORIGIN
                drone.move_to(
                    COMMAND_AGENT_ORIGIN,
                    constrain_to_assigned_zone=False,
                    avoid_suspect=False,
                )
                if (drone.x, drone.y) == COMMAND_AGENT_ORIGIN:
                    drone.mode = "standby"
                    drone.target = COMMAND_AGENT_ORIGIN
                continue

            if drone.mode == "verifying":
                if drone.target is None:
                    drone.mode = "exploring"
                    continue

                target = (int(drone.target[0]), int(drone.target[1]))
                if (drone.x, drone.y) != target:
                    drone.move_to(
                        target,
                        constrain_to_assigned_zone=False,
                        avoid_suspect=False,
                    )

                if (drone.x, drone.y) == target:
                    verifier_index = self.drones.index(drone)
                    self._resolve_secondary_verification(verifier_index, target)
                continue

            if drone.mode == "recharging":
                if (drone.x, drone.y) != COMMAND_AGENT_ORIGIN:
                    drone.mode = "returning"
                    drone.target = COMMAND_AGENT_ORIGIN
                else:
                    drone.battery = min(100, drone.battery + RECHARGE_RATE_PER_ROUND)
                    if drone.battery >= 100:
                        drone.mode = "exploring"
                        drone.target = None
                    continue

            if drone.mode != "returning":
                required_to_base = self._steps_to_origin(drone) + MIN_RETURN_RESERVE_BATTERY
                if drone.battery <= required_to_base:
                    drone.mode = "returning"
                    drone.target = COMMAND_AGENT_ORIGIN

            if drone.mode == "returning":
                if (drone.x, drone.y) == COMMAND_AGENT_ORIGIN:
                    drone.mode = "recharging"
                    drone.battery = min(100, drone.battery + RECHARGE_RATE_PER_ROUND)
                    if drone.battery >= 100:
                        drone.mode = "exploring"
                        drone.target = None
                    continue

                drone.move_to(
                    COMMAND_AGENT_ORIGIN,
                    constrain_to_assigned_zone=False,
                    avoid_suspect=False,
                )
                if (drone.x, drone.y) == COMMAND_AGENT_ORIGIN:
                    drone.mode = "recharging"
                continue

            if drone.target is None or (drone.x, drone.y) == drone.target:
                self._assign_next_target(drone)

            drone.move_to(drone.target)
            drone.thermal_scan()
            current_position = (drone.x, drone.y)
            if self.shared_memory.get(current_position) == "suspect":
                source_index = self.drones.index(drone)
                self._schedule_secondary_verification(current_position, source_index)

        if self.mission_phase == "returning_to_base" and self._all_active_drones_at_origin():
            self._mark_mission_complete()

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        zone_bounds: ZoneBounds | None = None,
        avoid_suspect: bool = True,
    ) -> tuple[list[tuple[int, int]], int]:
        return find_battery_efficient_path(
            self.search_grid,
            start,
            goal,
            self.shared_memory,
            zone_bounds=zone_bounds,
            avoid_suspect=avoid_suspect,
        )


def grid_to_graph(grid: np.ndarray) -> nx.Graph:
    """Convert a 2D terrain grid to a weighted graph.

    Each cell is a node identified by (x, y). Adjacent cells (4-neighborhood)
    are connected by weighted edges. Edge cost uses the destination cell weight,
    which models battery spent to move into that cell.
    """
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D NumPy array")

    height, width = grid.shape
    graph = nx.Graph()

    for y in range(height):
        for x in range(width):
            graph.add_node((x, y), terrain=int(grid[y, x]))

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for y in range(height):
        for x in range(width):
            for dx, dy in directions:
                nx_pos = x + dx
                ny_pos = y + dy
                if 0 <= nx_pos < width and 0 <= ny_pos < height:
                    graph.add_edge((x, y), (nx_pos, ny_pos), weight=int(grid[ny_pos, nx_pos]))

    return graph


def _manhattan_heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_battery_efficient_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    shared_memory: dict[tuple[int, int], str] | None = None,
    zone_bounds: ZoneBounds | None = None,
    avoid_suspect: bool = True,
) -> tuple[list[tuple[int, int]], int]:
    """Find the minimum-cost route between two coordinates using A*.

    Returns:
        (path, total_cost)
    """
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D NumPy array")

    height, width = grid.shape

    for point_name, (x, y) in (("start", start), ("goal", goal)):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{point_name} {x, y} is outside grid bounds")

    graph = grid_to_graph(grid)

    if zone_bounds is not None:
        min_x, max_x, min_y, max_y = zone_bounds
        outside_zone = [
            (x, y)
            for x, y in graph.nodes
            if not (min_x <= x <= max_x and min_y <= y <= max_y) and (x, y) not in {start, goal}
        ]
        graph.remove_nodes_from(outside_zone)

    if shared_memory and avoid_suspect:
        # Treat discovered suspect cells as non-traversable for all drones.
        blocked_nodes = [
            pos
            for pos, status in shared_memory.items()
            if status == "suspect" and pos not in {start, goal} and graph.has_node(pos)
        ]
        graph.remove_nodes_from(blocked_nodes)

    if not graph.has_node(start) or not graph.has_node(goal):
        raise ValueError("No traversable path: start or goal is blocked by shared memory")

    try:
        path = nx.astar_path(
            graph,
            start,
            goal,
            heuristic=_manhattan_heuristic,
            weight="weight",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise ValueError("No traversable A* path between start and goal") from exc
    total_cost = int(nx.path_weight(graph, path, weight="weight"))
    return path, total_cost
