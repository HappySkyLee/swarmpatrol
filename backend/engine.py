import numpy as np
import networkx as nx
from mesa import Agent, Model as MesaModel


CELL_SIZE_METERS = 100
GRID_WIDTH = 40
GRID_HEIGHT = 40
ZONE_SIZE = 20
COMMAND_AGENT_ORIGIN = (20, 20)
RECHARGE_RATE_PER_ROUND = 25
BATTERY_RETURN_BUFFER = 10

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

        # One round elapses per call: deduct exactly 1% battery.
        self.battery = max(0, self.battery - 1)
        if self.battery == 0:
            self.state = "failed"

        # Publish discovered terrain status to shared model memory.
        if hasattr(self.model, "update_shared_memory") and hasattr(self.model, "search_grid"):
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

# Randomly assign 10% of cells as heavy smoke (weight 3).
rng = np.random.default_rng()
total_cells = GRID_WIDTH * GRID_HEIGHT
hazard_cells = int(total_cells * 0.10)
hazard_indices = rng.choice(total_cells, size=hazard_cells, replace=False)
search_area.flat[hazard_indices] = 3

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
        if status not in {"clear", "suspect"}:
            raise ValueError("status must be 'clear' or 'suspect'")
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

    def step(self) -> None:
        """Advance the simulation by 1 round (1 minute)."""
        self.round_count += 1
        self.elapsed_minutes += 1

        for drone in self.drones:
            if drone.state != "active":
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
                required_to_base = self._steps_to_origin(drone) + BATTERY_RETURN_BUFFER
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
