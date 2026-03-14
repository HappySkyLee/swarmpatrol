import numpy as np
import networkx as nx
from mesa import Agent, Model as MesaModel


CELL_SIZE_METERS = 100
GRID_WIDTH = 40
GRID_HEIGHT = 40
COMMAND_AGENT_ORIGIN = (20, 20)


class Drone(Agent):
    def __init__(self, model, x: int, y: int, mode: str = "exploring"):
        super().__init__(model)
        if mode not in {"exploring", "relay"}:
            raise ValueError("mode must be 'exploring' or 'relay'")

        self.x = x
        self.y = y
        self.battery = 100
        self.mode = mode
        self.state = "active"
        self.target: tuple[int, int] | None = None
        self.last_scan_result: str | None = None

    def move_to(self, new_position: tuple[int, int]) -> tuple[int, int]:
        """Move one grid step toward new_position and consume 1% battery per round."""
        if self.state == "failed":
            return self.x, self.y

        target_x, target_y = new_position

        # Move one step per round toward the target using Manhattan motion.
        if self.x != target_x:
            self.x += 1 if target_x > self.x else -1
        elif self.y != target_y:
            self.y += 1 if target_y > self.y else -1

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


# 40x40 search area; each cell represents 100m x 100m.
search_area = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=int)

# Randomly assign 10% of cells as heavy smoke (weight 3).
rng = np.random.default_rng()
total_cells = GRID_WIDTH * GRID_HEIGHT
hazard_cells = int(total_cells * 0.10)
hazard_indices = rng.choice(total_cells, size=hazard_cells, replace=False)
search_area.flat[hazard_indices] = 3

# Keep deployment base clear: (x=20, y=20).
search_area[COMMAND_AGENT_ORIGIN[1], COMMAND_AGENT_ORIGIN[0]] = 1


class SwarmModel(MesaModel):
    """Main model holding terrain and shared drone intelligence."""

    def __init__(self, grid: np.ndarray | None = None, num_drones: int = 3):
        super().__init__()
        if not 3 <= num_drones <= 5:
            raise ValueError("num_drones must be between 3 and 5")

        self.search_grid = np.array(grid, copy=True) if grid is not None else np.array(search_area, copy=True)
        self.shared_memory: dict[tuple[int, int], str] = {}
        self.round_count = 0
        self.elapsed_minutes = 0
        self.drones: list[Drone] = []

        for index in range(num_drones):
            mode = "relay" if index == 0 else "exploring"
            drone = Drone(self, COMMAND_AGENT_ORIGIN[0], COMMAND_AGENT_ORIGIN[1], mode=mode)
            drone.target = self._random_target()
            self.drones.append(drone)

    def update_shared_memory(self, position: tuple[int, int], status: str) -> None:
        if status not in {"clear", "suspect"}:
            raise ValueError("status must be 'clear' or 'suspect'")
        self.shared_memory[position] = status

    def _random_target(self) -> tuple[int, int]:
        x = int(self.random.randrange(0, self.search_grid.shape[1]))
        y = int(self.random.randrange(0, self.search_grid.shape[0]))
        return x, y

    def step(self) -> None:
        """Advance the simulation by 1 round (1 minute)."""
        self.round_count += 1
        self.elapsed_minutes += 1

        for drone in self.drones:
            if drone.state != "active":
                continue

            if drone.target is None or (drone.x, drone.y) == drone.target:
                drone.target = self._random_target()

            drone.move_to(drone.target)
            drone.thermal_scan()

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[list[tuple[int, int]], int]:
        return find_battery_efficient_path(self.search_grid, start, goal, self.shared_memory)


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

    if shared_memory:
        # Treat discovered suspect cells as non-traversable for all drones.
        blocked_nodes = [
            pos
            for pos, status in shared_memory.items()
            if status == "suspect" and pos not in {start, goal} and graph.has_node(pos)
        ]
        graph.remove_nodes_from(blocked_nodes)

    if not graph.has_node(start) or not graph.has_node(goal):
        raise ValueError("No traversable path: start or goal is blocked by shared memory")

    path = nx.astar_path(
        graph,
        start,
        goal,
        heuristic=_manhattan_heuristic,
        weight="weight",
    )
    total_cost = int(nx.path_weight(graph, path, weight="weight"))
    return path, total_cost
