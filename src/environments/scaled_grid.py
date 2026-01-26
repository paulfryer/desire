"""
Scaled grid environments for testing agent scalability.

Provides larger environments (50x50+) with configurable resource density
to test how agents perform as environment complexity increases.
"""


import numpy as np

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, ModeConfig
from src.environments.resources import Resource, ResourceType
from src.environments.terrain import TerrainType


class ScaledGridEnv(ForagingGridEnv):
    """
    Scaled foraging environment for testing at larger scales.

    Extends ForagingGridEnv with:
    - Support for larger grids (50x50, 100x100)
    - Configurable resource density
    - Obstacle maze generation
    - Performance optimizations for larger state spaces
    """

    def __init__(
        self,
        grid_size: int = 50,
        max_steps: int = 2000,
        initial_energy: float = 200.0,
        max_energy: float | None = None,
        energy_decay_rate: float = 0.3,
        local_view_size: int = 7,
        resource_density: float = 0.02,  # Resources per cell
        obstacle_density: float = 0.05,  # Obstacles per cell
        generate_maze: bool = False,
        maze_complexity: float = 0.5,
        mode: EnvironmentMode = EnvironmentMode.BASELINE,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        """
        Initialize scaled grid environment.

        Args:
            grid_size: Size of the square grid (e.g., 50, 100)
            max_steps: Maximum steps per episode
            initial_energy: Starting energy
            max_energy: Maximum energy (default: initial_energy)
            energy_decay_rate: Energy lost per step
            local_view_size: Radius of local observation
            resource_density: Resources per cell (0.02 = 2% cells have resources)
            obstacle_density: Obstacles per cell (0.05 = 5% cells are obstacles)
            generate_maze: Whether to generate a maze structure
            maze_complexity: Complexity of maze (0-1)
            mode: Environment mode
            render_mode: Rendering mode
            seed: Random seed
        """
        self.resource_density = resource_density
        self.obstacle_density = obstacle_density
        self.generate_maze = generate_maze
        self.maze_complexity = maze_complexity

        # Scale energy based on grid size
        if max_energy is None:
            max_energy = initial_energy * (grid_size / 20)

        # Create custom mode config for scaled environment
        n_resources = max(1, int(grid_size * grid_size * resource_density))
        mode_config = ModeConfig(
            mode=mode,
            description=f"Scaled {grid_size}x{grid_size} environment",
            num_resources=n_resources,
            resource_types=["renewable"] * n_resources,
            add_obstacles=obstacle_density > 0 or generate_maze,
        )

        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            initial_energy=initial_energy,
            max_energy=max_energy,
            energy_decay_rate=energy_decay_rate,
            local_view_size=local_view_size,
            mode=mode,
            mode_config=mode_config,
            render_mode=render_mode,
            seed=seed,
        )

    def _setup_terrain(self) -> None:
        """Set up terrain with optional maze generation."""
        if self.generate_maze:
            self._generate_maze()
        elif self.obstacle_density > 0:
            self._generate_random_obstacles()

    def _generate_random_obstacles(self) -> None:
        """Generate random obstacles based on density."""
        n_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)

        for _ in range(n_obstacles):
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)

            # Don't place obstacles at center (agent start area)
            center = self.grid_size // 2
            if abs(x - center) > 3 or abs(y - center) > 3:
                self.terrain[x, y] = TerrainType.OBSTACLE

    def _generate_maze(self) -> None:
        """Generate a maze structure using recursive division."""
        # Start with all walls
        if self.maze_complexity > 0.5:
            self._recursive_division_maze()
        else:
            self._simple_corridor_maze()

    def _recursive_division_maze(self) -> None:
        """Generate maze using recursive division algorithm."""
        # Create grid of rooms with walls between
        room_size = max(3, int(10 / (self.maze_complexity + 0.5)))

        for x in range(0, self.grid_size, room_size):
            for y in range(self.grid_size):
                if x + room_size - 1 < self.grid_size:
                    self.terrain[x + room_size - 1, y] = TerrainType.OBSTACLE

        for y in range(0, self.grid_size, room_size):
            for x in range(self.grid_size):
                if y + room_size - 1 < self.grid_size:
                    self.terrain[x, y + room_size - 1] = TerrainType.OBSTACLE

        # Create passages
        for x in range(0, self.grid_size, room_size):
            for y in range(0, self.grid_size, room_size):
                if x + room_size < self.grid_size:
                    passage_y = y + self.np_random.integers(0, room_size - 1)
                    self.terrain[x + room_size - 1, passage_y] = TerrainType.OPEN

                if y + room_size < self.grid_size:
                    passage_x = x + self.np_random.integers(0, room_size - 1)
                    self.terrain[passage_x, y + room_size - 1] = TerrainType.OPEN

    def _simple_corridor_maze(self) -> None:
        """Generate simple corridor structure."""
        # Horizontal corridors
        n_corridors = int(self.grid_size / 10)
        for i in range(n_corridors):
            y = (i + 1) * self.grid_size // (n_corridors + 1)

            for x in range(self.grid_size):
                if self.np_random.random() > 0.1:  # 90% wall
                    self.terrain[x, y] = TerrainType.OBSTACLE

            # Create gaps
            n_gaps = max(2, int(self.grid_size / 20))
            for _ in range(n_gaps):
                gap_x = self.np_random.integers(0, self.grid_size)
                self.terrain[gap_x, y] = TerrainType.OPEN

    def _setup_resources(self) -> None:
        """Set up resources with density-based placement."""
        valid_positions = self.terrain.get_passable_positions()
        self.np_random.shuffle(valid_positions)

        n_resources = self.mode_config.num_resources

        for i in range(min(n_resources, len(valid_positions))):
            pos = valid_positions[i]
            self.resources.append(Resource(
                position=np.array(pos),
                resource_type=ResourceType.RENEWABLE,
                energy_value=20.0,
            ))

    def get_complexity_metrics(self) -> dict[str, float]:
        """Get metrics characterizing environment complexity."""
        # Count terrain types
        n_obstacles = sum(
            1 for x in range(self.grid_size)
            for y in range(self.grid_size)
            if not self.terrain.is_passable(x, y)
        )

        # Compute reachability (simplified)
        n_passable = self.grid_size * self.grid_size - n_obstacles

        # Compute resource accessibility
        accessible_resources = 0
        for resource in self.resources:
            if self.terrain.is_passable(resource.x, resource.y):
                accessible_resources += 1

        return {
            "grid_size": self.grid_size,
            "total_cells": self.grid_size * self.grid_size,
            "obstacle_fraction": n_obstacles / (self.grid_size * self.grid_size),
            "passable_cells": n_passable,
            "n_resources": len(self.resources),
            "accessible_resources": accessible_resources,
            "resource_density": len(self.resources) / n_passable if n_passable > 0 else 0,
        }


class ScaledGridEnv50(ScaledGridEnv):
    """50x50 scaled environment preset."""

    def __init__(self, **kwargs):
        kwargs.setdefault("grid_size", 50)
        kwargs.setdefault("max_steps", 2000)
        kwargs.setdefault("local_view_size", 7)
        super().__init__(**kwargs)


class ScaledGridEnv100(ScaledGridEnv):
    """100x100 scaled environment preset."""

    def __init__(self, **kwargs):
        kwargs.setdefault("grid_size", 100)
        kwargs.setdefault("max_steps", 5000)
        kwargs.setdefault("local_view_size", 9)
        super().__init__(**kwargs)


class MazeGridEnv(ScaledGridEnv):
    """Maze environment preset."""

    def __init__(self, **kwargs):
        kwargs.setdefault("grid_size", 30)
        kwargs.setdefault("generate_maze", True)
        kwargs.setdefault("maze_complexity", 0.7)
        kwargs.setdefault("resource_density", 0.03)
        super().__init__(**kwargs)
