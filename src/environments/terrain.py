"""
Terrain types and grid management for the foraging environment.
"""

from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np


class TerrainType(Enum):
    """Types of terrain in the environment."""

    OPEN = auto()           # Normal traversable terrain
    OBSTACLE = auto()       # Impassable terrain
    SLOW = auto()           # Traversable but costs extra energy
    DYNAMIC_OBSTACLE = auto()  # Obstacle that can appear/disappear

    def __str__(self) -> str:
        return self.name.lower()


class TerrainGrid:
    """
    Manages the terrain grid for the environment.

    Attributes:
        size: Size of the square grid
        grid: 2D array of TerrainType values
        slow_cost: Extra energy cost for slow terrain
        dynamic_state: Current state of dynamic obstacles (True = blocked)
    """

    def __init__(
        self,
        size: int = 20,
        slow_cost: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the terrain grid.

        Args:
            size: Size of the square grid
            slow_cost: Extra energy cost for slow terrain
            rng: Random number generator
        """
        self.size = size
        self.slow_cost = slow_cost
        self.rng = rng if rng is not None else np.random.default_rng()

        # Initialize all as open terrain
        self.grid = np.full((size, size), TerrainType.OPEN, dtype=object)

        # Track dynamic obstacle states
        self.dynamic_positions: list[Tuple[int, int]] = []
        self.dynamic_blocked: dict[Tuple[int, int], bool] = {}

    def __getitem__(self, key: Tuple[int, int]) -> TerrainType:
        """Get terrain type at position."""
        x, y = key
        return self.grid[x, y]

    def __setitem__(self, key: Tuple[int, int], value: TerrainType) -> None:
        """Set terrain type at position."""
        x, y = key
        self.grid[x, y] = value

        if value == TerrainType.DYNAMIC_OBSTACLE:
            if (x, y) not in self.dynamic_positions:
                self.dynamic_positions.append((x, y))
                self.dynamic_blocked[(x, y)] = True

    def is_passable(self, x: int, y: int) -> bool:
        """Check if a position is passable."""
        if not self.in_bounds(x, y):
            return False

        terrain = self.grid[x, y]

        if terrain == TerrainType.OBSTACLE:
            return False

        if terrain == TerrainType.DYNAMIC_OBSTACLE:
            return not self.dynamic_blocked.get((x, y), True)

        return True

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if a position is within grid bounds."""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_movement_cost(self, x: int, y: int) -> float:
        """Get the extra energy cost for moving to a position."""
        if not self.in_bounds(x, y):
            return float('inf')

        terrain = self.grid[x, y]

        if terrain == TerrainType.SLOW:
            return self.slow_cost

        return 0.0

    def toggle_dynamic_obstacle(self, x: int, y: int) -> None:
        """Toggle the state of a dynamic obstacle."""
        if (x, y) in self.dynamic_blocked:
            self.dynamic_blocked[(x, y)] = not self.dynamic_blocked[(x, y)]

    def set_dynamic_obstacle(self, x: int, y: int, blocked: bool) -> None:
        """Set the state of a dynamic obstacle."""
        if (x, y) in self.dynamic_blocked:
            self.dynamic_blocked[(x, y)] = blocked

    def add_wall(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        terrain_type: TerrainType = TerrainType.OBSTACLE,
    ) -> None:
        """
        Add a wall (line of obstacles) between two points.

        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            terrain_type: Type of terrain for the wall
        """
        x1, y1 = start
        x2, y2 = end

        # Bresenham's line algorithm (simplified for grid)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        while True:
            if self.in_bounds(x, y):
                self[x, y] = terrain_type

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def add_rectangle(
        self,
        top_left: Tuple[int, int],
        width: int,
        height: int,
        terrain_type: TerrainType = TerrainType.OBSTACLE,
        filled: bool = True,
    ) -> None:
        """
        Add a rectangular area of terrain.

        Args:
            top_left: Top-left corner position
            width: Width of the rectangle
            height: Height of the rectangle
            terrain_type: Type of terrain
            filled: If True, fill the rectangle; if False, only outline
        """
        x0, y0 = top_left

        for x in range(x0, x0 + width):
            for y in range(y0, y0 + height):
                if self.in_bounds(x, y):
                    if filled or x == x0 or x == x0 + width - 1 or y == y0 or y == y0 + height - 1:
                        self[x, y] = terrain_type

    def create_corridor(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        width: int = 1,
    ) -> None:
        """
        Create a corridor of open terrain between two points.

        Args:
            start: Starting position
            end: Ending position
            width: Width of the corridor
        """
        x1, y1 = start
        x2, y2 = end

        # Simple L-shaped corridor
        # First horizontal, then vertical
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for w in range(-(width // 2), width // 2 + 1):
                if self.in_bounds(x, y1 + w):
                    self[x, y1 + w] = TerrainType.OPEN

        for y in range(min(y1, y2), max(y1, y2) + 1):
            for w in range(-(width // 2), width // 2 + 1):
                if self.in_bounds(x2 + w, y):
                    self[x2 + w, y] = TerrainType.OPEN

    def get_passable_positions(self) -> list[Tuple[int, int]]:
        """Get all passable positions on the grid."""
        positions = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_passable(x, y):
                    positions.append((x, y))
        return positions

    def to_numpy(self) -> np.ndarray:
        """
        Convert terrain grid to numeric numpy array.

        Returns:
            2D array where:
            - 0 = OPEN
            - 1 = OBSTACLE (or blocked dynamic)
            - 2 = SLOW
        """
        result = np.zeros((self.size, self.size), dtype=np.float32)

        for x in range(self.size):
            for y in range(self.size):
                terrain = self.grid[x, y]
                if terrain == TerrainType.OBSTACLE:
                    result[x, y] = 1.0
                elif terrain == TerrainType.DYNAMIC_OBSTACLE:
                    result[x, y] = 1.0 if self.dynamic_blocked.get((x, y), True) else 0.0
                elif terrain == TerrainType.SLOW:
                    result[x, y] = 0.5

        return result

    def reset(self) -> None:
        """Reset dynamic obstacles to their initial blocked state."""
        for pos in self.dynamic_positions:
            self.dynamic_blocked[pos] = True

    def get_char(self, x: int, y: int) -> str:
        """Get ASCII character for terrain at position."""
        terrain = self.grid[x, y]

        if terrain == TerrainType.OBSTACLE:
            return '#'
        elif terrain == TerrainType.DYNAMIC_OBSTACLE:
            return '#' if self.dynamic_blocked.get((x, y), True) else '.'
        elif terrain == TerrainType.SLOW:
            return '~'
        else:
            return '.'
