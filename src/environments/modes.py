"""
Environment modes for different test configurations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple

import numpy as np


class EnvironmentMode(Enum):
    """Available environment modes for testing."""

    BASELINE = auto()       # Standard foraging, no special conditions
    PATH_BLOCKAGE = auto()  # Path to resource becomes blocked
    EXTINCTION = auto()     # Resources gradually disappear
    PHANTOM = auto()        # Some resources are phantom (no energy)
    TRADEOFF = auto()       # Resources with different risk/reward profiles

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class ModeConfig:
    """Configuration for an environment mode."""

    mode: EnvironmentMode
    description: str

    # Resource configuration
    num_resources: int = 5
    resource_types: List[str] = field(default_factory=lambda: ["renewable"])
    resource_positions: Optional[List[Tuple[int, int]]] = None

    # Terrain configuration
    add_obstacles: bool = False
    obstacle_positions: Optional[List[Tuple[int, int]]] = None
    wall_positions: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
    dynamic_obstacle_positions: Optional[List[Tuple[int, int]]] = None

    # Agent configuration
    agent_start_position: Optional[Tuple[int, int]] = None

    # Mode-specific parameters
    blockage_step: int = 100        # When to activate path blockage
    extinction_rate: float = 0.01  # Probability of resource disappearing per step
    phantom_ratio: float = 0.3     # Ratio of phantom resources

    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


def get_mode_config(mode: EnvironmentMode, grid_size: int = 20) -> ModeConfig:
    """
    Get the configuration for a specific environment mode.

    Args:
        mode: The environment mode
        grid_size: Size of the grid

    Returns:
        ModeConfig for the specified mode
    """
    configs = {
        EnvironmentMode.BASELINE: _baseline_config(grid_size),
        EnvironmentMode.PATH_BLOCKAGE: _path_blockage_config(grid_size),
        EnvironmentMode.EXTINCTION: _extinction_config(grid_size),
        EnvironmentMode.PHANTOM: _phantom_config(grid_size),
        EnvironmentMode.TRADEOFF: _tradeoff_config(grid_size),
    }

    return configs.get(mode, _baseline_config(grid_size))


def _baseline_config(grid_size: int) -> ModeConfig:
    """Configuration for baseline mode."""
    # Distribute resources evenly across the grid
    positions = [
        (grid_size // 4, grid_size // 4),
        (3 * grid_size // 4, grid_size // 4),
        (grid_size // 2, grid_size // 2),
        (grid_size // 4, 3 * grid_size // 4),
        (3 * grid_size // 4, 3 * grid_size // 4),
    ]

    return ModeConfig(
        mode=EnvironmentMode.BASELINE,
        description="Standard foraging environment with renewable resources",
        num_resources=5,
        resource_types=["renewable"] * 5,
        resource_positions=positions,
        agent_start_position=(grid_size // 2, grid_size // 2),
    )


def _path_blockage_config(grid_size: int) -> ModeConfig:
    """
    Configuration for path blockage mode.

    Setup:
    - Agent starts on left side
    - Resource on right side
    - Horizontal wall with gap in the middle
    - Gap becomes blocked after blockage_step
    - Alternative path around the wall
    """
    center = grid_size // 2

    # Wall spans most of the grid horizontally
    wall_start = (center, 2)
    wall_end = (center, grid_size - 3)

    # Gap position (will become blocked)
    gap_y = center

    # Dynamic obstacle at the gap
    dynamic_positions = [(center, gap_y)]

    # Resource on the right side
    resource_positions = [(3 * grid_size // 4, center)]

    # Agent starts on the left side
    agent_start = (grid_size // 4, center)

    return ModeConfig(
        mode=EnvironmentMode.PATH_BLOCKAGE,
        description="Path to resource becomes blocked, testing adaptation",
        num_resources=1,
        resource_types=["renewable"],
        resource_positions=resource_positions,
        add_obstacles=True,
        wall_positions=[(wall_start, wall_end)],
        dynamic_obstacle_positions=dynamic_positions,
        agent_start_position=agent_start,
        blockage_step=100,
        extra_params={
            "gap_position": (center, gap_y),
            "alternative_paths": [
                [(center, 1), (center + 1, 1)],  # Path around top
                [(center, grid_size - 2), (center + 1, grid_size - 2)],  # Path around bottom
            ],
        },
    )


def _extinction_config(grid_size: int) -> ModeConfig:
    """
    Configuration for extinction mode.

    Resources gradually disappear, testing persistence of seeking behavior.
    """
    # Start with more resources that will deplete
    num_resources = 10

    # Distribute resources randomly (positions will be generated)
    return ModeConfig(
        mode=EnvironmentMode.EXTINCTION,
        description="Resources gradually disappear, testing extinction behavior",
        num_resources=num_resources,
        resource_types=["depleting"] * num_resources,
        resource_positions=None,  # Random placement
        agent_start_position=(grid_size // 2, grid_size // 2),
        extinction_rate=0.02,
    )


def _phantom_config(grid_size: int) -> ModeConfig:
    """
    Configuration for phantom mode.

    Some resources appear real but provide no energy, testing discrimination.
    """
    num_real = 3
    num_phantom = 2

    # Real resources
    real_positions = [
        (grid_size // 4, grid_size // 4),
        (3 * grid_size // 4, grid_size // 4),
        (grid_size // 2, 3 * grid_size // 4),
    ]

    # Phantom resources (look the same but no energy)
    phantom_positions = [
        (grid_size // 4, 3 * grid_size // 4),
        (3 * grid_size // 4, 3 * grid_size // 4),
    ]

    return ModeConfig(
        mode=EnvironmentMode.PHANTOM,
        description="Some resources are phantom (provide no energy)",
        num_resources=num_real + num_phantom,
        resource_types=["renewable"] * num_real + ["phantom"] * num_phantom,
        resource_positions=real_positions + phantom_positions,
        agent_start_position=(grid_size // 2, grid_size // 2),
        phantom_ratio=num_phantom / (num_real + num_phantom),
    )


def _tradeoff_config(grid_size: int) -> ModeConfig:
    """
    Configuration for tradeoff mode.

    Resources with different risk/reward profiles:
    - High energy resources in dangerous areas (near obstacles)
    - Low energy resources in safe areas
    """
    # Safe resources (low energy)
    safe_positions = [
        (grid_size // 4, grid_size // 2),
        (grid_size // 2, grid_size // 4),
    ]

    # Risky resources (high energy, near obstacles)
    risky_positions = [
        (3 * grid_size // 4, grid_size // 2),
        (grid_size // 2, 3 * grid_size // 4),
    ]

    # Obstacles around risky resources
    obstacle_region = [
        # Around first risky resource
        (3 * grid_size // 4 - 1, grid_size // 2 - 1),
        (3 * grid_size // 4 + 1, grid_size // 2 - 1),
        (3 * grid_size // 4 - 1, grid_size // 2 + 1),
        (3 * grid_size // 4 + 1, grid_size // 2 + 1),
        # Around second risky resource
        (grid_size // 2 - 1, 3 * grid_size // 4 - 1),
        (grid_size // 2 + 1, 3 * grid_size // 4 - 1),
        (grid_size // 2 - 1, 3 * grid_size // 4 + 1),
        (grid_size // 2 + 1, 3 * grid_size // 4 + 1),
    ]

    return ModeConfig(
        mode=EnvironmentMode.TRADEOFF,
        description="Resources with different risk/reward profiles",
        num_resources=4,
        resource_types=["renewable"] * 4,
        resource_positions=safe_positions + risky_positions,
        add_obstacles=True,
        obstacle_positions=obstacle_region,
        agent_start_position=(grid_size // 2, grid_size // 2),
        extra_params={
            "safe_energy": 10.0,
            "risky_energy": 30.0,
            "safe_indices": [0, 1],
            "risky_indices": [2, 3],
        },
    )
