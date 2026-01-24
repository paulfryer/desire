"""
Main foraging grid environment implementation.
"""

from enum import IntEnum
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from gymnasium import spaces

from src.environments.base_env import BaseEnvironment
from src.environments.resources import Resource, ResourceType, create_resource_cluster
from src.environments.terrain import TerrainGrid, TerrainType
from src.environments.modes import EnvironmentMode, ModeConfig, get_mode_config


class Action(IntEnum):
    """Available actions in the environment."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    CONSUME = 5


# Direction vectors for movement actions
DIRECTION_VECTORS = {
    Action.UP: np.array([-1, 0]),
    Action.DOWN: np.array([1, 0]),
    Action.LEFT: np.array([0, -1]),
    Action.RIGHT: np.array([0, 1]),
    Action.STAY: np.array([0, 0]),
}


class ForagingGridEnv(BaseEnvironment):
    """
    A grid-world foraging environment for testing behavioral signatures.

    The agent must collect resources to maintain energy and survive.
    Different modes enable testing of specific behavioral predicates.

    Observation Space:
        - Local view: (2 * local_view_size + 1, 2 * local_view_size + 1, 3) array
          - Channel 0: Terrain (0=open, 1=obstacle, 0.5=slow)
          - Channel 1: Resources (1=active, 0.5=inactive, 0=none)
          - Channel 2: Agent position marker (1 at center)
        - Energy: normalized to [0, 1]
        - Position: (x, y) normalized to [0, 1]
        - Step: normalized to [0, 1]

    Action Space:
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY, 5: CONSUME
    """

    def __init__(
        self,
        grid_size: int = 20,
        max_steps: int = 1000,
        initial_energy: float = 100.0,
        max_energy: float = 100.0,
        energy_decay_rate: float = 1.0,
        local_view_size: int = 5,
        mode: EnvironmentMode = EnvironmentMode.BASELINE,
        mode_config: Optional[ModeConfig] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the foraging grid environment.

        Args:
            grid_size: Size of the square grid
            max_steps: Maximum steps per episode
            initial_energy: Starting energy
            max_energy: Maximum energy
            energy_decay_rate: Energy lost per step
            local_view_size: Radius of local observation window
            mode: Environment mode for test configuration
            mode_config: Custom mode configuration (overrides mode)
            render_mode: How to render
            seed: Random seed
        """
        self.local_view_size = local_view_size
        self.mode = mode

        # Get mode configuration
        if mode_config is not None:
            self.mode_config = mode_config
        else:
            self.mode_config = get_mode_config(mode, grid_size)

        # Initialize base (will call _setup_spaces)
        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            initial_energy=initial_energy,
            max_energy=max_energy,
            energy_decay_rate=energy_decay_rate,
            render_mode=render_mode,
            seed=seed,
        )

        # Initialize terrain and resources
        self.terrain = TerrainGrid(grid_size, rng=self.np_random)
        self.resources: List[Resource] = []

        # Tracking for mode-specific behavior
        self.path_blocked = False
        self.resources_consumed = 0

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces."""
        self.action_space = spaces.Discrete(len(Action))

        # Local view dimensions
        view_dim = 2 * self.local_view_size + 1

        # Observation is a dict with different components
        self.observation_space = spaces.Dict({
            "local_view": spaces.Box(
                low=0, high=1,
                shape=(view_dim, view_dim, 3),
                dtype=np.float32,
            ),
            "energy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "step": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

    def _reset_state(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Reset environment-specific state."""
        Resource.reset_id_counter()

        # Reset terrain
        self.terrain = TerrainGrid(self.grid_size, rng=self.np_random)
        self._setup_terrain()

        # Reset resources
        self.resources = []
        self._setup_resources()

        # Reset agent position
        if self.mode_config.agent_start_position is not None:
            self.agent_position = np.array(self.mode_config.agent_start_position)
        else:
            # Random valid position
            valid_positions = self.terrain.get_passable_positions()
            # Exclude resource positions
            resource_positions = {tuple(r.position) for r in self.resources}
            valid_positions = [p for p in valid_positions if p not in resource_positions]

            if valid_positions:
                idx = self.np_random.integers(len(valid_positions))
                self.agent_position = np.array(valid_positions[idx])
            else:
                self.agent_position = np.array([self.grid_size // 2, self.grid_size // 2])

        # Reset mode-specific state
        self.path_blocked = False
        self.resources_consumed = 0

    def _setup_terrain(self) -> None:
        """Set up terrain based on mode configuration."""
        config = self.mode_config

        # Add walls
        if config.wall_positions:
            for start, end in config.wall_positions:
                self.terrain.add_wall(start, end, TerrainType.OBSTACLE)

        # Add individual obstacles
        if config.obstacle_positions:
            for pos in config.obstacle_positions:
                self.terrain[pos] = TerrainType.OBSTACLE

        # Add dynamic obstacles
        if config.dynamic_obstacle_positions:
            for pos in config.dynamic_obstacle_positions:
                self.terrain[pos] = TerrainType.DYNAMIC_OBSTACLE
                # Start unblocked for path_blockage mode
                if self.mode == EnvironmentMode.PATH_BLOCKAGE:
                    self.terrain.set_dynamic_obstacle(pos[0], pos[1], blocked=False)

    def _setup_resources(self) -> None:
        """Set up resources based on mode configuration."""
        config = self.mode_config

        if config.resource_positions is not None:
            # Fixed positions
            for i, pos in enumerate(config.resource_positions):
                resource_type_name = config.resource_types[i] if i < len(config.resource_types) else "renewable"
                resource_type = ResourceType[resource_type_name.upper()]

                # Handle tradeoff mode energy values
                energy_value = 20.0
                if self.mode == EnvironmentMode.TRADEOFF:
                    extra = config.extra_params
                    if i in extra.get("safe_indices", []):
                        energy_value = extra.get("safe_energy", 10.0)
                    elif i in extra.get("risky_indices", []):
                        energy_value = extra.get("risky_energy", 30.0)

                self.resources.append(Resource(
                    position=np.array(pos),
                    resource_type=resource_type,
                    energy_value=energy_value,
                ))
        else:
            # Random positions
            valid_positions = self.terrain.get_passable_positions()
            self.np_random.shuffle(valid_positions)

            for i in range(min(config.num_resources, len(valid_positions))):
                resource_type_name = config.resource_types[i] if i < len(config.resource_types) else "renewable"
                resource_type = ResourceType[resource_type_name.upper()]

                self.resources.append(Resource(
                    position=np.array(valid_positions[i]),
                    resource_type=resource_type,
                ))

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        # Build local view
        local_view = self._get_local_view()

        # Normalized values
        energy = np.array([self.agent_energy / self.max_energy], dtype=np.float32)
        position = self.agent_position.astype(np.float32) / self.grid_size
        step = np.array([self.current_step / self.max_steps], dtype=np.float32)

        return {
            "local_view": local_view,
            "energy": energy,
            "position": position,
            "step": step,
        }

    def _get_local_view(self) -> np.ndarray:
        """Get the local observation view around the agent."""
        view_dim = 2 * self.local_view_size + 1
        local_view = np.zeros((view_dim, view_dim, 3), dtype=np.float32)

        ax, ay = self.agent_position

        for dx in range(-self.local_view_size, self.local_view_size + 1):
            for dy in range(-self.local_view_size, self.local_view_size + 1):
                gx, gy = ax + dx, ay + dy
                vx, vy = dx + self.local_view_size, dy + self.local_view_size

                if self.terrain.in_bounds(gx, gy):
                    # Channel 0: Terrain
                    if not self.terrain.is_passable(gx, gy):
                        local_view[vx, vy, 0] = 1.0
                    elif self.terrain[gx, gy] == TerrainType.SLOW:
                        local_view[vx, vy, 0] = 0.5

                    # Channel 1: Resources
                    for resource in self.resources:
                        if resource.x == gx and resource.y == gy:
                            local_view[vx, vy, 1] = 1.0 if resource.is_active else 0.5
                            break
                else:
                    # Out of bounds = obstacle
                    local_view[vx, vy, 0] = 1.0

        # Channel 2: Agent position (center)
        local_view[self.local_view_size, self.local_view_size, 2] = 1.0

        return local_view

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary."""
        # Find nearest active resource
        nearest_resource = None
        nearest_distance = float('inf')

        for resource in self.resources:
            if resource.is_active:
                dist = resource.distance_to(self.agent_position)
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_resource = resource

        # Check if on resource
        on_resource = any(
            r.is_active and tuple(r.position) == tuple(self.agent_position)
            for r in self.resources
        )

        return {
            "position": tuple(self.agent_position),
            "energy": self.agent_energy,
            "step": self.current_step,
            "on_resource": on_resource,
            "nearest_resource_distance": nearest_distance if nearest_resource else None,
            "active_resources": sum(1 for r in self.resources if r.is_active),
            "path_blocked": self.path_blocked,
            "resources_consumed": self.resources_consumed,
        }

    def _apply_action(self, action: int) -> float:
        """Apply an action and return the reward."""
        reward = 0.0
        action = Action(action)

        if action == Action.CONSUME:
            # Try to consume a resource
            for resource in self.resources:
                if resource.is_active and tuple(resource.position) == tuple(self.agent_position):
                    energy_gained = resource.consume()
                    self.agent_energy = min(self.max_energy, self.agent_energy + energy_gained)
                    self.resources_consumed += 1

                    if energy_gained > 0:
                        reward += energy_gained / 10.0  # Scale reward
                    break
        elif action in DIRECTION_VECTORS:
            # Movement action
            direction = DIRECTION_VECTORS[action]
            new_position = self.agent_position + direction

            if self.terrain.is_passable(new_position[0], new_position[1]):
                # Apply movement cost
                movement_cost = self.terrain.get_movement_cost(new_position[0], new_position[1])
                self.agent_energy -= movement_cost

                self.agent_position = new_position

        # Update resources
        for resource in self.resources:
            resource.update()

        # Handle mode-specific updates
        self._update_mode_state()

        # Small reward for survival
        reward += 0.01

        return reward

    def _update_mode_state(self) -> None:
        """Update mode-specific state."""
        if self.mode == EnvironmentMode.PATH_BLOCKAGE:
            # Block path after specified step
            if not self.path_blocked and self.current_step >= self.mode_config.blockage_step:
                self.path_blocked = True
                if self.mode_config.dynamic_obstacle_positions:
                    for pos in self.mode_config.dynamic_obstacle_positions:
                        self.terrain.set_dynamic_obstacle(pos[0], pos[1], blocked=True)

        elif self.mode == EnvironmentMode.EXTINCTION:
            # Randomly remove resources
            for resource in self.resources:
                if resource.is_active and resource.resource_type == ResourceType.DEPLETING:
                    if self.np_random.random() < self.mode_config.extinction_rate:
                        resource.is_active = False

    def _render_ansi(self) -> str:
        """Render as ASCII string."""
        lines = []

        # Header
        lines.append(f"Step: {self.current_step}/{self.max_steps} | Energy: {self.agent_energy:.1f}")
        lines.append("-" * (self.grid_size + 2))

        # Grid
        for x in range(self.grid_size):
            row = "|"
            for y in range(self.grid_size):
                if (x, y) == tuple(self.agent_position):
                    row += "A"
                elif any(r.is_active and r.x == x and r.y == y for r in self.resources):
                    row += "R"
                elif any(not r.is_active and r.x == x and r.y == y for r in self.resources):
                    row += "r"
                else:
                    row += self.terrain.get_char(x, y)
            row += "|"
            lines.append(row)

        lines.append("-" * (self.grid_size + 2))

        return "\n".join(lines)

    def _render_rgb(self) -> np.ndarray:
        """Render as RGB array."""
        # Create RGB image
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Colors
        OPEN = [200, 200, 200]
        OBSTACLE = [50, 50, 50]
        SLOW = [150, 150, 200]
        RESOURCE_ACTIVE = [0, 200, 0]
        RESOURCE_INACTIVE = [100, 150, 100]
        AGENT = [255, 0, 0]

        # Draw terrain
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if not self.terrain.is_passable(x, y):
                    img[x, y] = OBSTACLE
                elif self.terrain[x, y] == TerrainType.SLOW:
                    img[x, y] = SLOW
                else:
                    img[x, y] = OPEN

        # Draw resources
        for resource in self.resources:
            if resource.is_active:
                img[resource.x, resource.y] = RESOURCE_ACTIVE
            else:
                img[resource.x, resource.y] = RESOURCE_INACTIVE

        # Draw agent
        ax, ay = self.agent_position
        img[ax, ay] = AGENT

        return img

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get the complete environment state for analysis.

        Returns full state including all resources, terrain, etc.
        """
        return {
            "agent_position": tuple(self.agent_position),
            "agent_energy": self.agent_energy,
            "current_step": self.current_step,
            "resources": [
                {
                    "position": tuple(r.position),
                    "type": str(r.resource_type),
                    "active": r.is_active,
                    "energy_value": r.energy_value,
                }
                for r in self.resources
            ],
            "terrain": self.terrain.to_numpy(),
            "path_blocked": self.path_blocked,
            "mode": str(self.mode),
        }

    def set_path_blocked(self, blocked: bool) -> None:
        """Manually set path blockage (for testing)."""
        self.path_blocked = blocked
        if self.mode_config.dynamic_obstacle_positions:
            for pos in self.mode_config.dynamic_obstacle_positions:
                self.terrain.set_dynamic_obstacle(pos[0], pos[1], blocked=blocked)

    def get_resource_at(self, position: Tuple[int, int]) -> Optional[Resource]:
        """Get resource at a position if any."""
        for resource in self.resources:
            if tuple(resource.position) == position:
                return resource
        return None

    def get_visible_resources(self) -> List[Resource]:
        """Get all resources visible in the local view."""
        visible = []
        ax, ay = self.agent_position

        for resource in self.resources:
            dx = abs(resource.x - ax)
            dy = abs(resource.y - ay)

            if dx <= self.local_view_size and dy <= self.local_view_size:
                visible.append(resource)

        return visible
