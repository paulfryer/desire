"""
Heuristic (rule-based) agent for baseline comparison.
"""

from enum import IntEnum
from typing import Dict, Optional, List, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent


class Rule(IntEnum):
    """Rules the heuristic agent can follow."""

    CONSUME = 0      # Currently on resource, consuming
    SEEK_VISIBLE = 1 # Moving toward visible resource
    RANDOM_WALK = 2  # No resource visible, exploring
    EMERGENCY = 3    # Energy critical, seeking known resource


class HeuristicAgent(BaseAgent):
    """
    A simple rule-based agent for baseline comparison.

    Rules (in priority order):
    1. If on resource and energy < 80%, consume
    2. If resource visible, move toward nearest
    3. If no resource visible, random walk
    4. If energy critical (<20%), move toward nearest known resource

    Internal state: One-hot encoding of current rule being followed.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        energy_consume_threshold: float = 80.0,
        energy_critical_threshold: float = 20.0,
        name: str = "heuristic",
        seed: Optional[int] = None,
    ):
        """
        Initialize the heuristic agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            energy_consume_threshold: Consume when below this energy
            energy_critical_threshold: Emergency mode below this energy
            name: Agent name
            seed: Random seed
        """
        super().__init__(observation_space, action_space, name, seed)

        self.energy_consume_threshold = energy_consume_threshold
        self.energy_critical_threshold = energy_critical_threshold

        # Track which rule is currently active
        self.current_rule = Rule.RANDOM_WALK

        # Memory of known resource locations (for emergency seeking)
        self.known_resources: List[Tuple[int, int]] = []

        # Last action for momentum in random walk
        self.last_direction: Optional[int] = None

        # Internal state dimension (one-hot of rules)
        self._internal_state_dim = len(Rule)

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Select action based on rules."""
        local_view = observation["local_view"]
        energy_normalized = observation["energy"][0]
        energy = energy_normalized * 100.0  # Denormalize

        # Update current energy
        self.current_energy = energy

        # Check for resources in view
        resource_positions = self._find_resources_in_view(local_view)

        # Check if on resource (center of view)
        center = local_view.shape[0] // 2
        on_resource = local_view[center, center, 1] > 0.5

        # Update known resources
        self._update_known_resources(resource_positions, observation)

        # Determine action based on rules
        action = self._select_action(on_resource, energy, resource_positions)

        return action

    def _find_resources_in_view(self, local_view: np.ndarray) -> List[Tuple[int, int]]:
        """Find resource positions in local view (relative to center)."""
        resources = []
        center = local_view.shape[0] // 2

        for i in range(local_view.shape[0]):
            for j in range(local_view.shape[1]):
                if local_view[i, j, 1] > 0.5:  # Active resource
                    # Relative position from center
                    rel_x = i - center
                    rel_y = j - center
                    resources.append((rel_x, rel_y))

        return resources

    def _update_known_resources(
        self,
        visible_resources: List[Tuple[int, int]],
        observation: Dict[str, np.ndarray],
    ) -> None:
        """Update memory of known resource locations."""
        # Convert relative positions to absolute
        pos_normalized = observation["position"]
        # Approximate grid size (we don't have exact value)
        grid_size = 20  # Default

        agent_x = int(pos_normalized[0] * grid_size)
        agent_y = int(pos_normalized[1] * grid_size)

        for rel_x, rel_y in visible_resources:
            abs_pos = (agent_x + rel_x, agent_y + rel_y)
            if abs_pos not in self.known_resources:
                self.known_resources.append(abs_pos)
                # Limit memory size
                if len(self.known_resources) > 20:
                    self.known_resources.pop(0)

    def _select_action(
        self,
        on_resource: bool,
        energy: float,
        resource_positions: List[Tuple[int, int]],
    ) -> int:
        """Select action based on current state and rules."""

        # Rule 4: Emergency mode
        if energy < self.energy_critical_threshold and self.known_resources:
            self.current_rule = Rule.EMERGENCY
            # Move toward nearest known resource
            return self._move_toward_known_resource()

        # Rule 1: Consume if on resource and low energy
        if on_resource and energy < self.energy_consume_threshold:
            self.current_rule = Rule.CONSUME
            return 5  # CONSUME action

        # Rule 2: Move toward visible resource
        active_resources = [(x, y) for x, y in resource_positions if (x, y) != (0, 0)]
        if active_resources:
            self.current_rule = Rule.SEEK_VISIBLE
            return self._move_toward_nearest(active_resources)

        # Check for resource at current position that we should consume
        if on_resource and energy < 95:
            self.current_rule = Rule.CONSUME
            return 5  # CONSUME

        # Rule 3: Random walk
        self.current_rule = Rule.RANDOM_WALK
        return self._random_walk()

    def _move_toward_nearest(self, positions: List[Tuple[int, int]]) -> int:
        """Move toward the nearest position."""
        if not positions:
            return self._random_walk()

        # Find nearest
        nearest = min(positions, key=lambda p: abs(p[0]) + abs(p[1]))
        rel_x, rel_y = nearest

        # Determine movement direction
        if abs(rel_x) >= abs(rel_y):
            if rel_x < 0:
                return 0  # UP
            elif rel_x > 0:
                return 1  # DOWN
        else:
            if rel_y < 0:
                return 2  # LEFT
            elif rel_y > 0:
                return 3  # RIGHT

        return 4  # STAY (already there)

    def _move_toward_known_resource(self) -> int:
        """Move toward nearest known resource (emergency mode)."""
        # For simplicity, just pick a random direction
        # In a full implementation, we'd need the current position
        if self.known_resources:
            return self._rng.integers(0, 4)
        return self._random_walk()

    def _random_walk(self) -> int:
        """Random walk with momentum."""
        # 70% chance to continue in same direction (if we have one)
        if self.last_direction is not None and self._rng.random() < 0.7:
            return self.last_direction

        # Random direction
        direction = self._rng.integers(0, 4)
        self.last_direction = direction
        return direction

    def get_internal_state(self) -> np.ndarray:
        """Get one-hot encoding of current rule."""
        state = np.zeros(self._internal_state_dim, dtype=np.float32)
        state[self.current_rule] = 1.0
        return state

    def get_internal_response(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get internal response to observation."""
        # Temporarily compute what rule would be active
        local_view = observation["local_view"]
        energy_normalized = observation["energy"][0]
        energy = energy_normalized * 100.0

        resource_positions = self._find_resources_in_view(local_view)
        center = local_view.shape[0] // 2
        on_resource = local_view[center, center, 1] > 0.5

        # Determine which rule would apply
        if energy < self.energy_critical_threshold and self.known_resources:
            rule = Rule.EMERGENCY
        elif on_resource and energy < self.energy_consume_threshold:
            rule = Rule.CONSUME
        elif any(pos != (0, 0) for pos in resource_positions):
            rule = Rule.SEEK_VISIBLE
        else:
            rule = Rule.RANDOM_WALK

        state = np.zeros(self._internal_state_dim, dtype=np.float32)
        state[rule] = 1.0
        return state

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.current_rule = Rule.RANDOM_WALK
        self.known_resources = []
        self.last_direction = None
        self.step_count = 0
        self.current_energy = 100.0

    @property
    def internal_state_dim(self) -> int:
        return self._internal_state_dim
