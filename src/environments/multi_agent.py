"""
Multi-agent foraging environment.

Supports competitive and cooperative multi-agent scenarios for testing
social behavior and resource competition.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from src.environments.foraging_grid import ForagingGridEnv, Action, DIRECTION_VECTORS
from src.environments.modes import EnvironmentMode, ModeConfig
from src.environments.resources import Resource, ResourceType
from src.environments.terrain import TerrainGrid, TerrainType


class AgentRole(IntEnum):
    """Roles for agents in multi-agent scenarios."""

    FORAGER = 0  # Standard foraging agent
    COMPETITOR = 1  # Competes for same resources
    COOPERATOR = 2  # Can share resources
    PREDATOR = 3  # Tries to catch other agents


@dataclass
class MultiAgentState:
    """State for a single agent in multi-agent environment."""

    agent_id: int
    position: np.ndarray
    energy: float
    role: AgentRole = AgentRole.FORAGER
    is_alive: bool = True
    resources_collected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "position": tuple(self.position),
            "energy": self.energy,
            "role": str(self.role.name),
            "is_alive": self.is_alive,
            "resources_collected": self.resources_collected,
        }


class MultiAgentForagingEnv(ForagingGridEnv):
    """
    Multi-agent foraging environment.

    Extends ForagingGridEnv to support multiple agents with:
    - Competitive resource collection
    - Cooperative resource sharing
    - Predator-prey dynamics
    - Social observation (seeing other agents)
    """

    def __init__(
        self,
        grid_size: int = 20,
        n_agents: int = 2,
        max_steps: int = 1000,
        initial_energy: float = 150.0,
        max_energy: Optional[float] = None,
        energy_decay_rate: float = 0.3,
        local_view_size: int = 5,
        mode: str = "competitive",  # competitive, cooperative, predator_prey
        collision_penalty: float = 0.0,
        sharing_bonus: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize multi-agent environment.

        Args:
            grid_size: Size of the grid
            n_agents: Number of agents
            max_steps: Maximum steps
            initial_energy: Starting energy per agent
            max_energy: Maximum energy per agent
            energy_decay_rate: Energy decay per step
            local_view_size: Observation radius
            mode: Game mode (competitive, cooperative, predator_prey)
            collision_penalty: Penalty for agent collisions
            sharing_bonus: Bonus for cooperative sharing
            seed: Random seed
        """
        self.n_agents = n_agents
        self.multi_agent_mode = mode
        self.collision_penalty = collision_penalty
        self.sharing_bonus = sharing_bonus

        # Agent states (will be initialized in reset)
        self.agent_states: List[MultiAgentState] = []
        self.current_agent_idx = 0  # For turn-based stepping

        # Create mode config for multi-agent
        mode_config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description=f"Multi-agent {mode} environment",
            num_resources=max(5, n_agents * 3),
            resource_types=["renewable"] * max(5, n_agents * 3),
        )

        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            initial_energy=initial_energy,
            max_energy=max_energy,
            energy_decay_rate=energy_decay_rate,
            local_view_size=local_view_size,
            mode=EnvironmentMode.BASELINE,
            mode_config=mode_config,
            render_mode=None,
            seed=seed,
        )

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces for multi-agent."""
        self.action_space = spaces.Discrete(len(Action))

        view_dim = 2 * self.local_view_size + 1

        # Extended observation with agent channel
        self.observation_space = spaces.Dict({
            "local_view": spaces.Box(
                low=0, high=1,
                shape=(view_dim, view_dim, 4),  # Extra channel for other agents
                dtype=np.float32,
            ),
            "energy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "step": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "other_agents": spaces.Box(
                low=0, high=1,
                shape=(self.n_agents - 1, 4),  # (x, y, energy, role) for others
                dtype=np.float32,
            ),
        })

    def _reset_state(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Reset multi-agent state."""
        super()._reset_state(options)

        # Initialize agent states
        self.agent_states = []
        valid_positions = self.terrain.get_passable_positions()

        # Remove resource positions
        resource_positions = {tuple(r.position) for r in self.resources}
        valid_positions = [p for p in valid_positions if p not in resource_positions]

        self.np_random.shuffle(valid_positions)

        for i in range(self.n_agents):
            if i < len(valid_positions):
                pos = np.array(valid_positions[i])
            else:
                pos = np.array([self.grid_size // 2, self.grid_size // 2])

            # Assign roles based on mode
            if self.multi_agent_mode == "predator_prey" and i == 0:
                role = AgentRole.PREDATOR
            elif self.multi_agent_mode == "cooperative":
                role = AgentRole.COOPERATOR
            else:
                role = AgentRole.FORAGER

            self.agent_states.append(MultiAgentState(
                agent_id=i,
                position=pos,
                energy=self.initial_energy,
                role=role,
                is_alive=True,
            ))

        # Primary agent position for compatibility
        self.agent_position = self.agent_states[0].position.copy()
        self.agent_energy = self.agent_states[0].energy
        self.current_agent_idx = 0

    def _get_observation(self, agent_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get observation for a specific agent."""
        if agent_idx is None:
            agent_idx = self.current_agent_idx

        agent = self.agent_states[agent_idx]

        # Build local view with extra channel for other agents
        local_view = self._get_local_view_multi_agent(agent_idx)

        # Normalized values
        energy = np.array([agent.energy / self.max_energy], dtype=np.float32)
        position = agent.position.astype(np.float32) / self.grid_size
        step = np.array([self.current_step / self.max_steps], dtype=np.float32)

        # Other agents info
        other_agents = self._get_other_agents_info(agent_idx)

        return {
            "local_view": local_view,
            "energy": energy,
            "position": position,
            "step": step,
            "other_agents": other_agents,
        }

    def _get_local_view_multi_agent(self, agent_idx: int) -> np.ndarray:
        """Get local view with agent channel."""
        view_dim = 2 * self.local_view_size + 1
        local_view = np.zeros((view_dim, view_dim, 4), dtype=np.float32)

        agent = self.agent_states[agent_idx]
        ax, ay = agent.position

        for dx in range(-self.local_view_size, self.local_view_size + 1):
            for dy in range(-self.local_view_size, self.local_view_size + 1):
                gx, gy = ax + dx, ay + dy
                vx, vy = dx + self.local_view_size, dy + self.local_view_size

                if self.terrain.in_bounds(gx, gy):
                    # Channel 0: Terrain
                    if not self.terrain.is_passable(gx, gy):
                        local_view[vx, vy, 0] = 1.0

                    # Channel 1: Resources
                    for resource in self.resources:
                        if resource.x == gx and resource.y == gy and resource.is_active:
                            local_view[vx, vy, 1] = 1.0
                            break

                    # Channel 3: Other agents
                    for i, other_agent in enumerate(self.agent_states):
                        if i != agent_idx and other_agent.is_alive:
                            if tuple(other_agent.position) == (gx, gy):
                                local_view[vx, vy, 3] = (other_agent.role + 1) / 4.0
                else:
                    local_view[vx, vy, 0] = 1.0

        # Channel 2: Self position
        local_view[self.local_view_size, self.local_view_size, 2] = 1.0

        return local_view

    def _get_other_agents_info(self, agent_idx: int) -> np.ndarray:
        """Get information about other agents."""
        info = np.zeros((self.n_agents - 1, 4), dtype=np.float32)

        idx = 0
        for i, agent in enumerate(self.agent_states):
            if i != agent_idx:
                info[idx, 0] = agent.position[0] / self.grid_size
                info[idx, 1] = agent.position[1] / self.grid_size
                info[idx, 2] = agent.energy / self.max_energy if agent.is_alive else 0
                info[idx, 3] = agent.role / 4.0
                idx += 1

        return info

    def step_agent(self, action: int, agent_idx: Optional[int] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Step a specific agent.

        Args:
            action: Action to take
            agent_idx: Agent index (default: current)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if agent_idx is None:
            agent_idx = self.current_agent_idx

        agent = self.agent_states[agent_idx]
        reward = 0.0

        if not agent.is_alive:
            return self._get_observation(agent_idx), 0.0, True, False, {"agent_dead": True}

        action = Action(action)

        if action == Action.CONSUME:
            # Try to consume resource
            for resource in self.resources:
                if resource.is_active and tuple(resource.position) == tuple(agent.position):
                    energy_gained = resource.consume()
                    agent.energy = min(self.max_energy, agent.energy + energy_gained)
                    agent.resources_collected += 1

                    if energy_gained > 0:
                        reward += energy_gained / 10.0 + 5.0

                        # Cooperative sharing bonus
                        if self.multi_agent_mode == "cooperative":
                            reward += self.sharing_bonus
                            # Share small amount with nearby cooperators
                            for other in self.agent_states:
                                if other.agent_id != agent_idx and other.role == AgentRole.COOPERATOR:
                                    dist = np.linalg.norm(agent.position - other.position)
                                    if dist <= 3:
                                        other.energy = min(
                                            self.max_energy,
                                            other.energy + energy_gained * 0.2
                                        )
                    break

        elif action in DIRECTION_VECTORS:
            direction = DIRECTION_VECTORS[action]
            new_position = agent.position + direction

            if self.terrain.is_passable(new_position[0], new_position[1]):
                # Check collision with other agents
                collision = False
                for other in self.agent_states:
                    if other.agent_id != agent_idx and other.is_alive:
                        if tuple(other.position) == tuple(new_position):
                            collision = True

                            # Predator-prey interaction
                            if self.multi_agent_mode == "predator_prey":
                                if agent.role == AgentRole.PREDATOR:
                                    reward += 10.0  # Caught prey
                                    other.energy = 0
                                elif other.role == AgentRole.PREDATOR:
                                    reward -= 10.0  # Got caught
                                    agent.energy = 0
                            else:
                                reward -= self.collision_penalty
                            break

                if not collision or self.collision_penalty == 0:
                    agent.energy -= self.terrain.get_movement_cost(new_position[0], new_position[1])
                    agent.position = new_position

        # Energy decay
        agent.energy -= self.energy_decay_rate

        # Check death
        if agent.energy <= 0:
            agent.energy = 0
            agent.is_alive = False

        # Update for compatibility
        if agent_idx == 0:
            self.agent_position = agent.position.copy()
            self.agent_energy = agent.energy

        # Update resources
        for resource in self.resources:
            resource.update()

        self.current_step += 1

        # Check termination
        terminated = not agent.is_alive or not any(a.is_alive for a in self.agent_states)
        truncated = self.current_step >= self.max_steps

        # Survival bonus
        if agent.is_alive:
            reward += 0.01

        info = self._get_info_multi_agent(agent_idx)

        return self._get_observation(agent_idx), reward, terminated, truncated, info

    def _get_info_multi_agent(self, agent_idx: int) -> Dict[str, Any]:
        """Get info for multi-agent environment."""
        agent = self.agent_states[agent_idx]

        return {
            "agent_id": agent_idx,
            "position": tuple(agent.position),
            "energy": agent.energy,
            "is_alive": agent.is_alive,
            "resources_collected": agent.resources_collected,
            "agents_alive": sum(1 for a in self.agent_states if a.is_alive),
            "all_agent_states": [a.to_dict() for a in self.agent_states],
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step current agent and advance to next."""
        result = self.step_agent(action, self.current_agent_idx)

        # Advance to next alive agent
        for _ in range(self.n_agents):
            self.current_agent_idx = (self.current_agent_idx + 1) % self.n_agents
            if self.agent_states[self.current_agent_idx].is_alive:
                break

        return result

    def get_all_observations(self) -> List[Dict[str, np.ndarray]]:
        """Get observations for all agents."""
        return [self._get_observation(i) for i in range(self.n_agents)]
