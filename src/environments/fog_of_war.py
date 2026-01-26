"""
Fog of War environment wrapper for partial observability.

Provides various partial observability mechanisms including:
- Limited visibility range
- Memory decay
- Stochastic observation
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces, Wrapper

from src.environments.foraging_grid import ForagingGridEnv


class FogOfWarWrapper(Wrapper):
    """
    Wrapper that adds fog of war (partial observability) to foraging environment.

    Features:
    - Limited visibility radius (configurable)
    - Explored/unexplored tracking
    - Optional memory decay (forgetting unexplored areas)
    - Observation noise
    """

    def __init__(
        self,
        env: ForagingGridEnv,
        visibility_radius: Optional[int] = None,
        enable_memory: bool = True,
        memory_decay: float = 0.0,
        observation_noise: float = 0.0,
        mask_unexplored: bool = True,
    ):
        """
        Initialize fog of war wrapper.

        Args:
            env: Base foraging environment
            visibility_radius: Max visibility distance (None = use local_view_size)
            enable_memory: Whether to track explored areas
            memory_decay: Rate at which memory fades (0 = perfect memory)
            observation_noise: Noise level for observations
            mask_unexplored: Whether to mask unexplored areas
        """
        super().__init__(env)

        self.visibility_radius = visibility_radius or env.local_view_size
        self.enable_memory = enable_memory
        self.memory_decay = memory_decay
        self.observation_noise = observation_noise
        self.mask_unexplored = mask_unexplored

        # Exploration tracking
        self.explored_map: Optional[np.ndarray] = None  # 0 = unexplored, 1 = explored
        self.memory_map: Optional[np.ndarray] = None  # Strength of memory (decays)

        # Modify observation space to include exploration map
        if enable_memory:
            base_space = env.observation_space
            self.observation_space = spaces.Dict({
                **{k: v for k, v in base_space.spaces.items()},
                "explored": spaces.Box(
                    low=0, high=1,
                    shape=(env.grid_size, env.grid_size),
                    dtype=np.float32,
                ),
            })

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and exploration state."""
        obs, info = self.env.reset(**kwargs)

        # Initialize exploration maps
        grid_size = self.env.grid_size
        self.explored_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.memory_map = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Mark initial visible area as explored
        self._update_exploration()

        # Apply fog of war to observation
        obs = self._apply_fog(obs)

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step environment and update exploration."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update exploration
        self._update_exploration()

        # Apply memory decay
        if self.memory_decay > 0:
            self.memory_map *= (1 - self.memory_decay)
            # Re-strengthen currently visible areas
            self._update_exploration()

        # Apply fog of war
        obs = self._apply_fog(obs)

        # Add exploration info
        info["explored_fraction"] = float(np.mean(self.explored_map > 0))
        info["memory_strength"] = float(np.mean(self.memory_map))

        return obs, reward, terminated, truncated, info

    def _update_exploration(self) -> None:
        """Update exploration state based on current visibility."""
        ax, ay = self.env.agent_position

        for dx in range(-self.visibility_radius, self.visibility_radius + 1):
            for dy in range(-self.visibility_radius, self.visibility_radius + 1):
                gx, gy = ax + dx, ay + dy

                # Check if within visibility radius (circular)
                if dx * dx + dy * dy <= self.visibility_radius * self.visibility_radius:
                    if 0 <= gx < self.env.grid_size and 0 <= gy < self.env.grid_size:
                        self.explored_map[gx, gy] = 1.0
                        self.memory_map[gx, gy] = 1.0

    def _apply_fog(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply fog of war to observation."""
        obs = obs.copy()

        if self.mask_unexplored:
            # Mask local view based on visibility
            local_view = obs["local_view"].copy()
            ax, ay = self.env.agent_position

            for vx in range(local_view.shape[0]):
                for vy in range(local_view.shape[1]):
                    dx = vx - self.env.local_view_size
                    dy = vy - self.env.local_view_size

                    # Check if beyond visibility
                    if dx * dx + dy * dy > self.visibility_radius * self.visibility_radius:
                        local_view[vx, vy, :] = 0.5  # Unknown marker

            obs["local_view"] = local_view

        # Add observation noise
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, obs["local_view"].shape)
            obs["local_view"] = np.clip(obs["local_view"] + noise, 0, 1).astype(np.float32)

        # Add exploration map to observation
        if self.enable_memory:
            obs["explored"] = self.memory_map.copy()

        return obs

    def get_exploration_stats(self) -> Dict[str, float]:
        """Get exploration statistics."""
        return {
            "explored_fraction": float(np.mean(self.explored_map > 0)),
            "memory_strength": float(np.mean(self.memory_map)),
            "fully_explored": float(np.all(self.explored_map > 0)),
        }


class StochasticObservationWrapper(Wrapper):
    """
    Wrapper that adds stochastic observation noise and dropout.

    Simulates unreliable sensors or noisy perception.
    """

    def __init__(
        self,
        env: ForagingGridEnv,
        dropout_prob: float = 0.1,
        noise_std: float = 0.1,
        false_positive_prob: float = 0.05,
        false_negative_prob: float = 0.05,
    ):
        """
        Initialize stochastic observation wrapper.

        Args:
            env: Base environment
            dropout_prob: Probability of dropping observation elements
            noise_std: Standard deviation of Gaussian noise
            false_positive_prob: Probability of false resource detection
            false_negative_prob: Probability of missing real resource
        """
        super().__init__(env)

        self.dropout_prob = dropout_prob
        self.noise_std = noise_std
        self.false_positive_prob = false_positive_prob
        self.false_negative_prob = false_negative_prob

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        obs = self._add_noise(obs)
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with noisy observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._add_noise(obs)
        info["observation_noisy"] = True
        return obs, reward, terminated, truncated, info

    def _add_noise(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add noise to observation."""
        obs = obs.copy()
        local_view = obs["local_view"].copy()

        # Add Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, local_view.shape)
            local_view = np.clip(local_view + noise, 0, 1).astype(np.float32)

        # Apply dropout
        if self.dropout_prob > 0:
            mask = np.random.random(local_view.shape) > self.dropout_prob
            local_view = local_view * mask

        # Resource channel false positives/negatives
        resource_channel = local_view[:, :, 1]

        # False positives (see resource where there isn't one)
        if self.false_positive_prob > 0:
            fp_mask = np.random.random(resource_channel.shape) < self.false_positive_prob
            resource_channel[fp_mask] = np.maximum(resource_channel[fp_mask], 0.5)

        # False negatives (miss resource that is there)
        if self.false_negative_prob > 0:
            fn_mask = np.random.random(resource_channel.shape) < self.false_negative_prob
            resource_channel[fn_mask & (resource_channel > 0.5)] = 0

        local_view[:, :, 1] = resource_channel
        obs["local_view"] = local_view

        return obs


class DelayedObservationWrapper(Wrapper):
    """
    Wrapper that delays observations by a fixed number of steps.

    Tests agent ability to handle observation latency.
    """

    def __init__(self, env: ForagingGridEnv, delay: int = 1):
        """
        Initialize delayed observation wrapper.

        Args:
            env: Base environment
            delay: Number of steps to delay observations
        """
        super().__init__(env)
        self.delay = delay
        self.obs_buffer = []

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and clear buffer."""
        obs, info = self.env.reset(**kwargs)

        # Initialize buffer with current observation
        self.obs_buffer = [obs.copy() for _ in range(self.delay + 1)]

        return self.obs_buffer[0], info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with delayed observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add current observation to buffer
        self.obs_buffer.append(obs.copy())

        # Return delayed observation
        delayed_obs = self.obs_buffer.pop(0)

        info["observation_delay"] = self.delay
        info["current_position"] = tuple(self.env.agent_position)

        return delayed_obs, reward, terminated, truncated, info
