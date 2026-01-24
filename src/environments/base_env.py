"""
Abstract base environment interface for the Desire research project.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseEnvironment(gym.Env, ABC):
    """
    Abstract base class for all environments in the Desire research project.

    All environments must be Gymnasium-compatible and provide:
    - Standard Gym API (reset, step, render)
    - Energy-based survival mechanics
    - Configurable observation spaces
    - Mode support for different test configurations
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    def __init__(
        self,
        grid_size: int = 20,
        max_steps: int = 1000,
        initial_energy: float = 100.0,
        max_energy: float = 100.0,
        energy_decay_rate: float = 1.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the base environment.

        Args:
            grid_size: Size of the square grid
            max_steps: Maximum steps per episode
            initial_energy: Starting energy for the agent
            max_energy: Maximum energy the agent can have
            energy_decay_rate: Energy lost per step
            render_mode: How to render the environment
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.initial_energy = initial_energy
        self.max_energy = max_energy
        self.energy_decay_rate = energy_decay_rate
        self.render_mode = render_mode

        # Set random seed
        self._np_random: Optional[np.random.Generator] = None
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # State variables (initialized in reset)
        self.agent_position: np.ndarray = np.array([0, 0])
        self.agent_energy: float = initial_energy
        self.current_step: int = 0
        self.done: bool = False

        # Define action and observation spaces (to be overridden by subclasses)
        self._setup_spaces()

    @abstractmethod
    def _setup_spaces(self) -> None:
        """Set up action and observation spaces. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get the current observation. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _apply_action(self, action: int) -> float:
        """
        Apply an action and return the immediate reward.
        Must be implemented by subclasses.

        Args:
            action: The action to apply

        Returns:
            The immediate reward from this action
        """
        pass

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional seed for reproducibility
            options: Optional configuration options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Reset state
        self.agent_energy = self.initial_energy
        self.current_step = 0
        self.done = False

        # Position will be set by subclass
        self._reset_state(options)

        return self._get_observation(), self._get_info()

    @abstractmethod
    def _reset_state(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Reset environment-specific state. Must be implemented by subclasses."""
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() before stepping.")

        # Apply action and get reward
        reward = self._apply_action(action)

        # Decay energy
        self.agent_energy -= self.energy_decay_rate

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        terminated = self.agent_energy <= 0  # Death
        truncated = self.current_step >= self.max_steps  # Time limit

        if terminated:
            reward -= 100.0  # Death penalty

        self.done = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    @abstractmethod
    def _render_ansi(self) -> str:
        """Render as ASCII string. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _render_rgb(self) -> np.ndarray:
        """Render as RGB array. Must be implemented by subclasses."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    @property
    def np_random(self) -> np.random.Generator:
        """Get the random number generator."""
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random
