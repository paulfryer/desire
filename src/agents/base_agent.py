"""
Abstract base agent interface for the Desire research project.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Desire research project.

    All agents must implement a common interface to allow for fair
    behavioral comparison across different architectures.

    Critical methods that must be implemented:
    - act(): Select an action given an observation
    - get_internal_state(): Expose internal representations
    - get_internal_response(): Get response to hypothetical observation
    - reset(): Reset agent state for new episode
    - set_energy(): Update agent's knowledge of current energy
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        name: str = "base_agent",
        seed: int | None = None,
    ):
        """
        Initialize the base agent.

        Args:
            observation_space: The environment's observation space
            action_space: The environment's action space
            name: Name identifier for the agent
            seed: Random seed for reproducibility
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.name = name
        self.seed = seed

        # Random number generator
        self._rng = np.random.default_rng(seed)

        # Track current energy (set by environment/wrapper)
        self.current_energy: float = 100.0

        # Step counter for internal tracking
        self.step_count: int = 0

    @abstractmethod
    def act(self, observation: dict[str, np.ndarray]) -> int:
        """
        Select an action given the current observation.

        Args:
            observation: Dictionary containing observation components
                - local_view: (H, W, C) array of local observation
                - energy: (1,) normalized energy level
                - position: (2,) normalized position
                - step: (1,) normalized step count

        Returns:
            Action index to take
        """
        pass

    @abstractmethod
    def get_internal_state(self) -> np.ndarray:
        """
        Get the agent's internal state representation.

        This is critical for interpretability analysis. Different agents
        expose different internal states:
        - Heuristic: One-hot encoding of active rule
        - RL: Hidden layer activations
        - Active Inference: Belief state distribution

        Returns:
            1D numpy array representing internal state
        """
        pass

    @abstractmethod
    def get_internal_response(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """
        Get the agent's internal response to a hypothetical observation.

        Used for probing agent representations without taking an action.
        Similar to get_internal_state() but for arbitrary observations.

        Args:
            observation: The observation to respond to

        Returns:
            1D numpy array representing the internal response
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the agent's internal state for a new episode.

        This should clear any episode-specific state while preserving
        learned parameters (for trained agents).
        """
        pass

    def set_energy(self, energy: float) -> None:
        """
        Update the agent's knowledge of its current energy level.

        Some agents may use this for internal decision-making.

        Args:
            energy: Current energy level (0-100)
        """
        self.current_energy = energy

    def update(self, observation: dict[str, np.ndarray], action: int, reward: float,
               next_observation: dict[str, np.ndarray], done: bool) -> None:
        """
        Update the agent based on a transition (for learning agents).

        Default implementation does nothing. Override in learning agents.

        Args:
            observation: Starting observation
            action: Action taken
            reward: Reward received
            next_observation: Resulting observation
            done: Whether episode ended
        """
        self.step_count += 1

    def save(self, path: str) -> None:  # noqa: B027
        """
        Save the agent's state to disk.

        Default implementation does nothing. Override for agents with
        learnable parameters.

        Args:
            path: Path to save to
        """

    def load(self, path: str) -> None:  # noqa: B027
        """
        Load the agent's state from disk.

        Default implementation does nothing. Override for agents with
        learnable parameters.

        Args:
            path: Path to load from
        """

    @property
    def internal_state_dim(self) -> int:
        """Dimension of the internal state vector."""
        return len(self.get_internal_state())

    def get_action_probabilities(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """
        Get probability distribution over actions.

        Default implementation returns one-hot for the selected action.
        Override for probabilistic agents.

        Args:
            observation: Current observation

        Returns:
            Array of action probabilities
        """
        action = self.act(observation)
        probs = np.zeros(self.action_space.n)
        probs[action] = 1.0
        return probs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
