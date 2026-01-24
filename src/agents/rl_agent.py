"""
Reinforcement Learning agent using PPO from stable-baselines3.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from src.agents.base_agent import BaseAgent

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.policies import ActorCriticPolicy
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    PPO = None
    BaseFeaturesExtractor = nn.Module
    ActorCriticPolicy = None


class ForagingFeatureExtractor(BaseFeaturesExtractor if HAS_SB3 else nn.Module):
    """
    Custom feature extractor for the foraging environment.

    Processes the multi-modal observation (local view + scalars)
    into a unified feature vector.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
    ):
        if HAS_SB3:
            super().__init__(observation_space, features_dim)
        else:
            super().__init__()
            self._features_dim = features_dim

        # Get observation dimensions
        local_view_shape = observation_space["local_view"].shape
        view_size = local_view_shape[0]
        n_channels = local_view_shape[2]

        # CNN for local view processing
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        cnn_output_size = 32 * view_size * view_size

        # Scalar features: energy (1) + position (2) + step (1) = 4
        scalar_size = 4

        # MLP to combine features
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_size + scalar_size, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

        # Store for internal state access
        self._last_features: Optional[torch.Tensor] = None

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through feature extractor."""
        # Process local view with CNN
        # Input shape: (batch, H, W, C) -> need (batch, C, H, W)
        local_view = observations["local_view"]
        if local_view.dim() == 3:
            local_view = local_view.unsqueeze(0)
        local_view = local_view.permute(0, 3, 1, 2)

        cnn_features = self.cnn(local_view)

        # Concatenate scalar features
        energy = observations["energy"]
        position = observations["position"]
        step = observations["step"]

        if energy.dim() == 1:
            energy = energy.unsqueeze(0)
        if position.dim() == 1:
            position = position.unsqueeze(0)
        if step.dim() == 1:
            step = step.unsqueeze(0)

        scalars = torch.cat([energy, position, step], dim=-1)

        # Combine and process
        combined = torch.cat([cnn_features, scalars], dim=-1)
        features = self.mlp(combined)

        # Store for internal state access
        self._last_features = features.detach()

        return features

    @property
    def features_dim(self) -> int:
        return self._features_dim


class RLAgent(BaseAgent):
    """
    Reinforcement Learning agent using PPO.

    Uses stable-baselines3 for training with a custom feature extractor
    that exposes internal state through the penultimate layer activations.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        features_dim: int = 64,
        device: str = "auto",
        name: str = "rl_ppo",
        seed: Optional[int] = None,
    ):
        """
        Initialize the RL agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            learning_rate: Learning rate for PPO
            gamma: Discount factor
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Epochs per update
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            features_dim: Dimension of feature vector
            device: Device for training (auto, cpu, cuda)
            name: Agent name
            seed: Random seed
        """
        super().__init__(observation_space, action_space, name, seed)

        if not HAS_SB3:
            raise ImportError(
                "stable-baselines3 is required for RLAgent. "
                "Install with: pip install stable-baselines3"
            )

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.features_dim = features_dim
        self.device = device

        # Model will be created when training starts
        self.model: Optional[PPO] = None
        self._feature_extractor: Optional[ForagingFeatureExtractor] = None

        # Internal state storage
        self._last_internal_state = np.zeros(features_dim, dtype=np.float32)

        # Flag for training mode
        self.training_mode = False

    def create_model(self, env) -> None:
        """
        Create the PPO model for training.

        Args:
            env: The training environment
        """
        policy_kwargs = dict(
            features_extractor_class=ForagingFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=self.features_dim),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )

        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            policy_kwargs=policy_kwargs,
            device=self.device,
            seed=self.seed,
            verbose=0,
        )

        # Store reference to feature extractor
        self._feature_extractor = self.model.policy.features_extractor

    def train(self, total_timesteps: int, callback=None) -> None:
        """
        Train the agent.

        Args:
            total_timesteps: Total training steps
            callback: Optional callback for logging
        """
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")

        self.training_mode = True
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.training_mode = False

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Select action using the trained policy."""
        if self.model is None:
            # Random action if not trained
            return self._rng.integers(0, self.action_space.n)

        # Convert observation to tensor format
        obs = self._prepare_observation(observation)

        # Get action from model
        action, _states = self.model.predict(obs, deterministic=True)

        # Update internal state
        self._update_internal_state(observation)

        return int(action)

    def _prepare_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare observation for the model."""
        return {
            "local_view": observation["local_view"].astype(np.float32),
            "energy": observation["energy"].astype(np.float32),
            "position": observation["position"].astype(np.float32),
            "step": observation["step"].astype(np.float32),
        }

    def _update_internal_state(self, observation: Dict[str, np.ndarray]) -> None:
        """Update the internal state representation."""
        if self._feature_extractor is None:
            return

        # Convert observation to tensors
        device = next(self._feature_extractor.parameters()).device

        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in observation.items()
        }

        # Get features
        with torch.no_grad():
            features = self._feature_extractor(obs_tensors)
            self._last_internal_state = features.cpu().numpy().flatten()

    def get_internal_state(self) -> np.ndarray:
        """Get the penultimate layer activations."""
        return self._last_internal_state.copy()

    def get_internal_response(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get internal response to an observation."""
        if self._feature_extractor is None:
            return np.zeros(self.features_dim, dtype=np.float32)

        device = next(self._feature_extractor.parameters()).device

        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            features = self._feature_extractor(obs_tensors)
            return features.cpu().numpy().flatten()

    def get_action_probabilities(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action probability distribution."""
        if self.model is None:
            # Uniform distribution
            return np.ones(self.action_space.n) / self.action_space.n

        obs = self._prepare_observation(observation)

        # Get policy distribution
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]

        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()

        return probs.flatten()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._last_internal_state = np.zeros(self.features_dim, dtype=np.float32)
        self.step_count = 0
        self.current_energy = 100.0

    def save(self, path: str) -> None:
        """Save the trained model."""
        if self.model is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))

    def load(self, path: str) -> None:
        """Load a trained model."""
        path = Path(path)
        if path.exists():
            self.model = PPO.load(str(path))
            self._feature_extractor = self.model.policy.features_extractor

    @property
    def internal_state_dim(self) -> int:
        return self.features_dim
