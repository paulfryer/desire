"""
A2C (Advantage Actor-Critic) agent using stable-baselines3.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from src.agents.base_agent import BaseAgent

# Try to import stable-baselines3
try:
    from stable_baselines3 import A2C
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    A2C = None
    BaseFeaturesExtractor = nn.Module


class A2CFeatureExtractor(BaseFeaturesExtractor if HAS_SB3 else nn.Module):
    """
    Custom feature extractor for A2C agent.
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

        # CNN for local view
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_output_size = 64 * view_size * view_size
        scalar_size = 4

        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_size + scalar_size, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

        self._last_features: torch.Tensor | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        local_view = observations["local_view"]
        if local_view.dim() == 3:
            local_view = local_view.unsqueeze(0)
        local_view = local_view.permute(0, 3, 1, 2)

        cnn_features = self.cnn(local_view)

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
        combined = torch.cat([cnn_features, scalars], dim=-1)
        features = self.mlp(combined)

        self._last_features = features.detach()
        return features

    @property
    def features_dim(self) -> int:
        return self._features_dim


class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic agent using stable-baselines3.

    Synchronous version of A3C with shared actor-critic network.
    Internal state exposes shared feature extractor activations.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        normalize_advantage: bool = False,
        features_dim: int = 64,
        device: str = "auto",
        name: str = "rl_a2c",
        seed: int | None = None,
    ):
        """
        Initialize A2C agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            learning_rate: Learning rate
            gamma: Discount factor
            n_steps: Steps per update
            gae_lambda: GAE lambda parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            rms_prop_eps: RMSprop epsilon
            normalize_advantage: Whether to normalize advantages
            features_dim: Feature dimension
            device: Computation device
            name: Agent name
            seed: Random seed
        """
        super().__init__(observation_space, action_space, name, seed)

        if not HAS_SB3:
            raise ImportError(
                "stable-baselines3 is required for A2CAgent. "
                "Install with: pip install stable-baselines3"
            )

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rms_prop_eps = rms_prop_eps
        self.normalize_advantage = normalize_advantage
        self.features_dim = features_dim
        self.device = device

        self.model: A2C | None = None
        self._feature_extractor: A2CFeatureExtractor | None = None
        self._last_internal_state = np.zeros(features_dim, dtype=np.float32)
        self.training_mode = False

    def create_model(self, env) -> None:
        """Create the A2C model for training."""
        policy_kwargs = {
            "features_extractor_class": A2CFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": self.features_dim},
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        }

        self.model = A2C(
            "MultiInputPolicy",
            env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            rms_prop_eps=self.rms_prop_eps,
            normalize_advantage=self.normalize_advantage,
            policy_kwargs=policy_kwargs,
            device=self.device,
            seed=self.seed,
            verbose=0,
        )

        self._feature_extractor = self.model.policy.features_extractor

    def train(self, total_timesteps: int, callback=None) -> None:
        """Train the agent."""
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")

        self.training_mode = True
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.training_mode = False

    def act(self, observation: dict[str, np.ndarray]) -> int:
        """Select action using trained policy."""
        if self.model is None:
            return self._rng.integers(0, self.action_space.n)

        obs = self._prepare_observation(observation)
        action, _states = self.model.predict(obs, deterministic=True)
        self._update_internal_state(observation)

        return int(action)

    def _prepare_observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "local_view": observation["local_view"].astype(np.float32),
            "energy": observation["energy"].astype(np.float32),
            "position": observation["position"].astype(np.float32),
            "step": observation["step"].astype(np.float32),
        }

    def _update_internal_state(self, observation: dict[str, np.ndarray]) -> None:
        if self._feature_extractor is None:
            return

        device = next(self._feature_extractor.parameters()).device
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            features = self._feature_extractor(obs_tensors)
            self._last_internal_state = features.cpu().numpy().flatten()

    def get_internal_state(self) -> np.ndarray:
        """Get penultimate layer activations."""
        return self._last_internal_state.copy()

    def get_internal_response(self, observation: dict[str, np.ndarray]) -> np.ndarray:
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

    def get_action_probabilities(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Get action probability distribution."""
        if self.model is None:
            return np.ones(self.action_space.n) / self.action_space.n

        obs = self._prepare_observation(observation)
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]

        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()

        return probs.flatten()

    def reset(self) -> None:
        """Reset agent state."""
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
            self.model = A2C.load(str(path))
            self._feature_extractor = self.model.policy.features_extractor

    @property
    def internal_state_dim(self) -> int:
        return self.features_dim
