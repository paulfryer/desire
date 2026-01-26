"""
Model-Based RL agent using a Dreamer-style world model.

Implements a world model for planning in latent space using
cross-entropy method (CEM) for action selection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from src.agents.base_agent import BaseAgent


class WorldModelNetwork(nn.Module):
    """
    World model for model-based RL.

    Components:
    - Encoder: observation -> latent state
    - Dynamics: latent + action -> next latent
    - Decoder: latent -> reconstructed observation
    - Reward head: latent -> predicted reward
    - Value head: latent -> estimated value
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Get observation dimensions
        local_view_shape = observation_space["local_view"].shape
        view_size = local_view_shape[0]
        n_channels = local_view_shape[2]

        # Encoder: processes observation to latent state
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_output_size = 64 * view_size * view_size

        # Scalar features: energy (1) + position (2) + step (1) = 4
        scalar_size = 4

        self.encoder_mlp = nn.Sequential(
            nn.Linear(cnn_output_size + scalar_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Mean and log_var
        )

        # Dynamics model: predicts next latent state
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Mean and log_var
        )

        # Decoder: reconstructs observation from latent
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cnn_output_size + scalar_size),
        )

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Store for internal state access
        self._last_latent: Optional[torch.Tensor] = None
        self._last_imagined_rewards: Optional[torch.Tensor] = None
        self._last_value_estimate: Optional[torch.Tensor] = None

    def encode(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent state distribution."""
        # Process local view
        local_view = observation["local_view"]
        if local_view.dim() == 3:
            local_view = local_view.unsqueeze(0)
        local_view = local_view.permute(0, 3, 1, 2)  # (B, C, H, W)

        cnn_features = self.encoder_cnn(local_view)

        # Concatenate scalar features
        energy = observation["energy"]
        position = observation["position"]
        step = observation["step"]

        if energy.dim() == 1:
            energy = energy.unsqueeze(0)
        if position.dim() == 1:
            position = position.unsqueeze(0)
        if step.dim() == 1:
            step = step.unsqueeze(0)

        scalars = torch.cat([energy, position, step], dim=-1)
        combined = torch.cat([cnn_features, scalars], dim=-1)

        # Get latent distribution parameters
        latent_params = self.encoder_mlp(combined)
        mean, log_var = torch.chunk(latent_params, 2, dim=-1)

        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE-style sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def predict_next(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent state given current latent and action."""
        # One-hot encode action if needed
        if action.dim() == 1:
            action_onehot = F.one_hot(action.long(), self.action_dim).float()
        else:
            action_onehot = action

        combined = torch.cat([latent, action_onehot], dim=-1)
        next_params = self.dynamics(combined)
        mean, log_var = torch.chunk(next_params, 2, dim=-1)

        return mean, log_var

    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        return self.reward_head(latent)

    def predict_value(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict value from latent state."""
        return self.value_head(latent)

    def forward(
        self,
        observation: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through world model."""
        # Encode observation
        mean, log_var = self.encode(observation)
        latent = self.reparameterize(mean, log_var)

        self._last_latent = latent.detach()

        outputs = {
            "latent_mean": mean,
            "latent_log_var": log_var,
            "latent": latent,
        }

        # Predict reward and value
        reward = self.predict_reward(latent)
        value = self.predict_value(latent)

        self._last_value_estimate = value.detach()

        outputs["reward"] = reward
        outputs["value"] = value

        # If action provided, predict next state
        if action is not None:
            next_mean, next_log_var = self.predict_next(latent, action)
            outputs["next_latent_mean"] = next_mean
            outputs["next_latent_log_var"] = next_log_var

        return outputs


class ModelBasedRLAgent(BaseAgent):
    """
    Model-Based RL agent using world model for planning.

    Uses Cross-Entropy Method (CEM) for planning in latent space.
    Internal state exposes latent representations and planning information.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        planning_horizon: int = 15,
        num_samples: int = 1000,
        top_k: int = 100,
        cem_iterations: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        device: str = "auto",
        name: str = "model_based_rl",
        seed: Optional[int] = None,
    ):
        """
        Initialize Model-Based RL agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            planning_horizon: Steps to plan ahead
            num_samples: Action sequences to sample in CEM
            top_k: Top sequences to keep in CEM
            cem_iterations: CEM refinement iterations
            latent_dim: Dimension of latent state
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for world model
            device: Computation device
            name: Agent name
            seed: Random seed
        """
        super().__init__(observation_space, action_space, name, seed)

        self.planning_horizon = planning_horizon
        self.num_samples = num_samples
        self.top_k = top_k
        self.cem_iterations = cem_iterations
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create world model
        self.world_model = WorldModelNetwork(
            observation_space=observation_space,
            action_dim=action_space.n,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Optimizer for world model training
        self.optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=learning_rate,
        )

        # Internal state storage
        self._last_internal_state = np.zeros(latent_dim + planning_horizon + 1, dtype=np.float32)
        self._last_imagined_rewards = np.zeros(planning_horizon, dtype=np.float32)

        # Training flag
        self.training_mode = False

        # Experience buffer for training
        self.experience_buffer: List[Dict[str, Any]] = []
        self.buffer_capacity = 10000

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Select action using MPC planning in latent space."""
        if self.training_mode:
            # Epsilon-greedy during training
            if self._rng.random() < 0.1:
                return self._rng.integers(0, self.action_space.n)

        # Use CEM planning
        action = self.plan_with_mpc(observation)

        return action

    def plan_with_mpc(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Plan using Cross-Entropy Method in latent space.

        Args:
            observation: Current observation

        Returns:
            Best action to take
        """
        # Convert observation to tensors
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            # Encode current observation
            mean, log_var = self.world_model.encode(obs_tensors)
            current_latent = self.world_model.reparameterize(mean, log_var)

            # Initialize action distribution (uniform categorical)
            action_probs = torch.ones(self.planning_horizon, self.action_space.n) / self.action_space.n
            action_probs = action_probs.to(self.device)

            # CEM iterations
            for _ in range(self.cem_iterations):
                # Sample action sequences
                total_returns = torch.zeros(self.num_samples, device=self.device)
                action_sequences = torch.zeros(
                    self.num_samples, self.planning_horizon, dtype=torch.long, device=self.device
                )

                for i in range(self.num_samples):
                    # Sample action sequence from current distribution
                    actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
                    action_sequences[i] = actions

                    # Rollout in imagination
                    latent = current_latent.clone()
                    total_return = 0.0

                    for t in range(self.planning_horizon):
                        action = actions[t:t+1]
                        action_onehot = F.one_hot(action, self.action_space.n).float()

                        # Predict next state
                        next_mean, next_log_var = self.world_model.predict_next(latent, action_onehot)
                        latent = self.world_model.reparameterize(next_mean, next_log_var)

                        # Predict reward
                        reward = self.world_model.predict_reward(latent).squeeze()

                        # Discounted return
                        total_return += (0.99 ** t) * reward.item()

                    total_returns[i] = total_return

                # Select top-k sequences
                _, top_indices = torch.topk(total_returns, self.top_k)
                top_sequences = action_sequences[top_indices]

                # Update action distribution from top sequences
                for t in range(self.planning_horizon):
                    action_counts = torch.zeros(self.action_space.n, device=self.device)
                    for k in range(self.top_k):
                        action_counts[top_sequences[k, t]] += 1
                    action_probs[t] = action_counts / self.top_k

            # Store imagined rewards for internal state
            best_return = total_returns[top_indices[0]].item()
            self._last_imagined_rewards = np.zeros(self.planning_horizon, dtype=np.float32)

            # Return first action of best sequence
            best_action = int(action_sequences[top_indices[0], 0].item())

            # Update internal state
            self._update_internal_state(current_latent, best_return)

            return best_action

    def _update_internal_state(self, latent: torch.Tensor, value_estimate: float) -> None:
        """Update the internal state representation."""
        latent_np = latent.cpu().numpy().flatten()

        # Internal state = [latent, imagined_rewards, value_estimate]
        self._last_internal_state = np.concatenate([
            latent_np,
            self._last_imagined_rewards,
            np.array([value_estimate], dtype=np.float32),
        ])

    def get_internal_state(self) -> np.ndarray:
        """Get internal state: latent + imagined rewards + value estimates."""
        return self._last_internal_state.copy()

    def get_internal_response(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get internal response to an observation."""
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            outputs = self.world_model(obs_tensors)
            latent = outputs["latent"].cpu().numpy().flatten()
            value = outputs["value"].cpu().numpy().flatten()

        return np.concatenate([latent, np.zeros(self.planning_horizon), value])

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._last_internal_state = np.zeros(
            self.latent_dim + self.planning_horizon + 1, dtype=np.float32
        )
        self._last_imagined_rewards = np.zeros(self.planning_horizon, dtype=np.float32)
        self.step_count = 0
        self.current_energy = 100.0

    def update(
        self,
        observation: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """Store experience and train world model."""
        self.step_count += 1

        # Store experience
        self.experience_buffer.append({
            "observation": {k: v.copy() for k, v in observation.items()},
            "action": action,
            "reward": reward,
            "next_observation": {k: v.copy() for k, v in next_observation.items()},
            "done": done,
        })

        # Trim buffer if needed
        if len(self.experience_buffer) > self.buffer_capacity:
            self.experience_buffer = self.experience_buffer[-self.buffer_capacity:]

        # Train world model periodically
        if self.training_mode and len(self.experience_buffer) >= 64:
            if self.step_count % 10 == 0:
                self._train_world_model(batch_size=32)

    def _train_world_model(self, batch_size: int = 32) -> float:
        """Train the world model on collected experience."""
        if len(self.experience_buffer) < batch_size:
            return 0.0

        # Sample batch
        indices = self._rng.choice(len(self.experience_buffer), size=batch_size, replace=False)

        # Prepare batch
        obs_batch = {
            k: torch.stack([
                torch.tensor(self.experience_buffer[i]["observation"][k], dtype=torch.float32)
                for i in indices
            ]).to(self.device)
            for k in self.experience_buffer[0]["observation"].keys()
        }

        next_obs_batch = {
            k: torch.stack([
                torch.tensor(self.experience_buffer[i]["next_observation"][k], dtype=torch.float32)
                for i in indices
            ]).to(self.device)
            for k in self.experience_buffer[0]["next_observation"].keys()
        }

        actions = torch.tensor(
            [self.experience_buffer[i]["action"] for i in indices],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [self.experience_buffer[i]["reward"] for i in indices],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        # Forward pass
        self.optimizer.zero_grad()

        outputs = self.world_model(obs_batch, actions)

        # Compute losses
        # 1. Reconstruction loss (simplified - just reward prediction here)
        reward_loss = F.mse_loss(outputs["reward"], rewards)

        # 2. Dynamics loss - predict next latent
        with torch.no_grad():
            next_mean, next_log_var = self.world_model.encode(next_obs_batch)
            next_latent_target = self.world_model.reparameterize(next_mean, next_log_var)

        next_latent_pred = self.world_model.reparameterize(
            outputs["next_latent_mean"],
            outputs["next_latent_log_var"],
        )
        dynamics_loss = F.mse_loss(next_latent_pred, next_latent_target)

        # 3. KL divergence for VAE regularization
        kl_loss = -0.5 * torch.mean(
            1 + outputs["latent_log_var"] - outputs["latent_mean"].pow(2) - outputs["latent_log_var"].exp()
        )

        # Total loss
        total_loss = reward_loss + dynamics_loss + 0.01 * kl_loss

        total_loss.backward()
        self.optimizer.step()

        return float(total_loss.item())

    def save(self, path: str) -> None:
        """Save the world model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "world_model": self.world_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load a saved world model."""
        path = Path(path)
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.world_model.load_state_dict(checkpoint["world_model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    @property
    def internal_state_dim(self) -> int:
        return self.latent_dim + self.planning_horizon + 1
