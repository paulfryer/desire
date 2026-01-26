"""
SAC (Soft Actor-Critic) agent adapted for discrete action spaces.

Note: Standard SAC is designed for continuous action spaces. This implementation
uses a discrete action variant or falls back to DQN-style Q-learning with
entropy regularization for discrete environments.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from src.agents.base_agent import BaseAgent


class DiscreteSACNetwork(nn.Module):
    """
    Neural network for discrete SAC.

    Outputs Q-values for all actions and policy probabilities.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_dim: int,
        features_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.features_dim = features_dim
        self.action_dim = action_dim

        # Get observation dimensions
        local_view_shape = observation_space["local_view"].shape
        view_size = local_view_shape[0]
        n_channels = local_view_shape[2]

        # Shared feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_output_size = 64 * view_size * view_size
        scalar_size = 4

        self.feature_mlp = nn.Sequential(
            nn.Linear(cnn_output_size + scalar_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

        # Q-network 1
        self.q1 = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Q-network 2 (for double Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._last_features: Optional[torch.Tensor] = None

    def get_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observation."""
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
        features = self.feature_mlp(combined)

        self._last_features = features.detach()
        return features

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning Q-values and policy logits."""
        features = self.get_features(observations)

        q1_values = self.q1(features)
        q2_values = self.q2(features)
        policy_logits = self.policy(features)

        return q1_values, q2_values, policy_logits

    def get_action_probs(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get action probabilities."""
        features = self.get_features(observations)
        policy_logits = self.policy(features)
        return F.softmax(policy_logits, dim=-1)


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic agent for discrete action spaces.

    Uses entropy-regularized Q-learning with automatic temperature adjustment.
    Internal state exposes shared feature representations.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy_ratio: float = 0.98,
        buffer_size: int = 100000,
        batch_size: int = 64,
        learning_starts: int = 1000,
        update_interval: int = 4,
        features_dim: int = 64,
        hidden_dim: int = 128,
        device: str = "auto",
        name: str = "rl_sac",
        seed: Optional[int] = None,
    ):
        """
        Initialize SAC agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy coefficient
            auto_entropy_tuning: Whether to auto-tune entropy
            target_entropy_ratio: Target entropy as ratio of max
            buffer_size: Replay buffer size
            batch_size: Minibatch size
            learning_starts: Steps before learning
            update_interval: Steps between updates
            features_dim: Feature dimension
            hidden_dim: Hidden layer dimension
            device: Computation device
            name: Agent name
            seed: Random seed
        """
        super().__init__(observation_space, action_space, name, seed)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy_ratio = target_entropy_ratio
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_interval = update_interval
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.network = DiscreteSACNetwork(
            observation_space=observation_space,
            action_dim=action_space.n,
            features_dim=features_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Target network
        self.target_network = DiscreteSACNetwork(
            observation_space=observation_space,
            action_dim=action_space.n,
            features_dim=features_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.network.q1.parameters()) + list(self.network.q2.parameters()),
            lr=learning_rate,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.network.policy.parameters(),
            lr=learning_rate,
        )

        # Entropy tuning
        if auto_entropy_tuning:
            # Target entropy: -log(1/|A|) * ratio
            self.target_entropy = -np.log(1.0 / action_space.n) * target_entropy_ratio
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = torch.tensor([np.log(alpha)], device=self.device)

        # Experience buffer
        self.replay_buffer: List[Dict[str, Any]] = []

        # Internal state
        self._last_internal_state = np.zeros(features_dim, dtype=np.float32)
        self.training_mode = False

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Select action using current policy."""
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            action_probs = self.network.get_action_probs(obs_tensors)

            if self.training_mode and self._rng.random() < 0.05:
                # Exploration: sample from distribution
                action = torch.multinomial(action_probs, 1).item()
            else:
                # Exploitation: take argmax
                action = action_probs.argmax(dim=-1).item()

            self._last_internal_state = self.network._last_features.cpu().numpy().flatten()

        return int(action)

    def get_internal_state(self) -> np.ndarray:
        """Get feature extractor activations."""
        return self._last_internal_state.copy()

    def get_internal_response(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get internal response to an observation."""
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            _ = self.network.get_features(obs_tensors)
            return self.network._last_features.cpu().numpy().flatten()

    def get_action_probabilities(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action probability distribution."""
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            probs = self.network.get_action_probs(obs_tensors)
            return probs.cpu().numpy().flatten()

    def reset(self) -> None:
        """Reset agent state."""
        self._last_internal_state = np.zeros(self.features_dim, dtype=np.float32)
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
        """Store transition and update networks."""
        self.step_count += 1

        # Store transition
        self.replay_buffer.append({
            "observation": {k: v.copy() for k, v in observation.items()},
            "action": action,
            "reward": reward,
            "next_observation": {k: v.copy() for k, v in next_observation.items()},
            "done": done,
        })

        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]

        # Update if ready
        if (
            self.training_mode
            and len(self.replay_buffer) >= self.learning_starts
            and self.step_count % self.update_interval == 0
        ):
            self._update_networks()

    def _update_networks(self) -> Dict[str, float]:
        """Update Q-networks and policy."""
        # Sample batch
        indices = self._rng.choice(len(self.replay_buffer), size=self.batch_size, replace=False)

        # Prepare batch tensors
        obs_batch = {
            k: torch.stack([
                torch.tensor(self.replay_buffer[i]["observation"][k], dtype=torch.float32)
                for i in indices
            ]).to(self.device)
            for k in self.replay_buffer[0]["observation"].keys()
        }

        next_obs_batch = {
            k: torch.stack([
                torch.tensor(self.replay_buffer[i]["next_observation"][k], dtype=torch.float32)
                for i in indices
            ]).to(self.device)
            for k in self.replay_buffer[0]["next_observation"].keys()
        }

        actions = torch.tensor(
            [self.replay_buffer[i]["action"] for i in indices],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [self.replay_buffer[i]["reward"] for i in indices],
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            [self.replay_buffer[i]["done"] for i in indices],
            dtype=torch.float32,
            device=self.device,
        )

        # Current Q-values
        q1_values, q2_values, _ = self.network(obs_batch)
        q1 = q1_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2 = q2_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Target Q-values
        with torch.no_grad():
            next_q1, next_q2, next_logits = self.target_network(next_obs_batch)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            # V(s') = E_a[Q(s', a) - alpha * log(pi(a|s'))]
            min_next_q = torch.min(next_q1, next_q2)
            alpha = self.log_alpha.exp().item()
            next_v = (next_probs * (min_next_q - alpha * next_log_probs)).sum(dim=-1)

            target_q = rewards + (1 - dones) * self.gamma * next_v

        # Q-network loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Policy loss
        q1_values, q2_values, policy_logits = self.network(obs_batch)
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)

        min_q = torch.min(q1_values, q2_values)
        alpha = self.log_alpha.exp()
        policy_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Entropy tuning
        if self.auto_entropy_tuning:
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target network
        for param, target_param in zip(
            self.network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def save(self, path: str) -> None:
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str) -> None:
        """Load a saved model."""
        path = Path(path)
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.log_alpha = checkpoint["log_alpha"]

    @property
    def internal_state_dim(self) -> int:
        return self.features_dim
