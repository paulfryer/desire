"""
Decision Transformer agent for offline RL.

Implements a transformer-based agent that models the RL problem as
sequence modeling, predicting actions conditioned on desired returns.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from src.agents.base_agent import BaseAgent


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class DecisionTransformerNetwork(nn.Module):
    """
    Decision Transformer network.

    Takes (return-to-go, state, action) triples and predicts next action.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        context_length: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.context_length = context_length

        # Get observation dimensions
        local_view_shape = observation_space["local_view"].shape
        view_size = local_view_shape[0]
        n_channels = local_view_shape[2]

        # State encoder (CNN + MLP)
        self.state_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_output_size = 64 * view_size * view_size
        scalar_size = 4

        self.state_encoder = nn.Sequential(
            nn.Linear(cnn_output_size + scalar_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Return-to-go embedding
        self.return_embed = nn.Linear(1, hidden_dim)

        # Action embedding
        self.action_embed = nn.Embedding(action_dim, hidden_dim)

        # Timestep embedding
        self.timestep_embed = nn.Embedding(context_length * 3, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=context_length * 3)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # Layer norm
        self.ln = nn.LayerNorm(hidden_dim)

        # Store for internal state access
        self._last_hidden_state: torch.Tensor | None = None
        self._last_attention_weights: torch.Tensor | None = None

    def encode_state(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation to state embedding."""
        local_view = observations["local_view"]
        if local_view.dim() == 3:
            local_view = local_view.unsqueeze(0)
        local_view = local_view.permute(0, 3, 1, 2)

        cnn_features = self.state_cnn(local_view)

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

        return self.state_encoder(combined)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through Decision Transformer.

        Args:
            states: State embeddings (batch, seq_len, hidden_dim)
            actions: Action indices (batch, seq_len)
            returns_to_go: Return-to-go values (batch, seq_len, 1)
            timesteps: Timestep indices (batch, seq_len)

        Returns:
            Action logits (batch, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[:2]

        # Embed returns, actions
        return_embeddings = self.return_embed(returns_to_go)
        action_embeddings = self.action_embed(actions)

        # Stack as (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        # Shape: (batch, seq_len * 3, hidden_dim)
        stacked = torch.stack([return_embeddings, states, action_embeddings], dim=2)
        stacked = stacked.reshape(batch_size, seq_len * 3, self.hidden_dim)

        # Add positional encoding
        stacked = self.pos_encoding(stacked)
        stacked = self.ln(stacked)

        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len * 3, seq_len * 3, device=stacked.device), diagonal=1
        ).bool()

        # Transformer forward
        hidden = self.transformer(stacked, mask=mask)

        self._last_hidden_state = hidden.detach()

        # Extract state positions (every 3rd starting at index 1)
        state_hidden = hidden[:, 1::3]

        # Predict actions
        action_logits = self.action_head(state_hidden)

        return action_logits


class DecisionTransformerAgent(BaseAgent):
    """
    Decision Transformer agent for offline/online RL.

    Models RL as sequence prediction conditioned on desired returns.
    Internal state exposes transformer hidden states and attention patterns.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        context_length: int = 20,
        learning_rate: float = 1e-4,
        target_return: float = 100.0,
        return_scale: float = 100.0,
        dropout: float = 0.1,
        device: str = "auto",
        name: str = "decision_transformer",
        seed: int | None = None,
    ):
        """
        Initialize Decision Transformer agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            hidden_dim: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            context_length: Number of timesteps to consider
            learning_rate: Learning rate
            target_return: Target return-to-go for conditioning
            return_scale: Scale for normalizing returns
            dropout: Dropout rate
            device: Computation device
            name: Agent name
            seed: Random seed
        """
        super().__init__(observation_space, action_space, name, seed)

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.target_return = target_return
        self.return_scale = return_scale

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Network
        self.network = DecisionTransformerNetwork(
            observation_space=observation_space,
            action_dim=action_space.n,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            context_length=context_length,
            dropout=dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Context buffer for inference
        self._state_buffer: list[torch.Tensor] = []
        self._action_buffer: list[int] = []
        self._return_buffer: list[float] = []
        self._current_return_to_go = target_return

        # Internal state storage
        self._last_internal_state = np.zeros(hidden_dim * 2, dtype=np.float32)

        # Training data
        self.trajectory_buffer: list[dict[str, Any]] = []
        self.training_mode = False

    def act(self, observation: dict[str, np.ndarray]) -> int:
        """Select action using transformer prediction."""
        # Encode current state
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            state_embed = self.network.encode_state(obs_tensors)
            self._state_buffer.append(state_embed)

            # Prepare context
            seq_len = min(len(self._state_buffer), self.context_length)

            # Stack states
            states = torch.stack(self._state_buffer[-seq_len:], dim=1)

            # Prepare actions (pad with zeros for first step)
            if len(self._action_buffer) >= seq_len:
                actions = self._action_buffer[-seq_len:]
            else:
                actions = [0] * (seq_len - len(self._action_buffer)) + self._action_buffer

            actions = torch.tensor([actions], dtype=torch.long, device=self.device)

            # Prepare returns-to-go
            returns = []
            for _i in range(seq_len):
                rtg = self._current_return_to_go / self.return_scale
                returns.append(rtg)
            returns = torch.tensor([[returns]], dtype=torch.float32, device=self.device)
            returns = returns.permute(0, 2, 1)  # (1, seq_len, 1)

            # Timesteps
            timesteps = torch.arange(seq_len, device=self.device).unsqueeze(0)

            # Forward pass
            action_logits = self.network(states, actions, returns, timesteps)

            # Get action for current timestep (last in sequence)
            current_logits = action_logits[:, -1, :]
            action_probs = F.softmax(current_logits, dim=-1)

            if self.training_mode:
                action = torch.multinomial(action_probs, 1).item()
            else:
                action = action_probs.argmax(dim=-1).item()

            # Store action
            self._action_buffer.append(action)

            # Trim buffers
            if len(self._state_buffer) > self.context_length:
                self._state_buffer = self._state_buffer[-self.context_length:]
            if len(self._action_buffer) > self.context_length:
                self._action_buffer = self._action_buffer[-self.context_length:]

            # Update internal state
            hidden = self.network._last_hidden_state
            if hidden is not None:
                # Use mean of hidden states as internal representation
                mean_hidden = hidden.mean(dim=1).cpu().numpy().flatten()
                last_hidden = hidden[:, -1].cpu().numpy().flatten()
                self._last_internal_state = np.concatenate([
                    mean_hidden[:self.hidden_dim],
                    last_hidden[:self.hidden_dim],
                ])[:self.hidden_dim * 2]

        return int(action)

    def get_internal_state(self) -> np.ndarray:
        """Get transformer hidden states."""
        return self._last_internal_state.copy()

    def get_internal_response(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Get internal response to an observation."""
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            state_embed = self.network.encode_state(obs_tensors)
            return state_embed.cpu().numpy().flatten()

    def get_action_probabilities(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Get action probability distribution."""
        # This requires context, so return uniform if no context
        if len(self._state_buffer) == 0:
            return np.ones(self.action_space.n) / self.action_space.n

        # Similar to act but just get probabilities
        obs_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in observation.items()
        }

        with torch.no_grad():
            state_embed = self.network.encode_state(obs_tensors)

            seq_len = min(len(self._state_buffer) + 1, self.context_length)
            states = torch.stack(
                self._state_buffer[-(seq_len-1):] + [state_embed], dim=1
            )

            if len(self._action_buffer) >= seq_len:
                actions = self._action_buffer[-seq_len:]
            else:
                actions = [0] * (seq_len - len(self._action_buffer)) + self._action_buffer
            actions = torch.tensor([actions], dtype=torch.long, device=self.device)

            returns = torch.full(
                (1, seq_len, 1),
                self._current_return_to_go / self.return_scale,
                device=self.device,
            )

            timesteps = torch.arange(seq_len, device=self.device).unsqueeze(0)

            action_logits = self.network(states, actions, returns, timesteps)
            probs = F.softmax(action_logits[:, -1, :], dim=-1)

            return probs.cpu().numpy().flatten()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._state_buffer = []
        self._action_buffer = []
        self._return_buffer = []
        self._current_return_to_go = self.target_return
        self._last_internal_state = np.zeros(self.hidden_dim * 2, dtype=np.float32)
        self.step_count = 0
        self.current_energy = 100.0

    def update(
        self,
        observation: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """Update return-to-go and store trajectory."""
        self.step_count += 1

        # Update return-to-go
        self._current_return_to_go -= reward

        # Store for offline training
        if self.training_mode:
            self.trajectory_buffer.append({
                "observation": {k: v.copy() for k, v in observation.items()},
                "action": action,
                "reward": reward,
                "done": done,
            })

    def train_offline(
        self,
        trajectories: list[list[dict[str, Any]]],
        n_iterations: int = 1000,
        batch_size: int = 64,
    ) -> float:
        """
        Train on offline trajectory data.

        Args:
            trajectories: List of episode trajectories
            n_iterations: Training iterations
            batch_size: Batch size

        Returns:
            Final loss value
        """
        self.training_mode = True
        total_loss = 0.0

        for _iteration in range(n_iterations):
            # Sample trajectories
            batch_states = []
            batch_actions = []
            batch_returns = []

            for _ in range(batch_size):
                # Sample trajectory
                traj_idx = self._rng.integers(0, len(trajectories))
                traj = trajectories[traj_idx]

                if len(traj) < self.context_length:
                    continue

                # Sample starting point
                start = self._rng.integers(0, len(traj) - self.context_length)

                # Compute returns-to-go
                returns = []
                rtg = sum(t["reward"] for t in traj[start:])
                for t in traj[start:start + self.context_length]:
                    returns.append(rtg / self.return_scale)
                    rtg -= t["reward"]

                # Encode states
                states = []
                for t in traj[start:start + self.context_length]:
                    obs_tensors = {
                        k: torch.tensor(v, dtype=torch.float32, device=self.device)
                        for k, v in t["observation"].items()
                    }
                    with torch.no_grad():
                        state = self.network.encode_state(obs_tensors)
                    states.append(state.squeeze(0))

                batch_states.append(torch.stack(states))
                batch_actions.append([t["action"] for t in traj[start:start + self.context_length]])
                batch_returns.append(returns)

            if len(batch_states) == 0:
                continue

            # Prepare batch
            states = torch.stack(batch_states).to(self.device)
            actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
            returns = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)
            returns = returns.unsqueeze(-1)
            timesteps = torch.arange(self.context_length, device=self.device).unsqueeze(0)
            timesteps = timesteps.expand(len(batch_states), -1)

            # Forward pass
            self.optimizer.zero_grad()
            action_logits = self.network(states, actions, returns, timesteps)

            # Loss: predict action at each timestep
            loss = F.cross_entropy(
                action_logits.reshape(-1, self.action_space.n),
                actions.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.training_mode = False
        return total_loss / n_iterations

    def save(self, path: str) -> None:
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load a saved model."""
        path = Path(path)
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    @property
    def internal_state_dim(self) -> int:
        return self.hidden_dim * 2
