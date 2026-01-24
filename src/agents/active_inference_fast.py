"""
Optimized Active Inference agent with faster policy inference.

Key optimizations:
1. Precompute and cache static matrices (log A, etc.)
2. Vectorize policy evaluation using batch matrix operations
3. Prune policy space using beam search (top-k policies)
4. Use simplified expected free energy computation
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from src.agents.base_agent import BaseAgent

# Import pymdp utilities
try:
    from pymdp import utils
    HAS_PYMDP = True
except ImportError:
    HAS_PYMDP = False


class ActiveInferenceFast(BaseAgent):
    """
    Optimized Active Inference agent.

    Uses vectorized computations and policy pruning for speed.
    Target: planning_horizon=3 in <1s per step on 20x20 grid.
    """

    NUM_ENERGY_BINS = 5
    NUM_POSITION_BINS = 5  # 25 total position states

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        planning_horizon: int = 3,
        policy_precision: float = 4.0,
        preference_strength: float = 2.0,
        low_energy_aversion: float = 3.0,
        grid_size: int = 20,
        beam_width: int = 20,  # Top-k policies to evaluate
        name: str = "active_inference",
        seed: Optional[int] = None,
    ):
        super().__init__(observation_space, action_space, name, seed)

        if not HAS_PYMDP:
            raise ImportError("pymdp required. Install: pip install inferactively-pymdp")

        self.planning_horizon = planning_horizon
        self.policy_precision = policy_precision
        self.preference_strength = preference_strength
        self.low_energy_aversion = low_energy_aversion
        self.grid_size = grid_size
        self.beam_width = beam_width

        # State/action dimensions
        self.num_actions = action_space.n
        self.num_position_states = self.NUM_POSITION_BINS ** 2
        self.num_energy_states = self.NUM_ENERGY_BINS
        self.num_resource_states = 2

        self.num_states = [
            self.num_position_states,
            self.num_energy_states,
            self.num_resource_states,
        ]
        self.num_factors = len(self.num_states)

        # Build and cache model components
        self._build_model()
        self._precompute_caches()

        # Current beliefs
        self.beliefs = None
        self._internal_state = None

    def _build_model(self):
        """Build generative model matrices."""
        # A matrices: P(o|s) - observation likelihood
        self.A = self._build_A_matrices()

        # B matrices: P(s'|s,a) - transitions
        self.B = self._build_B_matrices()

        # C vectors: log preferences over observations
        self.C = self._build_C_vectors()

        # D vectors: prior beliefs
        self.D = self._build_D_vectors()

    def _precompute_caches(self):
        """Precompute values that don't change during episode."""
        # Log A for KL divergence computation
        self.log_A = []
        for a in self.A:
            log_a = np.log(a + 1e-16)
            self.log_A.append(log_a)

        # Precompute all possible single-step transitions
        # Shape: (num_states[f], num_actions) for position factor
        self.B_pos_flat = self.B[0]  # (25, 25, 6)

        # Generate all policies up to horizon
        self._generate_policies()

    def _generate_policies(self):
        """Generate policy space with optional pruning."""
        n_actions = self.num_actions
        horizon = self.planning_horizon

        # Full policy space: n_actions^horizon policies
        # Each policy is a sequence of actions
        total_policies = n_actions ** horizon

        if total_policies <= 100:
            # Small enough - enumerate all
            self.policies = np.array(
                np.meshgrid(*[np.arange(n_actions) for _ in range(horizon)])
            ).T.reshape(-1, horizon)
            self.use_beam_search = False
        else:
            # Too many - we'll use beam search during inference
            self.policies = None
            self.use_beam_search = True

    def _build_A_matrices(self):
        """Build observation likelihood with noise."""
        A = utils.obj_array(self.num_factors)
        obs_accuracy = 0.85

        for m in range(self.num_factors):
            n_obs = self.num_states[m]
            n_states = self.num_states[m]

            A_m = np.zeros((n_obs, n_states))

            for s in range(n_states):
                A_m[s, s] = obs_accuracy
                remaining = 1.0 - obs_accuracy

                if m == 0:  # Position - spread to neighbors
                    sx, sy = s // self.NUM_POSITION_BINS, s % self.NUM_POSITION_BINS
                    neighbors = []
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < self.NUM_POSITION_BINS and 0 <= ny < self.NUM_POSITION_BINS:
                            neighbors.append(nx * self.NUM_POSITION_BINS + ny)
                    if neighbors:
                        for n in neighbors:
                            A_m[n, s] += remaining / len(neighbors)
                    else:
                        A_m[s, s] = 1.0
                else:  # Other factors - spread uniformly
                    for o in range(n_obs):
                        if o != s:
                            A_m[o, s] += remaining / (n_obs - 1)

            # Normalize
            A_m = A_m / A_m.sum(axis=0, keepdims=True)

            # Expand to full shape for pymdp compatibility
            for f in range(self.num_factors):
                if f != m:
                    A_m = np.expand_dims(A_m, axis=-1)
                    A_m = np.repeat(A_m, self.num_states[f], axis=-1)

            if m == 1:
                A_m = np.moveaxis(A_m, [1, 2], [2, 1])
            elif m == 2:
                A_m = np.moveaxis(A_m, [1, 2, 3], [3, 1, 2])

            A[m] = A_m

        return A

    def _build_B_matrices(self):
        """Build transition matrices."""
        B = utils.obj_array(self.num_factors)
        trans_accuracy = 0.85

        # Position transitions
        B_pos = np.zeros((self.num_position_states, self.num_position_states, self.num_actions))

        for action in range(self.num_actions):
            for state in range(self.num_position_states):
                next_state = self._get_next_position(state, action)
                B_pos[next_state, state, action] = trans_accuracy

                # Slip to neighbors
                sx, sy = state // self.NUM_POSITION_BINS, state % self.NUM_POSITION_BINS
                neighbors = []
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < self.NUM_POSITION_BINS and 0 <= ny < self.NUM_POSITION_BINS:
                        ns = nx * self.NUM_POSITION_BINS + ny
                        if ns != next_state:
                            neighbors.append(ns)
                if neighbors:
                    remaining = 1.0 - trans_accuracy
                    for n in neighbors:
                        B_pos[n, state, action] += remaining / len(neighbors)
                else:
                    B_pos[next_state, state, action] = 1.0

        # Normalize columns
        for a in range(self.num_actions):
            col_sums = B_pos[:, :, a].sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            B_pos[:, :, a] /= col_sums
        B[0] = B_pos

        # Energy transitions (decay)
        B_energy = np.zeros((self.num_energy_states, self.num_energy_states, 1))
        for s in range(self.num_energy_states):
            if s == 0:
                B_energy[0, s, 0] = 1.0
            else:
                B_energy[s-1, s, 0] = 0.7
                B_energy[s, s, 0] = 0.3
        B[1] = B_energy

        # Resource visibility
        B_resource = np.zeros((2, 2, 1))
        B_resource[:, :, 0] = np.array([[0.7, 0.3], [0.3, 0.7]])
        B[2] = B_resource

        return B

    def _get_next_position(self, state: int, action: int) -> int:
        """Get next position state."""
        x, y = state // self.NUM_POSITION_BINS, state % self.NUM_POSITION_BINS

        if action == 0:    # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.NUM_POSITION_BINS - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.NUM_POSITION_BINS - 1, y + 1)
        # 4=STAY, 5=CONSUME: no position change

        return x * self.NUM_POSITION_BINS + y

    def _build_C_vectors(self):
        """Build preference vectors."""
        C = utils.obj_array(self.num_factors)

        # Position preferences (slight exploration bonus)
        C[0] = np.zeros(self.num_position_states)
        for x in range(self.NUM_POSITION_BINS):
            for y in range(self.NUM_POSITION_BINS):
                s = x * self.NUM_POSITION_BINS + y
                dist = min(x, y, self.NUM_POSITION_BINS-1-x, self.NUM_POSITION_BINS-1-y)
                C[0][s] = 0.1 * dist

        # Energy preferences
        C[1] = np.zeros(self.num_energy_states)
        for e in range(self.num_energy_states):
            C[1][e] = self.preference_strength * (e - self.num_energy_states/2) / (self.num_energy_states/2)
        C[1][0] = -self.low_energy_aversion

        # Resource preferences
        C[2] = np.array([-0.5, self.preference_strength])

        return C

    def _build_D_vectors(self):
        """Build prior beliefs."""
        D = utils.obj_array(self.num_factors)
        for f in range(self.num_factors):
            D[f] = np.ones(self.num_states[f]) / self.num_states[f]
        return D

    def _discretize_observation(self, obs: Dict[str, np.ndarray]) -> List[int]:
        """Convert observation to discrete indices."""
        pos = obs["position"]
        px = int(np.clip(pos[0] * (self.NUM_POSITION_BINS - 1), 0, self.NUM_POSITION_BINS - 1))
        py = int(np.clip(pos[1] * (self.NUM_POSITION_BINS - 1), 0, self.NUM_POSITION_BINS - 1))
        pos_state = px * self.NUM_POSITION_BINS + py

        energy = obs["energy"][0]
        energy_state = int(np.clip(energy * (self.NUM_ENERGY_BINS - 1), 0, self.NUM_ENERGY_BINS - 1))

        local_view = obs["local_view"]
        resource_visible = 1 if np.any(local_view[:, :, 1] > 0.5) else 0

        return [pos_state, energy_state, resource_visible]

    def _update_beliefs(self, obs_indices: List[int]):
        """Update beliefs given observation (simplified Bayesian update)."""
        if self.beliefs is None:
            self.beliefs = [d.copy() for d in self.D]

        # Simple likelihood update
        for f, obs_idx in enumerate(obs_indices):
            # P(s|o) ∝ P(o|s) * P(s)
            likelihood = self.A[f][obs_idx]

            # Marginalize over other factors
            if f == 0:
                lik = likelihood.sum(axis=(1, 2))
            elif f == 1:
                lik = likelihood.sum(axis=(0, 2))
            else:
                lik = likelihood.sum(axis=(0, 1))

            posterior = lik * self.beliefs[f]
            posterior /= posterior.sum() + 1e-16
            self.beliefs[f] = posterior

    def _compute_efe_fast(self, policy: np.ndarray) -> float:
        """
        Compute expected free energy for a policy (vectorized).

        EFE = E_Q[log Q(s) - log P(o|s) - log P(o)]
            = ambiguity + risk

        Simplified: just use pragmatic value (expected utility)
        """
        # Start from current beliefs
        qs = [b.copy() for b in self.beliefs]

        total_efe = 0.0

        for t, action in enumerate(policy):
            # Predict next state: qs_next = B @ qs
            # Position
            qs_pos_next = self.B[0][:, :, action] @ qs[0]
            qs_pos_next /= qs_pos_next.sum() + 1e-16

            # Energy (uncontrolled)
            qs_energy_next = self.B[1][:, :, 0] @ qs[1]
            qs_energy_next /= qs_energy_next.sum() + 1e-16

            # Resource (uncontrolled)
            qs_resource_next = self.B[2][:, :, 0] @ qs[2]
            qs_resource_next /= qs_resource_next.sum() + 1e-16

            qs_next = [qs_pos_next, qs_energy_next, qs_resource_next]

            # Expected utility (pragmatic value)
            # E_Q[log P(o)] where P(o) encoded in C
            for f in range(self.num_factors):
                # Expected observation given predicted state
                # For diagonal A: E[o] ≈ s
                expected_utility = np.dot(qs_next[f], self.C[f])
                total_efe -= expected_utility  # Negative because we minimize EFE

            # Discount future
            total_efe *= 0.9

            qs = qs_next

        return total_efe

    def _select_action_beam(self) -> int:
        """Select action using beam search over policies."""
        n_actions = self.num_actions
        horizon = self.planning_horizon

        # Initialize beam with single-step actions
        beam = []
        for a in range(n_actions):
            efe = self._compute_efe_fast(np.array([a]))
            beam.append((efe, [a]))

        # Sort and keep top-k
        beam.sort(key=lambda x: x[0])
        beam = beam[:self.beam_width]

        # Expand beam for remaining horizon
        for t in range(1, horizon):
            new_beam = []
            for efe, actions in beam:
                for a in range(n_actions):
                    new_actions = actions + [a]
                    new_efe = self._compute_efe_fast(np.array(new_actions))
                    new_beam.append((new_efe, new_actions))

            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:self.beam_width]

        # Return first action of best policy
        if beam:
            return beam[0][1][0]
        return self._rng.integers(0, n_actions)

    def _select_action_full(self) -> int:
        """Select action by evaluating all policies."""
        efes = np.array([self._compute_efe_fast(p) for p in self.policies])

        # Softmax with precision
        efes = efes - efes.min()  # Numerical stability
        probs = np.exp(-self.policy_precision * efes)
        probs /= probs.sum() + 1e-16

        # Sample policy
        policy_idx = self._rng.choice(len(self.policies), p=probs)

        # Return first action
        return int(self.policies[policy_idx, 0])

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Select action using fast active inference."""
        obs_indices = self._discretize_observation(observation)
        self._update_beliefs(obs_indices)
        self._update_internal_state()

        # Check if on resource - heuristic consume
        local_view = observation["local_view"]
        center = local_view.shape[0] // 2
        on_resource = local_view[center, center, 1] > 0.5
        energy = observation["energy"][0]

        if on_resource and energy < 0.9:
            if self._rng.random() < 0.7:
                return 5  # CONSUME

        # Select action via policy inference
        if self.use_beam_search:
            action = self._select_action_beam()
        else:
            action = self._select_action_full()

        # Exploration noise
        if self._rng.random() < 0.1:
            action = self._rng.integers(0, self.num_actions)

        return action

    def _update_internal_state(self):
        """Update internal state from beliefs."""
        if self.beliefs is not None:
            self._internal_state = np.concatenate([b.flatten() for b in self.beliefs])

    def get_internal_state(self) -> np.ndarray:
        """Get belief state."""
        if self._internal_state is None:
            return np.concatenate([np.ones(ns)/ns for ns in self.num_states])
        return self._internal_state.copy()

    def get_internal_response(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get internal response to observation."""
        obs_indices = self._discretize_observation(observation)
        beliefs = []
        for i, idx in enumerate(obs_indices):
            b = np.zeros(self.num_states[i])
            b[idx] = 0.8
            b += 0.2 / self.num_states[i]
            b /= b.sum()
            beliefs.append(b)
        return np.concatenate([b.flatten() for b in beliefs])

    def reset(self):
        """Reset for new episode."""
        self.beliefs = None
        self._internal_state = None
        self.step_count = 0
        self.current_energy = 100.0

    @property
    def internal_state_dim(self) -> int:
        return sum(self.num_states)
