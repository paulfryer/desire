"""
Active Inference agent using pymdp.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent

# Import pymdp
try:
    from pymdp import utils
    from pymdp.agent import Agent as PyMDPAgent
    HAS_PYMDP = True
except ImportError:
    HAS_PYMDP = False
    PyMDPAgent = None


class ActiveInferenceAgent(BaseAgent):
    """
    Active Inference agent using the pymdp library.

    Implements a discrete state-space active inference agent that:
    - Discretizes the continuous state space for pymdp compatibility
    - Uses a C matrix to encode preferences (high preference for resources, low-energy aversion)
    - Plans using expected free energy minimization
    - Exposes belief state as internal state for interpretability

    State factors:
    - Factor 0: Position (discretized into bins)
    - Factor 1: Energy level (discretized into bins)
    - Factor 2: Resource visibility (binary)

    Observation modalities:
    - Modality 0: Position observation
    - Modality 1: Energy observation
    - Modality 2: Resource visibility observation
    """

    # Discretization parameters
    NUM_ENERGY_BINS = 5
    NUM_POSITION_BINS = 5  # Per dimension, so 25 total position states

    def __init__(
        self,
        observation_space,
        action_space,
        planning_horizon: int = 3,
        policy_precision: float = 4.0,
        preference_strength: float = 2.0,
        low_energy_aversion: float = 3.0,
        grid_size: int = 20,
        name: str = "active_inference",
        seed: Optional[int] = None,
    ):
        """
        Initialize the Active Inference agent.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            planning_horizon: Depth of planning (future steps to consider)
            policy_precision: Precision of action selection (higher = more deterministic)
            preference_strength: Strength of preference for resources
            low_energy_aversion: Aversion to low energy states
            grid_size: Size of the environment grid
            name: Agent name
            seed: Random seed
        """
        super().__init__(observation_space, action_space, name, seed)

        if not HAS_PYMDP:
            raise ImportError(
                "pymdp is required for ActiveInferenceAgent. "
                "Install with: pip install inferactively-pymdp"
            )

        self.planning_horizon = planning_horizon
        self.policy_precision = policy_precision
        self.preference_strength = preference_strength
        self.low_energy_aversion = low_energy_aversion
        self.grid_size = grid_size

        # State space dimensions
        self.num_actions = action_space.n
        self.num_position_states = self.NUM_POSITION_BINS ** 2
        self.num_energy_states = self.NUM_ENERGY_BINS
        self.num_resource_states = 2  # visible or not

        # State factors and observation modalities
        self.num_states = [
            self.num_position_states,
            self.num_energy_states,
            self.num_resource_states,
        ]
        self.num_obs = [
            self.num_position_states,
            self.num_energy_states,
            self.num_resource_states,
        ]
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)

        # Control factors: only position is controlled
        self.num_controls = [self.num_actions, 1, 1]
        self.control_fac_idx = [0]

        # Build generative model
        self._build_model()

        # Current belief state
        self.beliefs = None

        # Internal state for tracking
        self._internal_state: Optional[np.ndarray] = None

    def _build_model(self) -> None:
        """Build the active inference generative model."""
        # A matrices: observation likelihoods P(o|s)
        self.A = self._build_A_matrices()

        # B matrices: transition probabilities P(s'|s,a)
        self.B = self._build_B_matrices()

        # C vectors: preferences over observations
        self.C = self._build_C_vectors()

        # D vectors: prior over initial states
        self.D = self._build_D_vectors()

        # Create pymdp agent
        self.pymdp_agent = PyMDPAgent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            control_fac_idx=self.control_fac_idx,
            policy_len=self.planning_horizon,
            gamma=self.policy_precision,
        )

    def _build_A_matrices(self):
        """
        Build observation likelihood matrices with observation noise.

        Each observation modality depends on its corresponding state factor.
        A[m] has shape (num_obs[m], num_states[0], num_states[1], ...)

        We add observation noise (accuracy < 1.0) to prevent beliefs from
        becoming point masses, which would cause deterministic stuck behavior.
        """
        A = utils.obj_array(self.num_modalities)

        # Observation accuracy (0.9 = 90% accurate, 10% spread to neighbors)
        obs_accuracy = 0.85

        for m in range(self.num_modalities):
            n_obs = self.num_obs[m]
            n_states = self.num_states[m]

            # Build likelihood with observation noise
            A_m = np.zeros((n_obs, n_states))

            for s in range(n_states):
                # Primary observation
                A_m[s % n_obs, s] = obs_accuracy

                # Spread remaining probability to nearby observations
                remaining = 1.0 - obs_accuracy
                if n_obs > 1:
                    # For position: spread to neighboring positions
                    if m == 0:  # Position
                        # Get grid coordinates
                        sx = s // self.NUM_POSITION_BINS
                        sy = s % self.NUM_POSITION_BINS
                        neighbors = []
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = sx + dx, sy + dy
                            if 0 <= nx < self.NUM_POSITION_BINS and 0 <= ny < self.NUM_POSITION_BINS:
                                neighbors.append(nx * self.NUM_POSITION_BINS + ny)
                        if neighbors:
                            for n in neighbors:
                                A_m[n, s] += remaining / len(neighbors)
                        else:
                            A_m[s, s] = 1.0  # Edge case: redistribute to self
                    else:
                        # For other factors: spread uniformly to all other obs
                        for o in range(n_obs):
                            if o != s % n_obs:
                                A_m[o, s] += remaining / (n_obs - 1)

            # Normalize columns to sum to 1
            A_m = A_m / A_m.sum(axis=0, keepdims=True)

            # Expand dimensions for other factors
            # pymdp expects A[m] to have shape (num_obs[m], *num_states)
            for f in range(self.num_factors):
                if f != m:
                    A_m = np.expand_dims(A_m, axis=-1)
                    A_m = np.repeat(A_m, self.num_states[f], axis=-1)

            # Reorder axes to match (num_obs[m], num_states[0], num_states[1], num_states[2])
            if m == 0:
                pass  # Already correct
            elif m == 1:
                A_m = np.moveaxis(A_m, [1, 2], [2, 1])
            elif m == 2:
                A_m = np.moveaxis(A_m, [1, 2, 3], [3, 1, 2])

            A[m] = A_m

        return A

    def _build_B_matrices(self):
        """
        Build state transition matrices with transition noise.

        B[f] has shape (num_states[f], num_states[f], num_controls[f])

        We add transition noise to prevent overly deterministic planning
        and encourage exploration.
        """
        B = utils.obj_array(self.num_factors)

        # Transition accuracy (0.9 = 90% intended, 10% slip to adjacent)
        trans_accuracy = 0.85

        # Factor 0: Position transitions (controlled by 6 actions)
        B_pos = np.zeros((
            self.num_position_states,
            self.num_position_states,
            self.num_actions
        ))

        for action in range(self.num_actions):
            for state in range(self.num_position_states):
                intended_next = self._get_next_position_state(state, action)

                # Primary transition
                B_pos[intended_next, state, action] = trans_accuracy

                # Add slip probability to adjacent states
                remaining = 1.0 - trans_accuracy
                sx = state // self.NUM_POSITION_BINS
                sy = state % self.NUM_POSITION_BINS
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < self.NUM_POSITION_BINS and 0 <= ny < self.NUM_POSITION_BINS:
                        neighbor_state = nx * self.NUM_POSITION_BINS + ny
                        if neighbor_state != intended_next:
                            neighbors.append(neighbor_state)

                if neighbors:
                    for n in neighbors:
                        B_pos[n, state, action] += remaining / len(neighbors)
                else:
                    # No neighbors to slip to, add to intended
                    B_pos[intended_next, state, action] = 1.0

        # Normalize columns
        for a in range(self.num_actions):
            col_sums = B_pos[:, :, a].sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            B_pos[:, :, a] = B_pos[:, :, a] / col_sums

        B[0] = B_pos

        # Factor 1: Energy transitions (uncontrolled, single action dimension)
        # Energy tends to decrease over time
        B_energy = np.zeros((
            self.num_energy_states,
            self.num_energy_states,
            1
        ))
        for state in range(self.num_energy_states):
            if state == 0:
                # Already at lowest, stay there
                B_energy[0, state, 0] = 1.0
            else:
                # Energy decays with high probability
                next_state = state - 1
                B_energy[next_state, state, 0] = 0.7
                B_energy[state, state, 0] = 0.3

        B[1] = B_energy

        # Factor 2: Resource visibility (uncontrolled)
        # Resources stay visible/invisible with some persistence
        B_resource = np.zeros((
            self.num_resource_states,
            self.num_resource_states,
            1
        ))
        # Columns must sum to 1
        B_resource[:, :, 0] = np.array([
            [0.7, 0.3],  # P(next=not visible | prev=not visible), P(next=not visible | prev=visible)
            [0.3, 0.7],  # P(next=visible | prev=not visible), P(next=visible | prev=visible)
        ])

        B[2] = B_resource

        return B

    def _get_next_position_state(self, state: int, action: int) -> int:
        """Get next position state given action."""
        x = state // self.NUM_POSITION_BINS
        y = state % self.NUM_POSITION_BINS

        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.NUM_POSITION_BINS - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.NUM_POSITION_BINS - 1, y + 1)
        # Actions 4 (STAY) and 5 (CONSUME) don't change position

        return x * self.NUM_POSITION_BINS + y

    def _build_C_vectors(self):
        """Build preference vectors over observations with exploration bonus."""
        C = utils.obj_array(self.num_modalities)

        # Modality 0: Position preferences - add exploration incentive
        # Slight preference for center positions to encourage movement
        C[0] = np.zeros(self.num_obs[0])
        center = self.NUM_POSITION_BINS // 2
        for x in range(self.NUM_POSITION_BINS):
            for y in range(self.NUM_POSITION_BINS):
                state = x * self.NUM_POSITION_BINS + y
                # Small preference for positions away from edges (exploration)
                dist_from_edge = min(x, y, self.NUM_POSITION_BINS - 1 - x,
                                    self.NUM_POSITION_BINS - 1 - y)
                C[0][state] = 0.1 * dist_from_edge

        # Modality 1: Energy preferences (prefer high energy, avoid low)
        C[1] = np.zeros(self.num_obs[1])
        for e in range(self.num_energy_states):
            # Linear preference from low (negative) to high (positive)
            C[1][e] = self.preference_strength * (e - self.num_energy_states / 2) / (self.num_energy_states / 2)
        C[1][0] = -self.low_energy_aversion  # Strong aversion to lowest energy

        # Modality 2: Resource preferences (strongly prefer seeing resources)
        C[2] = np.array([-0.5, self.preference_strength])

        return C

    def _build_D_vectors(self):
        """Build prior over initial states."""
        D = utils.obj_array(self.num_factors)

        for f in range(self.num_factors):
            D[f] = np.ones(self.num_states[f]) / self.num_states[f]

        return D

    def _discretize_observation(self, observation: Dict[str, np.ndarray]) -> List[int]:
        """Convert continuous observation to discrete state indices."""
        # Position: convert normalized (x, y) to bin indices
        pos = observation["position"]
        pos_x = int(pos[0] * (self.NUM_POSITION_BINS - 1))
        pos_y = int(pos[1] * (self.NUM_POSITION_BINS - 1))
        pos_x = np.clip(pos_x, 0, self.NUM_POSITION_BINS - 1)
        pos_y = np.clip(pos_y, 0, self.NUM_POSITION_BINS - 1)
        pos_state = int(pos_x * self.NUM_POSITION_BINS + pos_y)

        # Energy: convert normalized to bin
        energy = observation["energy"][0]
        energy_state = int(energy * (self.NUM_ENERGY_BINS - 1))
        energy_state = int(np.clip(energy_state, 0, self.NUM_ENERGY_BINS - 1))

        # Resource visible: check local view for resources
        local_view = observation["local_view"]
        resource_visible = 1 if np.any(local_view[:, :, 1] > 0.5) else 0

        return [pos_state, energy_state, resource_visible]

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Select action using active inference with resource-aware behavior."""
        # Discretize observation
        obs_indices = self._discretize_observation(observation)

        # Update beliefs (state estimation)
        self.beliefs = self.pymdp_agent.infer_states(obs_indices)

        # Update internal state representation
        self._update_internal_state()

        # Check if standing on a resource (center of local view)
        local_view = observation["local_view"]
        center = local_view.shape[0] // 2
        on_resource = local_view[center, center, 1] > 0.5  # Resource channel

        # Check energy level
        energy = observation["energy"][0]

        # If on resource and energy < 90%, prefer to consume
        # This heuristic helps since the generative model can't capture
        # the energy-increase effect of consuming
        if on_resource and energy < 0.9:
            if self._rng.random() < 0.7:  # 70% chance to consume when on resource
                return 5  # CONSUME action

        # Policy inference (action selection)
        q_pi, efe = self.pymdp_agent.infer_policies()

        # Sample action from policy
        action = self.pymdp_agent.sample_action()

        # pymdp returns action per control factor, we only have one controlled factor
        if isinstance(action, np.ndarray):
            action = int(action[0])
        elif isinstance(action, (list, tuple)):
            action = int(action[0])

        # Add exploration noise: small chance to take random action
        if self._rng.random() < 0.1:  # 10% exploration
            action = self._rng.integers(0, self.num_actions)

        return action

    def _update_internal_state(self) -> None:
        """Update internal state representation from beliefs."""
        if self.beliefs is None:
            return

        # Concatenate all belief distributions
        belief_list = []
        for b in self.beliefs:
            if isinstance(b, np.ndarray):
                belief_list.append(b.flatten())

        if belief_list:
            self._internal_state = np.concatenate(belief_list)

    def get_internal_state(self) -> np.ndarray:
        """Get the belief state distribution."""
        if self._internal_state is None:
            # Return uniform beliefs
            return np.concatenate([
                np.ones(ns) / ns for ns in self.num_states
            ])
        return self._internal_state.copy()

    def get_internal_response(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get internal response to a hypothetical observation."""
        # Discretize observation
        obs_indices = self._discretize_observation(observation)

        # Create pseudo-beliefs based on observation
        beliefs = []
        for i, idx in enumerate(obs_indices):
            belief = np.zeros(self.num_states[i])
            belief[idx] = 0.8  # High probability at observed state
            belief += 0.2 / self.num_states[i]  # Small probability elsewhere
            belief /= belief.sum()
            beliefs.append(belief)

        return np.concatenate([b.flatten() for b in beliefs])

    def reset(self) -> None:
        """Reset agent beliefs for new episode."""
        self.beliefs = None
        self._internal_state = None
        self.step_count = 0
        self.current_energy = 100.0

        # Reset pymdp agent's internal state
        if hasattr(self.pymdp_agent, 'reset'):
            self.pymdp_agent.reset()
        else:
            # Manually reset relevant attributes
            self.pymdp_agent.qs = None
            if hasattr(self.pymdp_agent, 'action'):
                self.pymdp_agent.action = None

    @property
    def internal_state_dim(self) -> int:
        """Dimension of the belief state vector."""
        return sum(self.num_states)
