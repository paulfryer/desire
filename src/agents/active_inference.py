"""
Active Inference agent using pymdp.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent

# Try to import pymdp
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
    - Uses a C matrix to encode preferences (high preference for consuming, low-energy aversion)
    - Plans using expected free energy minimization
    - Exposes belief state as internal state for interpretability

    The agent discretizes:
    - Position: Grid cells
    - Energy: Binned into discrete levels
    - Resource visibility: Binary per direction
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
                "Install with: pip install pymdp"
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

        # Total hidden state factors
        # Factor 0: Position (discretized)
        # Factor 1: Energy level (discretized)
        # Factor 2: Resource visible
        self.num_states = [
            self.num_position_states,
            self.num_energy_states,
            self.num_resource_states,
        ]
        self.num_factors = len(self.num_states)

        # Observation modalities
        # Modality 0: Position observation
        # Modality 1: Energy observation
        # Modality 2: Resource observation
        self.num_obs = [
            self.num_position_states,
            self.num_energy_states,
            self.num_resource_states,
        ]
        self.num_modalities = len(self.num_obs)

        # Build generative model
        self._build_model()

        # Current belief state
        self.beliefs: List[np.ndarray] = [
            np.ones(ns) / ns for ns in self.num_states
        ]

        # Internal state for tracking
        self._internal_state: Optional[np.ndarray] = None

    def _build_model(self) -> None:
        """Build the active inference generative model."""
        # A matrices: observation likelihoods P(o|s)
        # Each modality has its own A matrix
        self.A = self._build_A_matrices()

        # B matrices: transition probabilities P(s'|s,a)
        self.B = self._build_B_matrices()

        # C vectors: preferences over observations
        self.C = self._build_C_vectors()

        # D vectors: prior over initial states
        self.D = [np.ones(ns) / ns for ns in self.num_states]

        # Create pymdp agent
        self.pymdp_agent = PyMDPAgent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            control_fac_idx=[0],  # Only position is controllable
            policy_len=self.planning_horizon,
            inference_horizon=1,
            gamma=self.policy_precision,
        )

    def _build_A_matrices(self) -> List[np.ndarray]:
        """Build observation likelihood matrices."""
        A = []

        # Position observation (identity - we observe position directly)
        A_pos = np.eye(self.num_position_states)
        A.append(A_pos)

        # Energy observation (identity - we observe energy directly)
        A_energy = np.eye(self.num_energy_states)
        A.append(A_energy)

        # Resource observation (identity - we observe resource directly)
        A_resource = np.eye(self.num_resource_states)
        A.append(A_resource)

        return A

    def _build_B_matrices(self) -> List[np.ndarray]:
        """Build state transition matrices."""
        B = []

        # Position transitions (depends on action)
        # Shape: (num_positions, num_positions, num_actions)
        B_pos = np.zeros((self.num_position_states, self.num_position_states, self.num_actions))

        # For each action, define how position changes
        for action in range(self.num_actions):
            for state in range(self.num_position_states):
                next_state = self._get_next_position_state(state, action)
                B_pos[next_state, state, action] = 1.0

        B.append(B_pos)

        # Energy transitions (depends on action, but controlled separately)
        # Simplified: energy tends to decrease unless consuming
        B_energy = np.zeros((self.num_energy_states, self.num_energy_states, self.num_actions))

        for action in range(self.num_actions):
            for state in range(self.num_energy_states):
                if action == 5:  # CONSUME
                    # Energy increases
                    next_state = min(state + 1, self.num_energy_states - 1)
                else:
                    # Energy decreases
                    next_state = max(state - 1, 0)
                B_energy[next_state, state, action] = 0.8  # Stochastic
                B_energy[state, state, action] = 0.2

        B.append(B_energy)

        # Resource visibility transitions (semi-random, depends on position)
        B_resource = np.zeros((self.num_resource_states, self.num_resource_states, self.num_actions))
        for action in range(self.num_actions):
            # Resources become visible/invisible stochastically
            B_resource[:, :, action] = np.array([
                [0.6, 0.4],  # not visible -> stays not visible or becomes visible
                [0.3, 0.7],  # visible -> stays visible or becomes not visible
            ])

        B.append(B_resource)

        return B

    def _get_next_position_state(self, state: int, action: int) -> int:
        """Get next position state given action."""
        # Convert state to (x, y) grid coordinates
        x = state // self.NUM_POSITION_BINS
        y = state % self.NUM_POSITION_BINS

        # Apply action
        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.NUM_POSITION_BINS - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.NUM_POSITION_BINS - 1, y + 1)
        # STAY and CONSUME don't change position

        return x * self.NUM_POSITION_BINS + y

    def _build_C_vectors(self) -> List[np.ndarray]:
        """Build preference vectors over observations."""
        C = []

        # Position preferences (neutral)
        C_pos = np.zeros(self.num_position_states)
        C.append(C_pos)

        # Energy preferences (prefer high energy)
        C_energy = np.zeros(self.num_energy_states)
        for i in range(self.num_energy_states):
            # Low energy is aversive
            if i == 0:
                C_energy[i] = -self.low_energy_aversion
            elif i == self.num_energy_states - 1:
                C_energy[i] = self.preference_strength
        C.append(C_energy)

        # Resource preferences (prefer seeing resources)
        C_resource = np.array([0.0, self.preference_strength])
        C.append(C_resource)

        return C

    def _discretize_observation(self, observation: Dict[str, np.ndarray]) -> List[int]:
        """Convert continuous observation to discrete state indices."""
        # Position: convert normalized (x, y) to bin indices
        pos = observation["position"]
        pos_x = int(pos[0] * (self.NUM_POSITION_BINS - 1))
        pos_y = int(pos[1] * (self.NUM_POSITION_BINS - 1))
        pos_x = np.clip(pos_x, 0, self.NUM_POSITION_BINS - 1)
        pos_y = np.clip(pos_y, 0, self.NUM_POSITION_BINS - 1)
        pos_state = pos_x * self.NUM_POSITION_BINS + pos_y

        # Energy: convert normalized to bin
        energy = observation["energy"][0]
        energy_state = int(energy * (self.NUM_ENERGY_BINS - 1))
        energy_state = np.clip(energy_state, 0, self.NUM_ENERGY_BINS - 1)

        # Resource visible: check local view for resources
        local_view = observation["local_view"]
        resource_visible = 1 if np.any(local_view[:, :, 1] > 0.5) else 0

        return [pos_state, energy_state, resource_visible]

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Select action using active inference."""
        # Discretize observation
        obs_indices = self._discretize_observation(observation)

        # Convert to one-hot observations for pymdp
        obs_list = []
        for i, idx in enumerate(obs_indices):
            obs_vec = np.zeros(self.num_obs[i])
            obs_vec[idx] = 1.0
            obs_list.append(obs_vec)

        # Update beliefs (state estimation)
        self.beliefs = self.pymdp_agent.infer_states(obs_list)

        # Update internal state representation
        self._update_internal_state()

        # Policy inference (action selection)
        q_pi, efe = self.pymdp_agent.infer_policies()

        # Sample action from policy
        action = self.pymdp_agent.sample_action()

        # pymdp returns action per control factor, we only have one
        if isinstance(action, np.ndarray):
            action = int(action[0])

        return action

    def _update_internal_state(self) -> None:
        """Update internal state representation from beliefs."""
        # Concatenate all belief distributions
        belief_concat = np.concatenate([b.flatten() for b in self.beliefs])
        self._internal_state = belief_concat

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
        # Reset to uniform beliefs
        self.beliefs = [np.ones(ns) / ns for ns in self.num_states]
        self._internal_state = None
        self.step_count = 0
        self.current_energy = 100.0

        # Reset pymdp agent
        if hasattr(self.pymdp_agent, 'reset'):
            self.pymdp_agent.reset()

    @property
    def internal_state_dim(self) -> int:
        """Dimension of the belief state vector."""
        return sum(self.num_states)


class DummyActiveInferenceAgent(BaseAgent):
    """
    Dummy Active Inference agent for when pymdp is not available.

    Acts semi-randomly with slight preference for resources.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        internal_state_dim: int = 32,
        name: str = "dummy_active_inference",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, name, seed)
        self._internal_state_dim = internal_state_dim
        self._internal_state = np.zeros(internal_state_dim, dtype=np.float32)

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """Act with slight preference toward resources."""
        local_view = observation["local_view"]
        center = local_view.shape[0] // 2

        # Check for resources in each direction
        if local_view[center, center, 1] > 0.5:  # On resource
            if observation["energy"][0] < 0.8:
                return 5  # CONSUME

        # Look for resources in view
        for dx, dy, action in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
            if local_view[center + dx, center + dy, 1] > 0.5:
                if self._rng.random() < 0.7:
                    return action

        # Random action
        return self._rng.integers(0, self.action_space.n)

    def get_internal_state(self) -> np.ndarray:
        return self._internal_state.copy()

    def get_internal_response(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return self._internal_state.copy()

    def reset(self) -> None:
        self._internal_state = np.zeros(self._internal_state_dim, dtype=np.float32)
        self.step_count = 0

    @property
    def internal_state_dim(self) -> int:
        return self._internal_state_dim
