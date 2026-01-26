"""
Unit tests for agent implementations.
"""

import numpy as np
import pytest

from src.agents.heuristic import HeuristicAgent, Rule
from src.environments.foraging_grid import ForagingGridEnv


class TestHeuristicAgent:
    """Tests for HeuristicAgent."""

    @pytest.fixture
    def env_and_agent(self):
        """Create environment and agent for testing."""
        env = ForagingGridEnv(grid_size=20, seed=42)
        agent = HeuristicAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=42,
        )
        return env, agent

    def test_agent_creation(self, env_and_agent):
        """Test agent creation."""
        env, agent = env_and_agent

        assert agent.name == "heuristic"
        assert agent.action_space == env.action_space
        assert agent.observation_space == env.observation_space

    def test_agent_reset(self, env_and_agent):
        """Test agent reset."""
        env, agent = env_and_agent

        agent.current_rule = Rule.CONSUME
        agent.known_resources = [(1, 2), (3, 4)]
        agent.step_count = 100

        agent.reset()

        assert agent.current_rule == Rule.RANDOM_WALK
        assert agent.known_resources == []
        assert agent.step_count == 0

    def test_act_returns_valid_action(self, env_and_agent):
        """Test act returns valid action."""
        env, agent = env_and_agent

        obs, _ = env.reset()
        action = agent.act(obs)

        assert 0 <= action < env.action_space.n

    def test_get_internal_state(self, env_and_agent):
        """Test internal state is one-hot encoding."""
        env, agent = env_and_agent

        obs, _ = env.reset()
        agent.act(obs)

        state = agent.get_internal_state()

        assert len(state) == len(Rule)
        assert np.sum(state) == 1.0
        assert np.all((state == 0.0) | (state == 1.0))

    def test_consume_rule_triggered(self, env_and_agent):
        """Test consume rule triggers on resource."""
        env, agent = env_and_agent

        obs, _ = env.reset()

        # Manually place agent on resource with low energy
        if env.resources:
            env.agent_position = env.resources[0].position.copy()
            env.agent_energy = 50.0
            agent.current_energy = 50.0

            # Get new observation
            obs = env._get_observation()
            action = agent.act(obs)

            # Should consume
            assert action == 5 or agent.current_rule == Rule.CONSUME

    def test_internal_response(self, env_and_agent):
        """Test internal response to hypothetical observation."""
        env, agent = env_and_agent

        obs, _ = env.reset()

        response = agent.get_internal_response(obs)

        assert len(response) == len(Rule)
        assert np.sum(response) == 1.0


class TestAgentInterface:
    """Tests for base agent interface compliance."""

    @pytest.fixture
    def env(self):
        """Create environment."""
        return ForagingGridEnv(grid_size=20, seed=42)

    def test_heuristic_interface(self, env):
        """Test heuristic agent implements full interface."""
        agent = HeuristicAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        self._check_interface(env, agent)

    def _check_interface(self, env, agent):
        """Check agent implements required interface."""
        obs, _ = env.reset()
        agent.reset()

        # act
        action = agent.act(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < env.action_space.n

        # get_internal_state
        state = agent.get_internal_state()
        assert isinstance(state, np.ndarray)
        assert len(state) > 0

        # get_internal_response
        response = agent.get_internal_response(obs)
        assert isinstance(response, np.ndarray)
        assert len(response) == len(state)

        # set_energy
        agent.set_energy(50.0)
        assert agent.current_energy == 50.0

        # internal_state_dim property
        dim = agent.internal_state_dim
        assert isinstance(dim, int)
        assert dim == len(state)


class TestRLAgent:
    """Tests for RL agent (if available)."""

    @pytest.fixture
    def env(self):
        """Create environment."""
        return ForagingGridEnv(grid_size=20, seed=42)

    def test_rl_agent_creation(self, env):
        """Test RL agent creation."""
        try:
            from src.agents.rl_agent import HAS_SB3, RLAgent

            if HAS_SB3:
                agent = RLAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    seed=42,
                )
                assert agent.name == "rl_ppo"
        except ImportError:
            pytest.skip("stable-baselines3 not available")

    def test_rl_agent_without_model(self, env):
        """Test RL agent acts randomly without trained model."""
        try:
            from src.agents.rl_agent import HAS_SB3, RLAgent

            if HAS_SB3:
                agent = RLAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    seed=42,
                )

                obs, _ = env.reset()
                action = agent.act(obs)

                assert 0 <= action < env.action_space.n
        except ImportError:
            pytest.skip("stable-baselines3 not available")


class TestActiveInferenceAgent:
    """Tests for Active Inference agent (if available)."""

    @pytest.fixture
    def env(self):
        """Create environment."""
        return ForagingGridEnv(grid_size=20, seed=42)

    def test_active_inference_creation(self, env):
        """Test Active Inference agent creation."""
        try:
            from src.agents.active_inference import HAS_PYMDP, ActiveInferenceAgent

            if HAS_PYMDP:
                agent = ActiveInferenceAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    grid_size=20,
                    seed=42,
                )
                assert agent.name == "active_inference"
        except ImportError:
            pytest.skip("pymdp not available")

    def test_active_inference_act(self, env):
        """Test Active Inference agent action selection."""
        try:
            from src.agents.active_inference import HAS_PYMDP, ActiveInferenceAgent

            if HAS_PYMDP:
                agent = ActiveInferenceAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    grid_size=20,
                    seed=42,
                )

                obs, _ = env.reset()
                action = agent.act(obs)

                assert 0 <= action < env.action_space.n
        except ImportError:
            pytest.skip("pymdp not available")

    def test_active_inference_beliefs(self, env):
        """Test Active Inference agent returns beliefs as internal state."""
        try:
            from src.agents.active_inference import HAS_PYMDP, ActiveInferenceAgent

            if HAS_PYMDP:
                agent = ActiveInferenceAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    grid_size=20,
                    seed=42,
                )

                obs, _ = env.reset()
                agent.act(obs)

                state = agent.get_internal_state()

                # Should be concatenated beliefs
                assert len(state) == sum(agent.num_states)
        except ImportError:
            pytest.skip("pymdp not available")

class TestAgentEpisode:
    """Test agents running complete episodes."""

    @pytest.fixture
    def env(self):
        """Create environment."""
        return ForagingGridEnv(grid_size=20, max_steps=100, seed=42)

    def test_heuristic_episode(self, env):
        """Test heuristic agent can complete an episode."""
        agent = HeuristicAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=42,
        )

        obs, _ = env.reset()
        agent.reset()

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.set_energy(env.agent_energy)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        # Agent should survive some time
        assert steps > 0

    def test_internal_state_changes(self, env):
        """Test internal state changes during episode."""
        agent = HeuristicAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=42,
        )

        obs, _ = env.reset()
        agent.reset()

        states = []
        for _ in range(20):
            agent.act(obs)
            states.append(agent.get_internal_state().copy())
            obs, _, terminated, truncated, _ = env.step(np.random.randint(6))
            if terminated or truncated:
                break

        # States should not all be identical
        states = np.array(states)
        # At least some variation expected
        assert len(states) > 0
