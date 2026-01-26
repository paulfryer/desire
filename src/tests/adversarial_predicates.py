"""
Adversarial predicates that favor RL agents.

These tests create scenarios where pure RL approaches should excel,
serving as control conditions to ensure fair comparison.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, ModeConfig
from src.tests.base_test import BasePredicateTest, BaselineData, TestResult
from src.utils.logging import EpisodeLog


class TemporalCreditAssignmentTest(BasePredicateTest):
    """
    Test for Temporal Credit Assignment.

    Creates scenarios with long delayed rewards where RL's ability
    to propagate value through time should excel.

    Protocol:
    - Agent must traverse long path before receiving reward
    - Tests value function bootstrapping capability
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        delay_steps: int = 100,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize Temporal Credit Assignment test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Baseline episodes
            test_episodes: Test episodes
            grid_size: Environment grid size
            delay_steps: Minimum steps before reward is possible
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="temporal_credit_assignment",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.delay_steps = delay_steps

    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create baseline with immediate rewards."""
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            initial_energy=200.0,
            max_energy=200.0,
            energy_decay_rate=0.2,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def create_test_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create test environment with delayed reward (far resource)."""
        # Place resource in far corner
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Delayed reward test",
            num_resources=1,
            resource_types=["renewable"],
            resource_positions=[(self.grid_size - 2, self.grid_size - 2)],
            agent_start_position=(1, 1),
        )

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            initial_energy=200.0,
            max_energy=200.0,
            energy_decay_rate=0.2,
            mode=EnvironmentMode.BASELINE,
            mode_config=config,
            seed=seed,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute credit assignment score.

        Measures how well agent reaches delayed reward.
        """
        reached_target = []
        time_to_target = []

        target_pos = np.array([self.grid_size - 2, self.grid_size - 2])

        for episode in test_episodes:
            trajectory = episode.get_trajectory()

            # Find if and when target was reached
            reached = False
            for step_idx, pos in enumerate(trajectory):
                if np.linalg.norm(np.array(pos) - target_pos) < 1.5:
                    reached = True
                    time_to_target.append(step_idx)
                    break

            reached_target.append(reached)

        success_rate = np.mean(reached_target)
        mean_time = np.mean(time_to_target) if time_to_target else 500

        # Score: success rate with time bonus
        score = success_rate
        if mean_time < 200:
            score += (200 - mean_time) / 400  # Bonus for speed

        raw_data = {
            "success_rate": float(success_rate),
            "mean_time_to_target": float(mean_time),
            "n_reached": sum(reached_target),
            "n_episodes": len(test_episodes),
        }

        return float(score), raw_data


class StochasticRewardTest(BasePredicateTest):
    """
    Test for Stochastic Reward handling.

    High variance rewards where RL averaging should be robust.
    Tests ability to learn from noisy reward signals.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        reward_variance: float = 0.5,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize Stochastic Reward test.

        Args:
            n_trials: Number of trials
            baseline_episodes: Baseline episodes
            test_episodes: Test episodes
            grid_size: Environment grid size
            reward_variance: Variance of reward noise (as fraction of mean)
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="stochastic_reward",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.reward_variance = reward_variance

    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create baseline with deterministic rewards."""
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def create_test_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create test environment (same, rewards modified during episode)."""
        return self.create_baseline_env(seed)

    def _run_episode(
        self,
        env: ForagingGridEnv,
        agent: BaseAgent,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> EpisodeLog:
        """Run episode with stochastic rewards."""
        obs, info = env.reset(seed=seed)
        agent.reset()

        rng = np.random.default_rng(seed)

        steps = []
        total_reward = 0.0
        step = 0
        max_steps = max_steps or env.max_steps

        while step < max_steps:
            action = agent.act(obs)
            internal_state = agent.get_internal_state()
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Add noise to reward
            if reward > 0:
                noise_scale = reward * self.reward_variance
                noise = rng.normal(0, noise_scale)
                reward = max(0, reward + noise)

            total_reward += reward

            from src.utils.logging import StepLog
            steps.append(StepLog(
                step=step,
                action=action,
                position=tuple(env.agent_position),
                energy=env.agent_energy,
                reward=reward,
                internal_state=internal_state,
                info=info.copy(),
            ))

            agent.set_energy(env.agent_energy)

            if terminated or truncated:
                break

            obs = next_obs
            step += 1

        return EpisodeLog(
            episode_id=0,
            agent_type=agent.name,
            environment_mode=str(env.mode),
            total_steps=step + 1,
            total_reward=total_reward,
            final_energy=env.agent_energy,
            survived=not (env.agent_energy <= 0),
            steps=steps,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute stochastic reward handling score."""
        # Compare test performance to baseline
        baseline_rewards = [e.total_reward for e in baseline_data.episodes]
        test_rewards = [e.total_reward for e in test_episodes]

        baseline_mean = np.mean(baseline_rewards)
        test_mean = np.mean(test_rewards)
        test_std = np.std(test_rewards)

        # Score: relative performance under noise
        if baseline_mean > 0:
            relative_performance = test_mean / baseline_mean
        else:
            relative_performance = 1.0

        # Bonus for low variance (consistent despite noise)
        consistency = 1.0 / (1.0 + test_std / (test_mean + 0.01))

        score = relative_performance * (0.7 + 0.3 * consistency)

        raw_data = {
            "baseline_mean_reward": float(baseline_mean),
            "test_mean_reward": float(test_mean),
            "test_std_reward": float(test_std),
            "relative_performance": float(relative_performance),
            "consistency": float(consistency),
        }

        return float(score), raw_data


class RewardShapingRobustnessTest(BasePredicateTest):
    """
    Test for Reward Shaping Robustness.

    Tests agent performance when reward function is perturbed,
    measuring robustness to reward misspecification.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        perturbation_strength: float = 0.3,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize Reward Shaping Robustness test.

        Args:
            n_trials: Number of trials
            baseline_episodes: Baseline episodes
            test_episodes: Test episodes
            grid_size: Environment grid size
            perturbation_strength: Strength of reward perturbation
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="reward_shaping_robustness",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.perturbation_strength = perturbation_strength

    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create baseline environment."""
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def create_test_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create test environment."""
        return self.create_baseline_env(seed)

    def _run_episode(
        self,
        env: ForagingGridEnv,
        agent: BaseAgent,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> EpisodeLog:
        """Run episode with perturbed rewards."""
        obs, info = env.reset(seed=seed)
        agent.reset()

        rng = np.random.default_rng(seed)

        steps = []
        total_reward = 0.0
        step = 0
        max_steps = max_steps or env.max_steps

        while step < max_steps:
            action = agent.act(obs)
            internal_state = agent.get_internal_state()
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Perturb reward based on position (misleading shaping)
            ax, ay = env.agent_position
            center = self.grid_size / 2

            # Add misleading bonus for being in corners (suboptimal)
            corner_bonus = 0
            if (ax < 5 and ay < 5) or (ax > self.grid_size - 5 and ay > self.grid_size - 5):
                corner_bonus = self.perturbation_strength * 0.5

            # Add penalty for center (where resources typically are)
            center_dist = np.sqrt((ax - center) ** 2 + (ay - center) ** 2)
            center_penalty = self.perturbation_strength * (0.1 - center_dist / self.grid_size * 0.1)

            perturbed_reward = reward + corner_bonus + center_penalty
            total_reward += perturbed_reward

            from src.utils.logging import StepLog
            steps.append(StepLog(
                step=step,
                action=action,
                position=tuple(env.agent_position),
                energy=env.agent_energy,
                reward=perturbed_reward,
                internal_state=internal_state,
                info={**info, "original_reward": reward},
            ))

            agent.set_energy(env.agent_energy)

            if terminated or truncated:
                break

            obs = next_obs
            step += 1

        return EpisodeLog(
            episode_id=0,
            agent_type=agent.name,
            environment_mode=str(env.mode),
            total_steps=step + 1,
            total_reward=total_reward,
            final_energy=env.agent_energy,
            survived=not (env.agent_energy <= 0),
            steps=steps,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute robustness to reward perturbation."""
        # Measure survival and energy maintenance despite misleading rewards
        survival_rate = np.mean([e.survived for e in test_episodes])
        mean_energy = np.mean([e.final_energy for e in test_episodes])
        resources_consumed = np.mean([
            e.steps[-1].info.get("resources_consumed", 0) if e.steps else 0
            for e in test_episodes
        ])

        # Score: survival and resource acquisition despite misleading shaping
        score = survival_rate * 0.5 + (mean_energy / 150.0) * 0.3 + min(resources_consumed / 5, 1) * 0.2

        raw_data = {
            "survival_rate": float(survival_rate),
            "mean_final_energy": float(mean_energy),
            "mean_resources_consumed": float(resources_consumed),
        }

        return float(score), raw_data


class RLAdversarialTest(BasePredicateTest):
    """
    Composite adversarial test suite favoring RL.

    Combines multiple conditions where RL should excel.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="rl_adversarial_composite",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size

        # Initialize component tests
        self.credit_test = TemporalCreditAssignmentTest(
            n_trials=n_trials, seed=seed
        )
        self.stochastic_test = StochasticRewardTest(
            n_trials=n_trials, seed=seed
        )
        self.robustness_test = RewardShapingRobustnessTest(
            n_trials=n_trials, seed=seed
        )

    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            initial_energy=150.0,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def create_test_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        return self.create_baseline_env(seed)

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute composite adversarial score."""
        # Simple average for this baseline implementation
        score = np.mean([e.total_reward for e in test_episodes])

        raw_data = {
            "mean_reward": float(score),
            "survival_rate": float(np.mean([e.survived for e in test_episodes])),
        }

        return float(score), raw_data

    def run(self, agent: BaseAgent) -> TestResult:
        """Run all adversarial tests."""
        results = {}

        # Run each component test
        results["temporal_credit"] = self.credit_test.run(agent)
        results["stochastic_reward"] = self.stochastic_test.run(agent)
        results["reward_robustness"] = self.robustness_test.run(agent)

        # Compute composite score
        scores = [r.score for r in results.values()]
        mean_score = float(np.mean(scores))

        return TestResult(
            test_name=self.name,
            agent_name=agent.name,
            score=mean_score,
            raw_data={
                "component_scores": {name: r.score for name, r in results.items()},
                "component_results": {name: r.to_dict() for name, r in results.items()},
            },
            metadata={
                "n_trials": self.n_trials,
                "tests_run": list(results.keys()),
            },
        )


def run_adversarial_tests(
    agents: List[BaseAgent],
    n_trials: int = 10,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, TestResult]:
    """Run adversarial predicate tests on multiple agents."""
    test = RLAdversarialTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
