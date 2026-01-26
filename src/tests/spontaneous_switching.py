"""
Spontaneous Goal Switching test implementation.

Tests whether agents can flexibly switch goals when better
opportunities arise, even mid-pursuit of original goal.
"""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, ModeConfig
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


class SpontaneousSwitchingTest(BasePredicateTest):
    """
    Test for Spontaneous Goal Switching.

    Measures whether agents can switch to better opportunities that
    arise while pursuing an initial goal.

    Protocol:
    1. Setup: Agent starts pursuing a distant resource
    2. Intervention: A closer, better resource appears mid-pursuit
    3. Measure: Does agent switch to the new opportunity?

    Flexibility prediction: Agents with desire-like representations
    should flexibly update goals based on changing opportunities,
    while rigid RL may persist with original plan.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        switch_step: int = 50,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize Spontaneous Switching test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Baseline episodes
            test_episodes: Test episodes
            grid_size: Environment grid size
            switch_step: Step when new opportunity appears
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="spontaneous_switching",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.switch_step = switch_step

        # Original target (far corner)
        self.original_target = (grid_size - 3, grid_size - 3)

        # Better opportunity (appears mid-pursuit, closer to start)
        self.opportunity_target = (grid_size // 2 - 2, grid_size // 2 + 2)

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create baseline environment with only original target."""
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Single target baseline",
            num_resources=1,
            resource_types=["renewable"],
            resource_positions=[self.original_target],
            agent_start_position=(2, 2),
        )

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=300,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.BASELINE,
            mode_config=config,
            seed=seed,
        )

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create test environment with opportunity that appears."""
        # Initially same as baseline
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Switching opportunity test",
            num_resources=2,
            resource_types=["renewable", "renewable"],
            resource_positions=[self.original_target, self.opportunity_target],
            agent_start_position=(2, 2),
        )

        env = ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=300,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.BASELINE,
            mode_config=config,
            seed=seed,
        )

        # Hide the opportunity resource initially
        if len(env.resources) > 1:
            env.resources[1].is_active = False

        return env

    def _run_episode(
        self,
        env: ForagingGridEnv,
        agent: BaseAgent,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> EpisodeLog:
        """Run episode with opportunity appearing mid-episode."""
        obs, info = env.reset(seed=seed)
        agent.reset()

        steps = []
        total_reward = 0.0
        step = 0
        max_steps = max_steps or env.max_steps

        opportunity_revealed = False

        while step < max_steps:
            # Reveal opportunity at switch_step
            if step == self.switch_step and not opportunity_revealed:
                if len(env.resources) > 1:
                    env.resources[1].is_active = True
                    env.resources[1].energy_value = 30.0  # Better value
                opportunity_revealed = True

            action = agent.act(obs)
            internal_state = agent.get_internal_state()
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            from src.utils.logging import StepLog
            steps.append(StepLog(
                step=step,
                action=action,
                position=tuple(env.agent_position),
                energy=env.agent_energy,
                reward=reward,
                internal_state=internal_state,
                info={**info, "opportunity_revealed": opportunity_revealed},
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
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute Spontaneous Switching score.

        Score based on:
        1. Switch rate: How often agent switches to opportunity
        2. Switch efficiency: How quickly after opportunity appears
        """
        switched_to_opportunity = []
        switch_times = []
        reached_any_target = []

        opp_pos = np.array(self.opportunity_target)
        orig_pos = np.array(self.original_target)

        for episode in test_episodes:
            trajectory = episode.get_trajectory()

            # Find position after opportunity revealed
            post_reveal_traj = trajectory[self.switch_step:] if len(trajectory) > self.switch_step else []

            if len(post_reveal_traj) == 0:
                continue

            # Track distance to both targets
            distances_to_opp = [np.linalg.norm(np.array(pos) - opp_pos) for pos in post_reveal_traj]
            distances_to_orig = [np.linalg.norm(np.array(pos) - orig_pos) for pos in post_reveal_traj]

            # Did agent reach opportunity target?
            reached_opp = any(d < 1.5 for d in distances_to_opp)
            reached_orig = any(d < 1.5 for d in distances_to_orig)

            reached_any_target.append(reached_opp or reached_orig)

            # Did agent switch direction toward opportunity?
            if len(distances_to_opp) > 10:
                early_dist = np.mean(distances_to_opp[:5])
                later_dist = np.mean(distances_to_opp[5:15])

                switched = later_dist < early_dist - 1  # Meaningful movement toward
                switched_to_opportunity.append(switched)

                if switched:
                    # Find when switch happened
                    for i, d in enumerate(distances_to_opp):
                        if d < early_dist - 1:
                            switch_times.append(i + self.switch_step)
                            break

        # Compute metrics
        switch_rate = np.mean(switched_to_opportunity) if switched_to_opportunity else 0.0
        mean_switch_time = np.mean(switch_times) if switch_times else 300
        target_rate = np.mean(reached_any_target) if reached_any_target else 0.0

        # Score: switch rate + bonus for quick switching
        score = switch_rate
        if mean_switch_time < 150:
            score += (150 - mean_switch_time) / 300  # Bonus for quick switch

        raw_data = {
            "switch_rate": float(switch_rate),
            "mean_switch_time": float(mean_switch_time),
            "target_rate": float(target_rate),
            "n_switched": sum(switched_to_opportunity),
            "n_episodes": len(test_episodes),
        }

        return float(score), raw_data


class GoalFlexibilityTest(BasePredicateTest):
    """
    Test for general goal flexibility.

    Measures how well agents adapt when goals change or become unavailable.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        goal_change_step: int = 100,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        super().__init__(
            name="goal_flexibility",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.goal_change_step = goal_change_step

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create baseline environment."""
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=300,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create test environment (goal positions will be manipulated)."""
        return self.create_baseline_env(seed)

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """Compute goal flexibility score."""
        # Compare performance before and after goal change
        pre_change_rewards = []
        post_change_rewards = []

        for episode in test_episodes:
            rewards = episode.get_reward_curve()

            if len(rewards) > self.goal_change_step:
                pre = np.sum(rewards[:self.goal_change_step])
                post = np.sum(rewards[self.goal_change_step:])

                pre_change_rewards.append(pre)
                post_change_rewards.append(post)

        if not pre_change_rewards or not post_change_rewards:
            return 0.0, {"error": "Insufficient episode length"}

        # Flexibility score: ability to maintain performance after change
        pre_mean = np.mean(pre_change_rewards)
        post_mean = np.mean(post_change_rewards)

        # Normalize by episode length ratio
        pre_steps = self.goal_change_step
        post_steps = 300 - self.goal_change_step

        pre_rate = pre_mean / pre_steps if pre_steps > 0 else 0
        post_rate = post_mean / post_steps if post_steps > 0 else 0

        if pre_rate > 0:
            flexibility = post_rate / pre_rate
        else:
            flexibility = 1.0

        score = min(1.0, flexibility)

        raw_data = {
            "pre_change_mean": float(pre_mean),
            "post_change_mean": float(post_mean),
            "pre_rate": float(pre_rate),
            "post_rate": float(post_rate),
            "flexibility_ratio": float(flexibility),
        }

        return float(score), raw_data


def run_switching_test(
    agents: list[BaseAgent],
    n_trials: int = 10,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, TestResult]:
    """Run Spontaneous Switching test on multiple agents."""
    test = SpontaneousSwitchingTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
