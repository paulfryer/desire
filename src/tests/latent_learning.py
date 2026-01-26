"""
Latent Learning test implementation.

Tests whether agents learn about the environment structure
even without immediate reward, then use that knowledge later.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, ModeConfig
from src.tests.base_test import BasePredicateTest, BaselineData, TestResult
from src.utils.logging import EpisodeLog


class LatentLearningTest(BasePredicateTest):
    """
    Test for Latent Learning.

    Based on Tolman's classic experiments: agents explore an environment
    without reward, then are tested on whether they learned its structure.

    Protocol:
    Phase 1 (Exploration): Agent explores without resources (no reward)
    Phase 2 (Test): Resources appear, measure time to find them

    Latent learning prediction: Agents that built an internal model
    during exploration (like active inference) should find resources
    faster than agents that ignored non-rewarded exploration.
    """

    def __init__(
        self,
        n_trials: int = 10,
        exploration_episodes: int = 20,
        test_episodes: int = 20,
        grid_size: int = 20,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize Latent Learning test.

        Args:
            n_trials: Number of independent trials
            exploration_episodes: Episodes for unrewarded exploration
            test_episodes: Episodes for testing
            grid_size: Environment grid size
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="latent_learning",
            n_trials=n_trials,
            baseline_episodes=exploration_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.exploration_episodes = exploration_episodes

        # Fixed resource positions for consistent testing
        self.resource_positions = [
            (3, 3),
            (3, grid_size - 4),
            (grid_size - 4, 3),
            (grid_size - 4, grid_size - 4),
        ]

    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create exploration environment (no resources)."""
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Exploration phase - no resources",
            num_resources=0,
            resource_types=[],
            resource_positions=[],
            agent_start_position=(self.grid_size // 2, self.grid_size // 2),
        )

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=200,
            initial_energy=200.0,
            max_energy=200.0,
            energy_decay_rate=0.1,  # Low decay for exploration
            mode=EnvironmentMode.BASELINE,
            mode_config=config,
            seed=seed,
        )

    def create_test_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create test environment with resources at corners."""
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Test phase - resources at corners",
            num_resources=4,
            resource_types=["renewable"] * 4,
            resource_positions=self.resource_positions,
            agent_start_position=(self.grid_size // 2, self.grid_size // 2),
        )

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=200,
            initial_energy=100.0,
            max_energy=100.0,
            energy_decay_rate=0.5,  # Higher decay to motivate finding food
            mode=EnvironmentMode.BASELINE,
            mode_config=config,
            seed=seed,
        )

    def _collect_baseline(
        self,
        agent: BaseAgent,
        seed: Optional[int] = None,
    ) -> BaselineData:
        """
        Collect exploration data (baseline phase).

        This phase allows agent to explore without resources.
        We track what positions they visit.
        """
        env = self.create_baseline_env(seed)
        episodes = []
        visited_positions = set()

        for ep in range(self.exploration_episodes):
            episode = self._run_episode(env, agent, seed=seed)
            episodes.append(episode)

            # Track visited positions
            trajectory = episode.get_trajectory()
            for pos in trajectory:
                visited_positions.add((int(pos[0]), int(pos[1])))

        # Compute baseline statistics
        total_movement = sum(
            np.sum(e.get_action_sequence() < 4) for e in episodes
        )
        total_steps = sum(e.total_steps for e in episodes)

        return BaselineData(
            mean_actions_per_step=total_movement / total_steps if total_steps > 0 else 0,
            std_actions_per_step=0.0,
            mean_energy=np.mean([e.final_energy for e in episodes]),
            mean_path_length=total_movement / self.exploration_episodes,
            action_distribution=np.zeros(6),
            episodes=episodes,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Latent Learning score.

        Measures how quickly agent finds resources after exploration.
        Lower time-to-first-resource indicates better latent learning.
        """
        times_to_resource = []
        resources_found = []

        for episode in test_episodes:
            time_to_first = None

            for step_idx, step in enumerate(episode.steps):
                if step.info.get("resources_consumed", 0) > 0:
                    time_to_first = step_idx
                    break

            if time_to_first is not None:
                times_to_resource.append(time_to_first)
                resources_found.append(True)
            else:
                times_to_resource.append(episode.total_steps)
                resources_found.append(False)

        # Score: inverse of mean time to resource (normalized)
        mean_time = np.mean(times_to_resource)
        max_time = 200  # max_steps

        # Higher score = faster resource finding = better latent learning
        score = (max_time - mean_time) / max_time

        # Bonus for finding resources in most episodes
        find_rate = np.mean(resources_found)
        score = score * (0.5 + 0.5 * find_rate)

        # Check if exploration covered resource locations
        exploration_episodes = baseline_data.episodes
        positions_explored = set()
        for ep in exploration_episodes:
            traj = ep.get_trajectory()
            for pos in traj:
                positions_explored.add((int(pos[0]), int(pos[1])))

        # How many resource positions were visited during exploration?
        resource_areas_explored = 0
        for rx, ry in self.resource_positions:
            # Check if near the resource position (within 2 cells)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if (rx + dx, ry + dy) in positions_explored:
                        resource_areas_explored += 1
                        break

        exploration_coverage = resource_areas_explored / len(self.resource_positions)

        raw_data = {
            "mean_time_to_resource": float(mean_time),
            "find_rate": float(find_rate),
            "times_to_resource": times_to_resource,
            "exploration_coverage": exploration_coverage,
            "positions_explored": len(positions_explored),
        }

        return float(score), raw_data


class DelayedRewardTest(BasePredicateTest):
    """
    Test for learning with delayed rewards.

    Agent must learn environment structure during exploration,
    then use knowledge when reward becomes available.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        reward_delay_steps: int = 100,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="delayed_reward",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.reward_delay_steps = reward_delay_steps

    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create baseline with immediate rewards."""
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=300,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def create_test_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create test environment (same as baseline for this test)."""
        return self.create_baseline_env(seed)

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute delayed reward learning score."""
        # Compare performance early vs late in episodes
        early_rewards = []
        late_rewards = []

        for episode in test_episodes:
            rewards = episode.get_reward_curve()
            n_steps = len(rewards)

            if n_steps > self.reward_delay_steps:
                early = rewards[:self.reward_delay_steps]
                late = rewards[self.reward_delay_steps:]

                early_rewards.append(np.sum(early))
                late_rewards.append(np.sum(late))

        if not early_rewards or not late_rewards:
            return 0.0, {"error": "Insufficient episode length"}

        # Learning score: improvement from early to late
        early_mean = np.mean(early_rewards)
        late_mean = np.mean(late_rewards)

        if early_mean > 0:
            improvement = (late_mean - early_mean) / early_mean
        else:
            improvement = late_mean

        score = max(0, improvement + 1)  # Shift to positive range

        raw_data = {
            "early_mean_reward": float(early_mean),
            "late_mean_reward": float(late_mean),
            "improvement_ratio": float(improvement),
        }

        return float(score), raw_data


def run_latent_learning_test(
    agents: List[BaseAgent],
    n_trials: int = 10,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, TestResult]:
    """Run Latent Learning test on multiple agents."""
    test = LatentLearningTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
