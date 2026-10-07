"""
Means-End Decoupling (MED) test implementation.
"""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, ModeConfig
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


class MeansEndDecouplingTest(BasePredicateTest):
    """
    Test for Means-End Decoupling (MED).

    Measures whether agents can flexibly switch to alternative paths
    when their primary path is blocked - indicating separation of
    goal representation from action sequence.

    Protocol:
    1. Training: Agent learns Path P1 to Goal G
    2. Block P1, provide alternative P2
    3. Measure: Transfer efficiency (time to reach G via P2)

    MED Score = optimal_P2_time / actual_P2_time

    A higher MED score (closer to 1.0) indicates better means-end
    decoupling. Desire-theoretic prediction: Active inference agents
    should show higher MED scores due to explicit goal representation.
    """

    def __init__(
        self,
        n_trials: int = 10,
        training_episodes: int = 100,
        transfer_episodes: int = 50,
        grid_size: int = 20,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize the Means-End Decoupling test.

        Args:
            n_trials: Number of independent trials
            training_episodes: Episodes for learning P1
            transfer_episodes: Episodes for testing P2
            grid_size: Environment grid size
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="means_end_decoupling",
            n_trials=n_trials,
            baseline_episodes=training_episodes,
            test_episodes=transfer_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.training_episodes = training_episodes
        self.transfer_episodes = transfer_episodes
        self.grid_size = grid_size

        # Track optimal path lengths for scoring
        self.optimal_p1_length = self._compute_optimal_path_length(blocked=False)
        self.optimal_p2_length = self._compute_optimal_path_length(blocked=True)

    def _compute_optimal_path_length(self, blocked: bool) -> int:
        """Compute optimal path length for the environment configuration."""
        # In PATH_BLOCKAGE mode:
        # Agent starts at (grid_size//4, center)
        # Resource at (3*grid_size//4, center)
        # Wall at center with gap

        center = self.grid_size // 2
        start_x = self.grid_size // 4
        goal_x = 3 * self.grid_size // 4

        if not blocked:
            # Direct path through gap
            # Distance = |goal_x - start_x|
            return goal_x - start_x
        else:
            # Must go around the wall
            # Go to edge, around, back to center
            # Approximate: direct distance + 2 * distance to edge
            direct = goal_x - start_x
            detour = 2 * (center - 2)  # Distance to go around
            return direct + detour

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create training environment with open path P1."""
        mode_config = self._create_mode_config(blocked=False)

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            mode=EnvironmentMode.PATH_BLOCKAGE,
            mode_config=mode_config,
            seed=seed,
        )

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create test environment with P1 blocked, P2 open."""
        mode_config = self._create_mode_config(blocked=True)

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            mode=EnvironmentMode.PATH_BLOCKAGE,
            mode_config=mode_config,
            seed=seed,
        )

    def _create_mode_config(self, blocked: bool) -> ModeConfig:
        """Create mode configuration for the MED test."""
        center = self.grid_size // 2

        # Wall with gap configuration
        wall_start = (center, 2)
        wall_end = (center, self.grid_size - 3)
        gap_y = center

        # Resource position
        resource_pos = (3 * self.grid_size // 4, center)

        # Agent start position
        agent_start = (self.grid_size // 4, center)

        # Dynamic obstacle at gap
        dynamic_positions = [(center, gap_y)]

        config = ModeConfig(
            mode=EnvironmentMode.PATH_BLOCKAGE,
            description="MED test configuration",
            num_resources=1,
            resource_types=["renewable"],
            resource_positions=[resource_pos],
            add_obstacles=True,
            wall_positions=[(wall_start, wall_end)],
            dynamic_obstacle_positions=dynamic_positions,
            agent_start_position=agent_start,
            blockage_step=0 if blocked else 100000,  # Immediate or never
        )

        return config

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute the Means-End Decoupling score.

        MED = optimal_path_length / mean(actual_path_length)
        """
        # Compute how efficiently agent reaches goal in training (P1)
        p1_times = self._compute_goal_reach_times(baseline_data.episodes)

        # Compute how efficiently agent reaches goal in transfer (P2)
        p2_times = self._compute_goal_reach_times(test_episodes)

        # Compute MED score
        if not p2_times:
            # Never reached goal
            med_score = 0.0
        else:
            mean_p2_time = np.mean(p2_times)
            if mean_p2_time > 0:
                med_score = self.optimal_p2_length / mean_p2_time
            else:
                med_score = 1.0

        # Clamp to [0, 1]
        med_score = np.clip(med_score, 0.0, 1.0)

        # Additional metrics
        raw_data = {
            "p1_reach_times": p1_times,
            "p2_reach_times": p2_times,
            "mean_p1_time": float(np.mean(p1_times)) if p1_times else None,
            "mean_p2_time": float(np.mean(p2_times)) if p2_times else None,
            "p1_success_rate": len(p1_times) / len(baseline_data.episodes) if baseline_data.episodes else 0,
            "p2_success_rate": len(p2_times) / len(test_episodes) if test_episodes else 0,
            "optimal_p1_length": self.optimal_p1_length,
            "optimal_p2_length": self.optimal_p2_length,
            "transfer_efficiency": self._compute_transfer_efficiency(p1_times, p2_times),
        }

        return med_score, raw_data

    def _compute_goal_reach_times(self, episodes: list[EpisodeLog]) -> list[int]:
        """
        Compute time to first resource consumption for each episode.

        Returns list of step counts at which goal was reached.
        Empty list entries are episodes where goal was never reached.
        """
        reach_times = []

        for episode in episodes:
            for step in episode.steps:
                # info is computed after the step, and a successful consume deactivates the
                # resource, so "on_resource" is always False here; use the consumption counter.
                if step.info.get("resources_consumed", 0) > 0:
                    reach_times.append(step.step)
                    break
            # If loop completes without break, goal was not reached

        return reach_times

    def _compute_transfer_efficiency(
        self,
        p1_times: list[int],
        p2_times: list[int],
    ) -> float:
        """
        Compute transfer efficiency ratio.

        Measures how well training on P1 transfers to P2.
        """
        if not p1_times or not p2_times:
            return 0.0

        # Adjust for different optimal path lengths
        normalized_p1 = np.mean(p1_times) / self.optimal_p1_length
        normalized_p2 = np.mean(p2_times) / self.optimal_p2_length

        # Transfer efficiency: how close P2 performance is to P1
        if normalized_p2 > 0:
            return normalized_p1 / normalized_p2
        return 0.0


def run_means_end_test(
    agents: list[BaseAgent],
    n_trials: int = 10,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, TestResult]:
    """
    Run the Means-End Decoupling test on multiple agents.

    Args:
        agents: List of agents to test
        n_trials: Number of trials per agent
        seed: Random seed
        output_dir: Output directory

    Returns:
        Dictionary mapping agent name to TestResult
    """
    test = MeansEndDecouplingTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
