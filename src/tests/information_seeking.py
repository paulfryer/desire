"""
Information Seeking test implementation.

Tests whether agents engage in exploration without immediate reward,
a hallmark of curiosity-driven behavior.
"""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, ModeConfig
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


class InformationSeekingTest(BasePredicateTest):
    """
    Test for Information Seeking behavior.

    Measures whether agents explore novel areas of the environment
    even when not immediately necessary for survival.

    Protocol:
    1. Baseline: Agent in familiar environment with visible resources
    2. Test: Agent in same environment but with hidden novel region
    3. Measure: Exploration of novel region vs staying in familiar area

    Information-seeking prediction: Active inference agents should
    explore novel areas more (reduce uncertainty) while pure RL
    may stay in familiar, reward-rich areas.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        novel_region_size: int = 6,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize Information Seeking test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Episodes for baseline measurement
            test_episodes: Episodes for test condition
            grid_size: Environment grid size
            novel_region_size: Size of the novel region to explore
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="information_seeking",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size
        self.novel_region_size = novel_region_size

        # Define novel region (upper right corner)
        self.novel_region = (
            (0, grid_size - novel_region_size),  # (x_start, y_start)
            (novel_region_size, grid_size),       # (x_end, y_end)
        )

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create baseline environment with resources in familiar area."""
        # Resources only in familiar (lower left) area
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Familiar area with resources",
            num_resources=5,
            resource_types=["renewable"] * 5,
            resource_positions=[
                (self.grid_size // 2 + 2, self.grid_size // 4),
                (self.grid_size // 2 - 2, self.grid_size // 4),
                (self.grid_size * 3 // 4, self.grid_size // 2),
                (self.grid_size // 2, self.grid_size // 2),
                (self.grid_size // 4, self.grid_size // 2),
            ],
            agent_start_position=(self.grid_size // 2, self.grid_size // 2),
        )

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            initial_energy=200.0,
            max_energy=200.0,
            energy_decay_rate=0.2,  # Low decay to allow exploration
            mode=EnvironmentMode.BASELINE,
            mode_config=config,
            seed=seed,
        )

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create test environment with novel region containing resources."""
        # Resources in both familiar and novel regions
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Environment with novel region",
            num_resources=7,
            resource_types=["renewable"] * 7,
            resource_positions=[
                # Familiar area
                (self.grid_size // 2 + 2, self.grid_size // 4),
                (self.grid_size // 2 - 2, self.grid_size // 4),
                (self.grid_size * 3 // 4, self.grid_size // 2),
                # Novel region (upper right)
                (2, self.grid_size - 3),
                (3, self.grid_size - 5),
                (1, self.grid_size - 2),
                (4, self.grid_size - 4),
            ],
            agent_start_position=(self.grid_size // 2, self.grid_size // 2),
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
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute Information Seeking score.

        Score = novel_region_visits / total_steps
        Higher scores indicate more exploration of novel areas.
        """
        novel_visits = 0
        total_steps = 0
        novel_entry_times = []

        x_start, y_start = self.novel_region[0]
        x_end, y_end = self.novel_region[1]

        for episode in test_episodes:
            trajectory = episode.get_trajectory()
            total_steps += len(trajectory)

            entered_novel = False
            for step_idx, pos in enumerate(trajectory):
                x, y = pos
                if x_start <= x < x_end and y_start <= y < y_end:
                    novel_visits += 1
                    if not entered_novel:
                        entered_novel = True
                        novel_entry_times.append(step_idx)

        # Compute score (fraction of time in novel region)
        if total_steps > 0:
            exploration_rate = novel_visits / total_steps
        else:
            exploration_rate = 0.0

        # Compute average entry time (earlier = more curious)
        if novel_entry_times:
            mean_entry_time = np.mean(novel_entry_times)
            entry_rate = len(novel_entry_times) / len(test_episodes)
        else:
            mean_entry_time = np.inf
            entry_rate = 0.0

        # Combined score: exploration rate + bonus for early entry
        score = exploration_rate * 10  # Scale for visibility
        if mean_entry_time < 100:
            score += (100 - mean_entry_time) / 100  # Bonus for early entry

        raw_data = {
            "novel_visits": novel_visits,
            "total_steps": total_steps,
            "exploration_rate": exploration_rate,
            "mean_entry_time": float(mean_entry_time) if mean_entry_time != np.inf else -1,
            "entry_rate": entry_rate,
            "n_episodes": len(test_episodes),
        }

        return float(score), raw_data


class PureExplorationTest(BasePredicateTest):
    """
    Test for exploration in absence of reward.

    Measures exploration behavior when no resources are present,
    testing intrinsic motivation vs reward-driven behavior.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        grid_size: int = 20,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        super().__init__(
            name="pure_exploration",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.grid_size = grid_size

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create baseline environment with resources."""
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=300,
            initial_energy=200.0,
            max_energy=200.0,
            energy_decay_rate=0.1,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create environment with NO resources."""
        config = ModeConfig(
            mode=EnvironmentMode.BASELINE,
            description="Empty environment for exploration",
            num_resources=0,
            resource_types=[],
            resource_positions=[],
            agent_start_position=(self.grid_size // 2, self.grid_size // 2),
        )

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=300,
            initial_energy=200.0,
            max_energy=200.0,
            energy_decay_rate=0.1,
            mode=EnvironmentMode.BASELINE,
            mode_config=config,
            seed=seed,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute exploration score based on area coverage.

        Score = unique_positions_visited / total_possible_positions
        """
        all_positions_visited = set()
        total_movement = 0

        for episode in test_episodes:
            trajectory = episode.get_trajectory()
            for pos in trajectory:
                all_positions_visited.add((int(pos[0]), int(pos[1])))

            # Count movement actions
            actions = episode.get_action_sequence()
            total_movement += np.sum(actions < 4)  # Actions 0-3 are movement

        # Coverage score
        total_cells = self.grid_size * self.grid_size
        coverage = len(all_positions_visited) / total_cells

        # Movement rate (exploration activity)
        total_steps = sum(e.total_steps for e in test_episodes)
        movement_rate = total_movement / total_steps if total_steps > 0 else 0

        # Combined score
        score = coverage + movement_rate * 0.5

        raw_data = {
            "unique_positions": len(all_positions_visited),
            "total_cells": total_cells,
            "coverage": coverage,
            "movement_rate": movement_rate,
            "total_movement": total_movement,
        }

        return float(score), raw_data


def run_information_seeking_test(
    agents: list[BaseAgent],
    n_trials: int = 10,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, TestResult]:
    """Run Information Seeking test on multiple agents."""
    test = InformationSeekingTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
