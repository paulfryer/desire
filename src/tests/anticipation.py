"""
Anticipatory Activation (AA) test implementation.
"""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


class AnticipatoryActivationTest(BasePredicateTest):
    """
    Test for Anticipatory Activation (AA).

    Measures whether agents show increased internal state activity when
    approaching a goal (before actually receiving reward) - a behavioral
    signature that may indicate prospective desire-like states.

    Protocol:
    1. Run episodes where agent approaches and reaches resources
    2. Track internal state activation as function of distance to goal
    3. Measure: Activation increase before goal vs far from goal

    AA Score = mean(activation when near goal) / mean(activation when far)

    Higher AA scores indicate more anticipatory activation.
    Desire-theoretic prediction: Active inference agents should show
    higher AA scores (more anticipatory activation) because they
    explicitly represent future expected states.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        near_threshold: int = 3,  # Distance considered "near" goal
        far_threshold: int = 8,   # Distance considered "far" from goal
        grid_size: int = 20,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize the Anticipatory Activation test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Episodes for baseline
            test_episodes: Episodes for measurement
            near_threshold: Distance threshold for "near" goal
            far_threshold: Distance threshold for "far" from goal
            grid_size: Environment grid size
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="anticipatory_activation",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.grid_size = grid_size

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
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

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create test environment (same as baseline for AA test)."""
        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.BASELINE,
            seed=seed,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute the Anticipatory Activation score.

        AA = mean(activation near goal) / mean(activation far from goal)
        """
        # Collect activations by distance
        near_activations = []
        far_activations = []
        all_distance_activations = []

        for episode in test_episodes:
            for step in episode.steps:
                info = step.info
                internal_state = step.internal_state

                # Skip if no internal state or distance info
                if internal_state is None:
                    continue

                # Compute activation magnitude
                activation = np.linalg.norm(internal_state)

                # Get distance to nearest resource
                distance = info.get("nearest_resource_distance")
                if distance is None:
                    continue

                all_distance_activations.append((distance, activation))

                if distance <= self.near_threshold:
                    near_activations.append(activation)
                elif distance >= self.far_threshold:
                    far_activations.append(activation)

        # Compute AA score
        if not near_activations or not far_activations:
            return 1.0, {"error": "Insufficient distance data"}

        mean_near = float(np.mean(near_activations))
        mean_far = float(np.mean(far_activations))

        # Avoid division by zero
        if mean_far < 0.001:
            mean_far = 0.001

        aa_score = mean_near / mean_far

        # Compute correlation between distance and activation
        if len(all_distance_activations) > 10:
            distances = np.array([d for d, _ in all_distance_activations])
            activations = np.array([a for _, a in all_distance_activations])
            correlation = float(np.corrcoef(distances, activations)[0, 1])
        else:
            correlation = 0.0

        raw_data = {
            "mean_near_activation": mean_near,
            "mean_far_activation": mean_far,
            "std_near_activation": float(np.std(near_activations)),
            "std_far_activation": float(np.std(far_activations)),
            "n_near_samples": len(near_activations),
            "n_far_samples": len(far_activations),
            "distance_activation_correlation": correlation,
        }

        return aa_score, raw_data


def run_anticipation_test(
    agents: list[BaseAgent],
    n_trials: int = 10,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, TestResult]:
    """Run the Anticipatory Activation test on multiple agents."""
    test = AnticipatoryActivationTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
