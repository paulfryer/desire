"""
State-Dependent Valuation (SDV) test implementation.
"""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


class StateDependentValuationTest(BasePredicateTest):
    """
    Test for State-Dependent Valuation (SDV).

    Measures whether agents value resources differently based on their
    internal state (energy level) - a behavioral signature indicating
    that goals are pursued based on need rather than just availability.

    Protocol:
    1. Measure resource-seeking behavior at different energy levels
    2. Compare urgency/speed of approach when hungry vs satiated

    SDV Score = correlation(energy, resource_seeking_intensity)

    Negative SDV scores indicate more seeking when hungry (expected).
    Desire-theoretic prediction: Active inference agents should show
    stronger SDV (more state-dependent) because they explicitly model
    homeostatic setpoints and use interoceptive signals.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        high_energy_threshold: float = 0.8,  # As fraction of max
        low_energy_threshold: float = 0.4,   # As fraction of max
        grid_size: int = 20,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize the State-Dependent Valuation test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Episodes for baseline
            test_episodes: Episodes for measurement
            high_energy_threshold: Threshold for "satiated" state
            low_energy_threshold: Threshold for "hungry" state
            grid_size: Environment grid size
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="state_dependent_valuation",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.high_energy_threshold = high_energy_threshold
        self.low_energy_threshold = low_energy_threshold
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
        """Create test environment (same as baseline)."""
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
        Compute the State-Dependent Valuation score.

        SDV = |correlation(energy, seeking_intensity)|
        """
        # Collect behavior data at different energy levels
        high_energy_intensities = []
        low_energy_intensities = []
        energy_intensity_pairs = []

        max_energy = 150.0  # From environment config

        for episode in test_episodes:
            for i, step in enumerate(episode.steps):
                energy_fraction = step.energy / max_energy

                # Compute local seeking intensity (using nearby steps)
                window_start = max(0, i - 5)
                window_end = min(len(episode.steps), i + 5)
                window_steps = episode.steps[window_start:window_end]

                # Seeking intensity = rate of movement actions
                if window_steps:
                    movement_actions = sum(
                        1 for s in window_steps
                        if s.action in [0, 1, 2, 3]  # Movement actions
                    )
                    intensity = movement_actions / len(window_steps)
                else:
                    intensity = 0.0

                energy_intensity_pairs.append((energy_fraction, intensity))

                # Categorize by energy level
                if energy_fraction >= self.high_energy_threshold:
                    high_energy_intensities.append(intensity)
                elif energy_fraction <= self.low_energy_threshold:
                    low_energy_intensities.append(intensity)

        # Compute SDV score
        if not high_energy_intensities or not low_energy_intensities:
            return 0.0, {"error": "Insufficient energy variation"}

        mean_high = float(np.mean(high_energy_intensities))
        mean_low = float(np.mean(low_energy_intensities))

        # Intensity difference (expected: higher intensity when hungry)
        intensity_diff = mean_low - mean_high

        # Compute correlation
        if len(energy_intensity_pairs) > 10:
            energies = np.array([e for e, _ in energy_intensity_pairs])
            intensities = np.array([i for _, i in energy_intensity_pairs])
            correlation = float(np.corrcoef(energies, intensities)[0, 1])
        else:
            correlation = 0.0

        # SDV score = magnitude of negative correlation (more negative = more state-dependent)
        # We want higher scores to mean more state-dependent, so use absolute value
        # but sign should indicate correct direction
        sdv_score = abs(correlation) if correlation < 0 else -abs(correlation)

        # Alternatively, use normalized intensity difference
        if mean_high + mean_low > 0:
            relative_diff = (mean_low - mean_high) / (mean_high + mean_low)
        else:
            relative_diff = 0.0

        # Final score combines both metrics
        sdv_score = (abs(correlation) + abs(relative_diff)) / 2 if correlation < 0 else 0.0

        raw_data = {
            "mean_high_energy_intensity": mean_high,
            "mean_low_energy_intensity": mean_low,
            "intensity_difference": float(intensity_diff),
            "relative_difference": float(relative_diff),
            "energy_intensity_correlation": correlation,
            "n_high_samples": len(high_energy_intensities),
            "n_low_samples": len(low_energy_intensities),
        }

        return sdv_score, raw_data


def run_state_dependent_test(
    agents: list[BaseAgent],
    n_trials: int = 10,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, TestResult]:
    """Run the State-Dependent Valuation test on multiple agents."""
    test = StateDependentValuationTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
