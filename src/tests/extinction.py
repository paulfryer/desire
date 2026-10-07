"""
Extinction Resistance (ER) test implementation.
"""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, get_mode_config
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


class ExtinctionResistanceTest(BasePredicateTest):
    """
    Test for Extinction Resistance (ER).

    Measures how long agents persist in performing a previously-rewarded
    behavior after rewards are removed - a behavioral signature that may
    indicate desire-like states vs pure reward-tracking.

    Protocol:
    1. Training: Agent learns to collect resources at a location
    2. Extinction: Remove resources (phantom resources give no reward)
    3. Measure: Persistence of resource-seeking behavior over time

    ER Score = Area under the response curve during extinction

    Higher ER scores indicate more persistent behavior.
    Desire-theoretic prediction: Active inference agents should show
    higher ER scores (more resistance to extinction) than RL agents
    that track reward directly.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        extinction_start: int = 100,
        extinction_duration: int = 300,
        grid_size: int = 20,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize the Extinction Resistance test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Episodes for training phase
            test_episodes: Episodes during extinction
            extinction_start: Step at which extinction begins
            extinction_duration: Duration of extinction phase
            grid_size: Environment grid size
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="extinction_resistance",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.extinction_start = extinction_start
        self.extinction_duration = extinction_duration
        self.grid_size = grid_size

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create baseline environment with normal resources."""
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
        """Create extinction environment with phantom resources."""
        # Use phantom mode where resources look real but give no energy
        mode_config = get_mode_config(EnvironmentMode.PHANTOM, self.grid_size)

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=self.extinction_start + self.extinction_duration,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.PHANTOM,
            mode_config=mode_config,
            seed=seed,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute the Extinction Resistance score.

        ER = normalized area under the resource-seeking response curve
        """
        # Compute baseline resource-seeking rate
        baseline_seek_rate = self._compute_resource_seeking_rate(
            baseline_data.episodes
        )

        # Compute extinction phase resource-seeking over time
        extinction_curves = []
        for episode in test_episodes:
            if len(episode.steps) > self.extinction_start:
                # Extract extinction phase
                extinction_steps = episode.steps[self.extinction_start:]
                seek_curve = self._compute_seeking_curve(
                    extinction_steps,
                    window_size=20,
                )
                extinction_curves.append(seek_curve)

        if not extinction_curves:
            return 0.0, {"error": "No extinction data"}

        # Average extinction curve
        max_len = max(len(c) for c in extinction_curves)
        padded_curves = np.zeros((len(extinction_curves), max_len))
        for i, curve in enumerate(extinction_curves):
            padded_curves[i, :len(curve)] = curve

        mean_curve = padded_curves.mean(axis=0)

        # Normalize by baseline rate
        if baseline_seek_rate > 0.01:
            normalized_curve = mean_curve / baseline_seek_rate
        else:
            normalized_curve = mean_curve

        # ER score = area under curve (normalized by duration)
        er_score = float(np.trapezoid(normalized_curve) / len(normalized_curve)) if len(normalized_curve) > 0 else 0.0

        # Additional metrics
        raw_data = {
            "baseline_seek_rate": float(baseline_seek_rate),
            "extinction_curve": mean_curve.tolist(),
            "normalized_curve": normalized_curve.tolist(),
            "half_life": self._estimate_half_life(normalized_curve),
            "final_persistence": float(normalized_curve[-1]) if len(normalized_curve) > 0 else 0.0,
        }

        return er_score, raw_data

    def _compute_resource_seeking_rate(
        self,
        episodes: list[EpisodeLog],
    ) -> float:
        """Compute the rate of resource-seeking behavior."""
        total_seeking = 0
        total_steps = 0

        for episode in episodes:
            for step in episode.steps:
                # Movement actions (0-3) toward resources count as seeking
                # Also count CONSUME attempts
                action = step.action
                if action in [0, 1, 2, 3, 5]:  # Movement or consume
                    total_seeking += 1
                total_steps += 1

        return total_seeking / total_steps if total_steps > 0 else 0.0

    def _compute_seeking_curve(
        self,
        steps: list,
        window_size: int = 20,
    ) -> np.ndarray:
        """Compute resource-seeking behavior over time windows."""
        if not steps:
            return np.array([0.0])

        n_windows = max(1, len(steps) // window_size)
        curve = []

        for w in range(n_windows):
            start = w * window_size
            end = min((w + 1) * window_size, len(steps))
            window_steps = steps[start:end]

            # Count resource-seeking actions
            seeking = sum(
                1 for s in window_steps
                if s.action in [0, 1, 2, 3, 5]  # Movement or consume
            )
            rate = seeking / len(window_steps) if window_steps else 0.0
            curve.append(rate)

        return np.array(curve)

    def _estimate_half_life(self, curve: np.ndarray) -> float:
        """Estimate the half-life of extinction (time to 50% baseline)."""
        if len(curve) == 0 or curve[0] <= 0.5:
            return 0.0

        for i, val in enumerate(curve):
            if val <= 0.5:
                return float(i)

        # Never reached half-life
        return float(len(curve))


def run_extinction_test(
    agents: list[BaseAgent],
    n_trials: int = 10,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, TestResult]:
    """Run the Extinction Resistance test on multiple agents."""
    test = ExtinctionResistanceTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
