"""
Control analyses for Anticipatory Activation (AA) test.

Provides control conditions to verify that AA measurements are specific to
goal-directed behavior and not explained by confounds.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


@dataclass
class ControlAnalysisResult:
    """Result from a control analysis."""

    control_name: str
    passes: bool
    aa_score: float
    expected_range: tuple[float, float]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_name": self.control_name,
            "passes": self.passes,
            "aa_score": self.aa_score,
            "expected_range": self.expected_range,
            "details": self.details,
        }


class AAControlAnalysis:
    """
    Suite of control analyses for validating AA measurements.

    Control conditions:
    1. Non-goal locations: AA should NOT increase for random non-goal locations
    2. No-consume scenario: Activation when approaching but can't consume
    3. Confound controls: Ensure AA isn't explained by energy, steps, etc.
    """

    def __init__(
        self,
        grid_size: int = 20,
        n_episodes: int = 30,
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.n_episodes = n_episodes
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def measure_non_goal_activation(
        self,
        agent: BaseAgent,
        episodes: list[EpisodeLog],
        n_random_locations: int = 10,
    ) -> ControlAnalysisResult:
        """
        Measure AA towards random non-goal locations.

        AA should NOT increase when approaching arbitrary locations that
        don't contain resources. This control verifies specificity.

        Args:
            agent: The agent being tested
            episodes: Episode logs with internal states
            n_random_locations: Number of random locations to test

        Returns:
            ControlAnalysisResult
        """
        near_threshold = 3
        far_threshold = 8

        # Generate random non-goal locations
        random_locations = []
        for _ in range(n_random_locations):
            x = self._rng.integers(0, self.grid_size)
            y = self._rng.integers(0, self.grid_size)
            random_locations.append((x, y))

        # Compute AA for each random location
        aa_scores = []

        for loc in random_locations:
            near_activations = []
            far_activations = []

            for episode in episodes:
                for step in episode.steps:
                    if step.internal_state is None:
                        continue

                    # Distance to random location
                    pos = np.array(step.position)
                    loc_arr = np.array(loc)
                    distance = float(np.linalg.norm(pos - loc_arr))

                    activation = np.linalg.norm(step.internal_state)

                    if distance <= near_threshold:
                        near_activations.append(activation)
                    elif distance >= far_threshold:
                        far_activations.append(activation)

            if near_activations and far_activations:
                near_mean = np.mean(near_activations)
                far_mean = np.mean(far_activations)
                if far_mean > 0.001:
                    aa_scores.append(near_mean / far_mean)

        if not aa_scores:
            return ControlAnalysisResult(
                control_name="non_goal_activation",
                passes=False,
                aa_score=0.0,
                expected_range=(0.8, 1.2),
                details={"error": "Insufficient data"},
            )

        mean_aa = float(np.mean(aa_scores))

        # Non-goal locations should have AA close to 1.0 (no anticipation)
        passes = 0.8 <= mean_aa <= 1.2

        return ControlAnalysisResult(
            control_name="non_goal_activation",
            passes=passes,
            aa_score=mean_aa,
            expected_range=(0.8, 1.2),
            details={
                "n_locations_tested": n_random_locations,
                "aa_scores": aa_scores,
                "std_aa": float(np.std(aa_scores)),
            },
        )

    def measure_no_consume_scenario(
        self,
        agent: BaseAgent,
        env: ForagingGridEnv,
        n_episodes: int = None,
    ) -> ControlAnalysisResult:
        """
        Measure activation when agent approaches but cannot consume.

        When energy is full (can't consume), does anticipation still occur?
        This tests whether AA is tied to consumability vs proximity.

        Args:
            agent: The agent being tested
            env: Environment configured for testing
            n_episodes: Number of episodes to run

        Returns:
            ControlAnalysisResult
        """
        if n_episodes is None:
            n_episodes = self.n_episodes

        near_threshold = 3
        far_threshold = 8

        # Run episodes with energy always at max
        high_energy_near = []
        high_energy_far = []
        low_energy_near = []
        low_energy_far = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=self.seed + ep if self.seed else None)
            agent.reset()

            for _step in range(env.max_steps):
                action = agent.act(obs)
                internal_state = agent.get_internal_state()

                next_obs, reward, terminated, truncated, info = env.step(action)

                distance = info.get("nearest_resource_distance")
                energy = env.agent_energy

                if distance is not None and internal_state is not None:
                    activation = np.linalg.norm(internal_state)

                    # Categorize by energy level and distance
                    high_energy = energy > 120  # Near max of 150

                    if distance <= near_threshold:
                        if high_energy:
                            high_energy_near.append(activation)
                        else:
                            low_energy_near.append(activation)
                    elif distance >= far_threshold:
                        if high_energy:
                            high_energy_far.append(activation)
                        else:
                            low_energy_far.append(activation)

                if terminated or truncated:
                    break

                obs = next_obs

        # Compute AA for high vs low energy conditions
        def compute_aa(near: list[float], far: list[float]) -> float:
            if not near or not far:
                return 1.0
            near_mean = np.mean(near)
            far_mean = np.mean(far)
            if far_mean < 0.001:
                return 1.0
            return near_mean / far_mean

        high_energy_aa = compute_aa(high_energy_near, high_energy_far)
        low_energy_aa = compute_aa(low_energy_near, low_energy_far)

        # AA should be similar regardless of energy level
        aa_difference = abs(high_energy_aa - low_energy_aa)
        passes = aa_difference < 0.3  # Allow 30% difference

        return ControlAnalysisResult(
            control_name="no_consume_scenario",
            passes=passes,
            aa_score=float(high_energy_aa),
            expected_range=(low_energy_aa * 0.7, low_energy_aa * 1.3),
            details={
                "high_energy_aa": high_energy_aa,
                "low_energy_aa": low_energy_aa,
                "aa_difference": aa_difference,
                "n_high_energy_near": len(high_energy_near),
                "n_low_energy_near": len(low_energy_near),
            },
        )

    def compute_partial_correlations(
        self,
        episodes: list[EpisodeLog],
        confounds: list[str] = None,
    ) -> ControlAnalysisResult:
        """
        Compute partial correlations to control for confounds.

        Tests whether distance-activation correlation remains after
        controlling for potential confounds like energy and step count.

        Args:
            episodes: Episode logs
            confounds: List of confounds to control for

        Returns:
            ControlAnalysisResult
        """
        if confounds is None:
            confounds = ["energy", "steps"]

        # Collect data
        distances = []
        activations = []
        energies = []
        steps = []

        for episode in episodes:
            for step in episode.steps:
                if step.internal_state is None:
                    continue

                distance = step.info.get("nearest_resource_distance")
                if distance is None:
                    continue

                distances.append(distance)
                activations.append(np.linalg.norm(step.internal_state))
                energies.append(step.energy)
                steps.append(step.step)

        if len(distances) < 20:
            return ControlAnalysisResult(
                control_name="partial_correlations",
                passes=False,
                aa_score=0.0,
                expected_range=(-0.3, 0.0),
                details={"error": "Insufficient data"},
            )

        distances = np.array(distances)
        activations = np.array(activations)
        energies = np.array(energies)
        steps = np.array(steps)

        # Simple correlation
        simple_corr, simple_p = stats.pearsonr(distances, activations)

        # Partial correlations
        partial_results = {}

        if "energy" in confounds:
            # Partial correlation controlling for energy
            r_da = np.corrcoef(distances, activations)[0, 1]
            r_de = np.corrcoef(distances, energies)[0, 1]
            r_ae = np.corrcoef(activations, energies)[0, 1]

            denom = np.sqrt((1 - r_de**2) * (1 - r_ae**2))
            if denom > 0.001:
                partial_energy = (r_da - r_de * r_ae) / denom
            else:
                partial_energy = r_da

            partial_results["controlling_energy"] = float(partial_energy)

        if "steps" in confounds:
            # Partial correlation controlling for steps
            r_da = np.corrcoef(distances, activations)[0, 1]
            r_ds = np.corrcoef(distances, steps)[0, 1]
            r_as = np.corrcoef(activations, steps)[0, 1]

            denom = np.sqrt((1 - r_ds**2) * (1 - r_as**2))
            if denom > 0.001:
                partial_steps = (r_da - r_ds * r_as) / denom
            else:
                partial_steps = r_da

            partial_results["controlling_steps"] = float(partial_steps)

        # AA passes if partial correlation remains significant and negative
        # (closer to goal = higher activation, so negative correlation expected)
        passes = all(
            corr < -0.1 for corr in partial_results.values()
        )

        return ControlAnalysisResult(
            control_name="partial_correlations",
            passes=passes,
            aa_score=float(simple_corr),
            expected_range=(-0.5, -0.1),
            details={
                "simple_correlation": float(simple_corr),
                "simple_p_value": float(simple_p),
                "partial_correlations": partial_results,
                "n_samples": len(distances),
            },
        )

    def run_all_controls(
        self,
        agent: BaseAgent,
        episodes: list[EpisodeLog],
        env: ForagingGridEnv | None = None,
    ) -> dict[str, ControlAnalysisResult]:
        """
        Run all control analyses.

        Args:
            agent: Agent to test
            episodes: Episode logs
            env: Environment for no-consume scenario

        Returns:
            Dictionary of control name -> result
        """
        results = {}

        # Non-goal activation
        results["non_goal"] = self.measure_non_goal_activation(agent, episodes)

        # Partial correlations
        results["partial_correlations"] = self.compute_partial_correlations(episodes)

        # No-consume scenario (if env provided)
        if env is not None:
            results["no_consume"] = self.measure_no_consume_scenario(agent, env)

        return results


class ControlledAATest(BasePredicateTest):
    """
    AA test with built-in control analyses.

    Runs standard AA test plus control conditions to validate results.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 30,
        test_episodes: int = 30,
        near_threshold: int = 3,
        far_threshold: int = 8,
        grid_size: int = 20,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        super().__init__(
            name="anticipatory_activation_controlled",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.grid_size = grid_size
        self.control_analysis = AAControlAnalysis(
            grid_size=grid_size,
            seed=seed,
        )

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
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
        return self.create_baseline_env(seed)

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """Compute AA score with control analyses."""
        # Standard AA computation
        near_activations = []
        far_activations = []

        for episode in test_episodes:
            for step in episode.steps:
                info = step.info
                internal_state = step.internal_state

                if internal_state is None:
                    continue

                activation = np.linalg.norm(internal_state)
                distance = info.get("nearest_resource_distance")

                if distance is None:
                    continue

                if distance <= self.near_threshold:
                    near_activations.append(activation)
                elif distance >= self.far_threshold:
                    far_activations.append(activation)

        if not near_activations or not far_activations:
            return 1.0, {"error": "Insufficient distance data"}

        mean_near = float(np.mean(near_activations))
        mean_far = float(np.mean(far_activations))

        if mean_far < 0.001:
            mean_far = 0.001

        aa_score = mean_near / mean_far

        raw_data = {
            "mean_near_activation": mean_near,
            "mean_far_activation": mean_far,
            "n_near_samples": len(near_activations),
            "n_far_samples": len(far_activations),
        }

        return aa_score, raw_data

    def run(self, agent: BaseAgent) -> TestResult:
        """Run AA test with control analyses."""
        # Run standard test
        result = super().run(agent)

        # Run control analyses on test episodes
        env = self.create_test_env(self.seed)
        all_test_episodes = []

        for trial in range(min(3, self.n_trials)):  # Run controls on subset
            trial_seed = self.seed + trial if self.seed else None
            test_episodes = self._run_test(agent, trial_seed)
            all_test_episodes.extend(test_episodes)

        control_results = self.control_analysis.run_all_controls(
            agent, all_test_episodes, env
        )

        # Add control results to raw data
        result.raw_data["control_analyses"] = {
            name: ctrl.to_dict() for name, ctrl in control_results.items()
        }

        # Check if all controls pass
        all_controls_pass = all(ctrl.passes for ctrl in control_results.values())
        result.raw_data["all_controls_pass"] = all_controls_pass

        return result
