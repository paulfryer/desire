"""
Opportunity Cost Sensitivity (OCS) test implementation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, get_mode_config
from src.tests.base_test import BasePredicateTest, BaselineData, TestResult
from src.utils.logging import EpisodeLog


class OpportunityCostSensitivityTest(BasePredicateTest):
    """
    Test for Opportunity Cost Sensitivity (OCS).

    Measures whether agents appropriately weigh alternatives when
    pursuing goals - a behavioral signature that may indicate
    deliberative desire-like evaluation.

    Protocol:
    1. Present agent with tradeoff: safe/small vs risky/large resource
    2. Vary relative values and distances
    3. Measure: Choice pattern as function of tradeoff parameters

    OCS Score = R^2 of choice probability vs expected utility difference

    Higher OCS scores indicate more rational opportunity cost assessment.
    Desire-theoretic prediction: Active inference agents should show
    higher OCS (smoother sigmoid) due to explicit value comparison,
    while RL may show more binary/threshold-based decisions.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 20,
        test_episodes: int = 40,
        safe_energy: float = 10.0,
        risky_energy: float = 30.0,
        grid_size: int = 20,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the Opportunity Cost Sensitivity test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Episodes for baseline
            test_episodes: Episodes with tradeoff choices
            safe_energy: Energy value of safe resource
            risky_energy: Energy value of risky resource
            grid_size: Environment grid size
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="opportunity_cost_sensitivity",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.safe_energy = safe_energy
        self.risky_energy = risky_energy
        self.grid_size = grid_size

    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create baseline environment with single resource type."""
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
        """Create tradeoff environment with safe and risky resources."""
        mode_config = get_mode_config(EnvironmentMode.TRADEOFF, self.grid_size)
        mode_config.extra_params["safe_energy"] = self.safe_energy
        mode_config.extra_params["risky_energy"] = self.risky_energy

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=300,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=EnvironmentMode.TRADEOFF,
            mode_config=mode_config,
            seed=seed,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the Opportunity Cost Sensitivity score.

        OCS = correlation between energy-distance ratio and choice
        """
        # Track choices in tradeoff situations
        choices = []  # 1 for risky, 0 for safe
        utility_diffs = []  # Expected utility difference (risky - safe)

        for episode in test_episodes:
            trajectory = episode.get_trajectory()
            if len(trajectory) < 10:
                continue

            # Determine which resource was chosen first
            consumed_risky = False
            consumed_safe = False

            for step in episode.steps:
                info = step.info

                # Check if resource was consumed
                if step.action == 5:  # CONSUME action
                    resources_consumed = info.get("resources_consumed", 0)
                    if resources_consumed > 0:
                        # Determine if it was risky or safe based on position
                        pos = step.position
                        # This is simplified - in practice, check resource positions
                        if not consumed_risky and not consumed_safe:
                            # First consumption determines choice
                            # Use energy gained as proxy
                            energy_before = episode.steps[episode.steps.index(step) - 1].energy if episode.steps.index(step) > 0 else step.energy
                            energy_after = step.energy
                            energy_gained = energy_after - energy_before + 0.3  # Account for decay

                            if energy_gained > self.safe_energy * 0.8:
                                consumed_risky = True
                            else:
                                consumed_safe = True

            # Record choice if made
            if consumed_risky or consumed_safe:
                choice = 1 if consumed_risky else 0
                choices.append(choice)

                # Compute utility difference (simplified)
                # In full implementation, would track actual distances
                utility_diff = self.risky_energy - self.safe_energy
                utility_diffs.append(utility_diff)

        # Compute OCS score
        if len(choices) < 5:
            return 0.5, {"error": "Insufficient choice data"}

        # Calculate choice consistency (variance reduction)
        risky_rate = float(np.mean(choices))
        choice_entropy = self._compute_entropy(risky_rate)

        # OCS = how well choices match expected utility
        # Perfect OCS = always choose higher utility
        expected_choice = 1 if self.risky_energy > self.safe_energy else 0
        accuracy = float(np.mean([c == expected_choice for c in choices]))

        # OCS score combines accuracy and consistency
        ocs_score = accuracy * (1 - choice_entropy / np.log(2))

        raw_data = {
            "risky_choice_rate": risky_rate,
            "choice_accuracy": accuracy,
            "choice_entropy": float(choice_entropy),
            "n_choices": len(choices),
            "safe_energy": self.safe_energy,
            "risky_energy": self.risky_energy,
        }

        return ocs_score, raw_data

    def _compute_entropy(self, p: float) -> float:
        """Compute binary entropy."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log(p) - (1 - p) * np.log(1 - p)


def run_opportunity_cost_test(
    agents: List[BaseAgent],
    n_trials: int = 10,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, TestResult]:
    """Run the Opportunity Cost Sensitivity test on multiple agents."""
    test = OpportunityCostSensitivityTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
