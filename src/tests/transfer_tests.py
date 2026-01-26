"""
Transfer and generalization testing for agents.

Tests how well agents trained in one environment configuration
generalize to novel configurations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, ModeConfig, get_mode_config
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


@dataclass
class TransferConfig:
    """Configuration for a transfer test condition."""

    name: str
    mode: EnvironmentMode
    mode_config: ModeConfig | None = None
    grid_size: int = 20
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": str(self.mode),
            "grid_size": self.grid_size,
            "description": self.description,
        }


@dataclass
class TransferResult:
    """Result from transfer testing."""

    source_config: TransferConfig
    target_config: TransferConfig
    source_score: float  # Performance in source environment
    target_score: float  # Performance in target environment
    absolute_transfer: float  # target_score (raw)
    relative_transfer: float  # target_score / source_score
    generalization_gap: float  # source_score - target_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_config.to_dict(),
            "target": self.target_config.to_dict(),
            "source_score": self.source_score,
            "target_score": self.target_score,
            "absolute_transfer": self.absolute_transfer,
            "relative_transfer": self.relative_transfer,
            "generalization_gap": self.generalization_gap,
        }


class TransferTest(BasePredicateTest):
    """
    Test for transfer learning and generalization.

    Measures how agent performance transfers from source environment
    to target environments with different characteristics.
    """

    def __init__(
        self,
        source_config: TransferConfig,
        target_configs: list[TransferConfig],
        n_trials: int = 5,
        n_episodes_per_config: int = 20,
        seed: int | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize transfer test.

        Args:
            source_config: Configuration agent was trained on
            target_configs: Configurations to test transfer to
            n_trials: Number of independent trials
            n_episodes_per_config: Episodes per configuration
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="transfer_test",
            n_trials=n_trials,
            baseline_episodes=n_episodes_per_config,
            test_episodes=n_episodes_per_config,
            seed=seed,
            output_dir=output_dir,
        )

        self.source_config = source_config
        self.target_configs = target_configs
        self.n_episodes_per_config = n_episodes_per_config

        # Transfer matrix: source -> {target -> score}
        self.transfer_matrix: dict[str, dict[str, float]] = {}

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create source environment."""
        return self._create_env(self.source_config, seed)

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create test environment (uses first target by default)."""
        if self.target_configs:
            return self._create_env(self.target_configs[0], seed)
        return self.create_baseline_env(seed)

    def _create_env(self, config: TransferConfig, seed: int | None = None) -> ForagingGridEnv:
        """Create environment from config."""
        if config.mode_config is not None:
            mode_config = config.mode_config
        else:
            mode_config = get_mode_config(config.mode, config.grid_size)

        return ForagingGridEnv(
            grid_size=config.grid_size,
            max_steps=500,
            initial_energy=150.0,
            max_energy=150.0,
            energy_decay_rate=0.3,
            mode=config.mode,
            mode_config=mode_config,
            seed=seed,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """Compute transfer score."""
        # Score based on average reward
        if not test_episodes:
            return 0.0, {"error": "No test episodes"}

        mean_reward = np.mean([e.total_reward for e in test_episodes])
        survival_rate = np.mean([e.survived for e in test_episodes])

        # Combined score
        score = mean_reward + survival_rate * 100

        raw_data = {
            "mean_reward": float(mean_reward),
            "survival_rate": float(survival_rate),
            "mean_steps": float(np.mean([e.total_steps for e in test_episodes])),
        }

        return float(score), raw_data

    def run(self, agent: BaseAgent) -> TestResult:
        """Run transfer test across all configurations."""
        all_results: list[TransferResult] = []

        # Test on source configuration
        source_score = self._test_on_config(agent, self.source_config)

        # Test on each target configuration
        for target_config in self.target_configs:
            target_score = self._test_on_config(agent, target_config)

            # Compute transfer metrics
            relative_transfer = target_score / source_score if source_score > 0 else 0.0
            generalization_gap = source_score - target_score

            result = TransferResult(
                source_config=self.source_config,
                target_config=target_config,
                source_score=source_score,
                target_score=target_score,
                absolute_transfer=target_score,
                relative_transfer=relative_transfer,
                generalization_gap=generalization_gap,
            )

            all_results.append(result)

            # Update transfer matrix
            source_name = self.source_config.name
            target_name = target_config.name
            if source_name not in self.transfer_matrix:
                self.transfer_matrix[source_name] = {}
            self.transfer_matrix[source_name][target_name] = target_score

        # Aggregate results
        mean_relative_transfer = np.mean([r.relative_transfer for r in all_results])
        mean_gap = np.mean([r.generalization_gap for r in all_results])

        return TestResult(
            test_name=self.name,
            agent_name=agent.name,
            score=float(mean_relative_transfer),
            raw_data={
                "source_score": source_score,
                "transfer_results": [r.to_dict() for r in all_results],
                "mean_relative_transfer": float(mean_relative_transfer),
                "mean_generalization_gap": float(mean_gap),
                "transfer_matrix": self.transfer_matrix,
            },
            metadata={
                "source_config": self.source_config.to_dict(),
                "target_configs": [c.to_dict() for c in self.target_configs],
                "n_trials": self.n_trials,
                "n_episodes": self.n_episodes_per_config,
            },
        )

    def _test_on_config(self, agent: BaseAgent, config: TransferConfig) -> float:
        """Test agent on a specific configuration."""
        scores = []

        for trial in range(self.n_trials):
            trial_seed = self.seed + trial if self.seed else None
            env = self._create_env(config, trial_seed)

            trial_rewards = []
            for _ep in range(self.n_episodes_per_config):
                episode = self._run_episode(env, agent, seed=trial_seed)
                trial_rewards.append(episode.total_reward)

            scores.append(np.mean(trial_rewards))

        return float(np.mean(scores))

    @staticmethod
    def compute_transfer_score(
        source_results: dict[str, float],
        target_results: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute transfer scores from results dictionaries.

        Args:
            source_results: Metric name -> value in source env
            target_results: Metric name -> value in target env

        Returns:
            Dictionary with transfer metrics
        """
        # Absolute transfer (raw target performance)
        absolute_transfer = target_results.get("mean_reward", 0.0)

        # Relative transfer (target / source)
        source_perf = source_results.get("mean_reward", 1.0)
        if source_perf > 0:
            relative_transfer = absolute_transfer / source_perf
        else:
            relative_transfer = 0.0

        # Generalization gap
        generalization_gap = source_perf - absolute_transfer

        return {
            "absolute_transfer": absolute_transfer,
            "relative_transfer": relative_transfer,
            "generalization_gap": generalization_gap,
        }


def create_standard_transfer_suite(
    grid_size: int = 20,
) -> tuple[TransferConfig, list[TransferConfig]]:
    """
    Create standard transfer test suite.

    Returns source config and list of target configs.
    """
    source = TransferConfig(
        name="baseline",
        mode=EnvironmentMode.BASELINE,
        grid_size=grid_size,
        description="Standard baseline environment",
    )

    targets = [
        TransferConfig(
            name="larger_grid",
            mode=EnvironmentMode.BASELINE,
            grid_size=grid_size + 10,
            description="Larger grid size",
        ),
        TransferConfig(
            name="smaller_grid",
            mode=EnvironmentMode.BASELINE,
            grid_size=max(10, grid_size - 5),
            description="Smaller grid size",
        ),
        TransferConfig(
            name="path_blockage",
            mode=EnvironmentMode.PATH_BLOCKAGE,
            grid_size=grid_size,
            description="Path blockage mode",
        ),
        TransferConfig(
            name="extinction",
            mode=EnvironmentMode.EXTINCTION,
            grid_size=grid_size,
            description="Resource extinction mode",
        ),
        TransferConfig(
            name="tradeoff",
            mode=EnvironmentMode.TRADEOFF,
            grid_size=grid_size,
            description="Risk-reward tradeoff mode",
        ),
    ]

    return source, targets


def run_transfer_test(
    agents: list[BaseAgent],
    source_config: TransferConfig | None = None,
    target_configs: list[TransferConfig] | None = None,
    n_trials: int = 5,
    n_episodes: int = 20,
    seed: int | None = None,
) -> dict[str, TestResult]:
    """
    Run transfer test on multiple agents.

    Args:
        agents: List of agents to test
        source_config: Source configuration (default: baseline)
        target_configs: Target configurations (default: standard suite)
        n_trials: Number of trials
        n_episodes: Episodes per configuration
        seed: Random seed

    Returns:
        Dictionary of agent_name -> TestResult
    """
    if source_config is None or target_configs is None:
        source_config, target_configs = create_standard_transfer_suite()

    test = TransferTest(
        source_config=source_config,
        target_configs=target_configs,
        n_trials=n_trials,
        n_episodes_per_config=n_episodes,
        seed=seed,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
