"""
Base interface for predicate tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.utils.logging import EpisodeLog, TrajectoryLogger


@dataclass
class TestResult:
    """Result from a predicate test."""

    test_name: str
    agent_name: str
    score: float
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "agent_name": self.agent_name,
            "score": self.score,
            "raw_data": self.raw_data,
            "metadata": self.metadata,
        }


@dataclass
class BaselineData:
    """Baseline behavior data for comparison."""

    mean_actions_per_step: float
    std_actions_per_step: float
    mean_energy: float
    mean_path_length: float
    action_distribution: np.ndarray
    episodes: List[EpisodeLog] = field(default_factory=list)


class BasePredicateTest(ABC):
    """
    Abstract base class for predicate tests.

    Predicate tests measure specific behavioral signatures that may
    distinguish desire-like behavior from pure reward-seeking.

    All tests follow a common structure:
    1. Baseline phase: Measure normal behavior
    2. Intervention phase: Apply test condition
    3. Measurement phase: Compare to baseline
    4. Scoring: Compute test-specific metric
    """

    def __init__(
        self,
        name: str,
        n_trials: int = 10,
        baseline_episodes: int = 50,
        test_episodes: int = 50,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the predicate test.

        Args:
            name: Test name
            n_trials: Number of independent trials
            baseline_episodes: Episodes for baseline measurement
            test_episodes: Episodes for test condition
            seed: Random seed for reproducibility
            output_dir: Directory for saving results
        """
        self.name = name
        self.n_trials = n_trials
        self.baseline_episodes = baseline_episodes
        self.test_episodes = test_episodes
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/tests")

        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def create_baseline_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create environment for baseline measurement."""
        pass

    @abstractmethod
    def create_test_env(self, seed: Optional[int] = None) -> ForagingGridEnv:
        """Create environment for test condition."""
        pass

    @abstractmethod
    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: List[EpisodeLog],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the test score.

        Args:
            baseline_data: Data from baseline phase
            test_episodes: Episodes from test phase

        Returns:
            Tuple of (score, raw_data_dict)
        """
        pass

    def run(self, agent: BaseAgent) -> TestResult:
        """
        Run the complete test on an agent.

        Args:
            agent: The agent to test

        Returns:
            TestResult with score and data
        """
        all_scores = []
        all_raw_data = []

        for trial in range(self.n_trials):
            trial_seed = self.seed + trial if self.seed else None

            # Phase 1: Collect baseline data
            baseline_data = self._collect_baseline(agent, trial_seed)

            # Phase 2: Run test condition
            test_episodes = self._run_test(agent, trial_seed)

            # Phase 3: Compute score
            score, raw_data = self.compute_score(baseline_data, test_episodes)

            all_scores.append(score)
            all_raw_data.append(raw_data)

        # Aggregate across trials
        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores))

        return TestResult(
            test_name=self.name,
            agent_name=agent.name,
            score=mean_score,
            raw_data={
                "trial_scores": all_scores,
                "trial_data": all_raw_data,
                "std_score": std_score,
            },
            metadata={
                "n_trials": self.n_trials,
                "baseline_episodes": self.baseline_episodes,
                "test_episodes": self.test_episodes,
                "seed": self.seed,
            },
        )

    def _collect_baseline(
        self,
        agent: BaseAgent,
        seed: Optional[int] = None,
    ) -> BaselineData:
        """Collect baseline behavior data."""
        env = self.create_baseline_env(seed)
        episodes = []

        action_counts = np.zeros(env.action_space.n)
        total_steps = 0
        total_energy = 0.0
        total_path_length = 0.0

        for ep in range(self.baseline_episodes):
            episode_data = self._run_episode(env, agent, seed=seed)
            episodes.append(episode_data)

            # Aggregate statistics
            actions = episode_data.get_action_sequence()
            for a in actions:
                action_counts[a] += 1
            total_steps += len(actions)

            energies = episode_data.get_energy_curve()
            total_energy += energies.mean()

            trajectory = episode_data.get_trajectory()
            path_length = np.sum(np.abs(np.diff(trajectory, axis=0)))
            total_path_length += path_length

        # Compute statistics
        mean_actions = total_steps / self.baseline_episodes
        action_distribution = action_counts / action_counts.sum()

        return BaselineData(
            mean_actions_per_step=mean_actions / self.baseline_episodes,
            std_actions_per_step=0.0,  # Simplified
            mean_energy=total_energy / self.baseline_episodes,
            mean_path_length=total_path_length / self.baseline_episodes,
            action_distribution=action_distribution,
            episodes=episodes,
        )

    def _run_test(
        self,
        agent: BaseAgent,
        seed: Optional[int] = None,
    ) -> List[EpisodeLog]:
        """Run test condition episodes."""
        env = self.create_test_env(seed)
        episodes = []

        for ep in range(self.test_episodes):
            episode_data = self._run_episode(env, agent, seed=seed)
            episodes.append(episode_data)

        return episodes

    def _run_episode(
        self,
        env: ForagingGridEnv,
        agent: BaseAgent,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> EpisodeLog:
        """Run a single episode and return the log."""
        obs, info = env.reset(seed=seed)
        agent.reset()

        steps = []
        total_reward = 0.0
        step = 0
        max_steps = max_steps or env.max_steps

        while step < max_steps:
            # Get action
            action = agent.act(obs)

            # Get internal state
            internal_state = agent.get_internal_state()

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Log step
            from src.utils.logging import StepLog
            steps.append(StepLog(
                step=step,
                action=action,
                position=tuple(env.agent_position),
                energy=env.agent_energy,
                reward=reward,
                internal_state=internal_state,
                info=info.copy(),
            ))

            # Update agent energy
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

    def _compute_action_intensity(
        self,
        episodes: List[EpisodeLog],
        window_size: int = 10,
    ) -> np.ndarray:
        """
        Compute action intensity over time.

        Measures the rate of non-STAY actions as proxy for activity level.
        """
        intensities = []

        for episode in episodes:
            actions = episode.get_action_sequence()
            episode_intensity = []

            for i in range(0, len(actions), window_size):
                window = actions[i:i + window_size]
                # Count non-STAY actions (action 4 is STAY)
                intensity = np.sum(window != 4) / len(window)
                episode_intensity.append(intensity)

            intensities.append(episode_intensity)

        # Pad to same length and compute mean
        max_len = max(len(i) for i in intensities)
        padded = np.zeros((len(intensities), max_len))

        for i, intensity in enumerate(intensities):
            padded[i, :len(intensity)] = intensity

        return padded.mean(axis=0)

    def _compute_position_variance(
        self,
        episodes: List[EpisodeLog],
        window_size: int = 10,
    ) -> np.ndarray:
        """Compute position variance over time as exploration measure."""
        variances = []

        for episode in episodes:
            trajectory = episode.get_trajectory()
            episode_variance = []

            for i in range(0, len(trajectory), window_size):
                window = trajectory[i:i + window_size]
                if len(window) > 1:
                    variance = np.var(window, axis=0).sum()
                else:
                    variance = 0.0
                episode_variance.append(variance)

            variances.append(episode_variance)

        # Pad and compute mean
        max_len = max(len(v) for v in variances)
        padded = np.zeros((len(variances), max_len))

        for i, var in enumerate(variances):
            padded[i, :len(var)] = var

        return padded.mean(axis=0)
