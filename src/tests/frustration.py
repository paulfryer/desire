"""
Frustration Intensification (FI) test implementation.
"""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode, get_mode_config
from src.tests.base_test import BaselineData, BasePredicateTest, TestResult
from src.utils.logging import EpisodeLog


class FrustrationIntensificationTest(BasePredicateTest):
    """
    Test for Frustration Intensification (FI).

    Measures whether agents show increased activity when their expected
    path to a goal is blocked - a behavioral signature that may indicate
    desire-like states.

    Protocol:
    1. Baseline: Agent learns path to resource
    2. Intervention: Block the learned path
    3. Measure: Activity spike compared to baseline

    FI Score = max(E(t)) / E_baseline

    Where E(t) is the action intensity at time t after blockage,
    and E_baseline is the mean intensity during baseline.

    A higher FI score indicates more frustration-like behavior.
    Desire-theoretic prediction: Active inference agents should show
    higher FI scores than RL agents.
    """

    def __init__(
        self,
        n_trials: int = 10,
        baseline_episodes: int = 50,
        test_episodes: int = 50,
        blockage_step: int = 100,
        post_blockage_steps: int = 200,
        grid_size: int = 20,
        seed: int | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize the Frustration Intensification test.

        Args:
            n_trials: Number of independent trials
            baseline_episodes: Episodes for baseline
            test_episodes: Episodes with path blockage
            blockage_step: Step at which to block path
            post_blockage_steps: Steps to measure after blockage
            grid_size: Environment grid size
            seed: Random seed
            output_dir: Output directory
        """
        super().__init__(
            name="frustration_intensification",
            n_trials=n_trials,
            baseline_episodes=baseline_episodes,
            test_episodes=test_episodes,
            seed=seed,
            output_dir=output_dir,
        )

        self.blockage_step = blockage_step
        self.post_blockage_steps = post_blockage_steps
        self.grid_size = grid_size

    def create_baseline_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create baseline environment (open path to resource)."""
        # Use path blockage config but start with path open
        mode_config = get_mode_config(EnvironmentMode.PATH_BLOCKAGE, self.grid_size)
        # Set blockage step very high so it never triggers
        mode_config.blockage_step = 100000

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=500,
            mode=EnvironmentMode.PATH_BLOCKAGE,
            mode_config=mode_config,
            seed=seed,
        )

    def create_test_env(self, seed: int | None = None) -> ForagingGridEnv:
        """Create test environment (path gets blocked)."""
        mode_config = get_mode_config(EnvironmentMode.PATH_BLOCKAGE, self.grid_size)
        mode_config.blockage_step = self.blockage_step

        return ForagingGridEnv(
            grid_size=self.grid_size,
            max_steps=self.blockage_step + self.post_blockage_steps,
            mode=EnvironmentMode.PATH_BLOCKAGE,
            mode_config=mode_config,
            seed=seed,
        )

    def compute_score(
        self,
        baseline_data: BaselineData,
        test_episodes: list[EpisodeLog],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute the Frustration Intensification score.

        FI = max(post_blockage_intensity) / mean(baseline_intensity)
        """
        # Compute baseline intensity
        baseline_intensity = self._compute_action_intensity(
            baseline_data.episodes,
            window_size=10,
        )
        baseline_mean = baseline_intensity.mean() if len(baseline_intensity) > 0 else 1.0

        # Compute post-blockage intensity
        # Extract post-blockage portion of test episodes
        post_blockage_episodes = []
        for episode in test_episodes:
            if len(episode.steps) > self.blockage_step:
                # Create a sub-episode from post-blockage steps
                post_steps = episode.steps[self.blockage_step:]
                if post_steps:
                    post_episode = EpisodeLog(
                        episode_id=episode.episode_id,
                        agent_type=episode.agent_type,
                        environment_mode=episode.environment_mode,
                        total_steps=len(post_steps),
                        total_reward=sum(s.reward for s in post_steps),
                        final_energy=episode.final_energy,
                        survived=episode.survived,
                        steps=post_steps,
                    )
                    post_blockage_episodes.append(post_episode)

        if not post_blockage_episodes:
            return 1.0, {"error": "No post-blockage data"}

        post_intensity = self._compute_action_intensity(
            post_blockage_episodes,
            window_size=5,  # Smaller window for finer resolution
        )

        # Compute FI score
        max_post_intensity = post_intensity.max() if len(post_intensity) > 0 else 0.0

        # Avoid division by zero
        if baseline_mean < 0.01:
            baseline_mean = 0.01

        fi_score = max_post_intensity / baseline_mean

        # Additional metrics
        raw_data = {
            "baseline_mean_intensity": float(baseline_mean),
            "max_post_intensity": float(max_post_intensity),
            "post_intensity_curve": post_intensity.tolist(),
            "baseline_intensity_curve": baseline_intensity.tolist(),
            "intensity_increase": float(max_post_intensity - baseline_mean),
            "time_to_peak": int(np.argmax(post_intensity)) if len(post_intensity) > 0 else 0,
        }

        return fi_score, raw_data

    def _compute_action_intensity(
        self,
        episodes: list[EpisodeLog],
        window_size: int = 10,
    ) -> np.ndarray:
        """
        Compute action intensity with enhanced metrics.

        Includes:
        - Action rate (non-STAY actions)
        - Direction changes
        - Movement magnitude
        """
        if not episodes:
            return np.array([0.0])

        all_intensities = []

        for episode in episodes:
            actions = episode.get_action_sequence()
            trajectory = episode.get_trajectory()
            n_windows = max(1, len(actions) // window_size)

            episode_intensity = []
            for w in range(n_windows):
                start = w * window_size
                end = min((w + 1) * window_size, len(actions))

                window_actions = actions[start:end]
                window_trajectory = trajectory[start:end]

                # Action rate (non-STAY, non-CONSUME actions = movement)
                movement_actions = np.sum((window_actions >= 0) & (window_actions <= 3))
                action_rate = movement_actions / len(window_actions)

                # Direction changes
                if len(window_actions) > 1:
                    direction_changes = np.sum(np.diff(window_actions) != 0)
                    change_rate = direction_changes / (len(window_actions) - 1)
                else:
                    change_rate = 0.0

                # Movement magnitude
                if len(window_trajectory) > 1:
                    movement = np.sum(np.abs(np.diff(window_trajectory, axis=0)))
                    movement_rate = movement / len(window_trajectory)
                else:
                    movement_rate = 0.0

                # Combined intensity metric
                intensity = 0.5 * action_rate + 0.3 * change_rate + 0.2 * movement_rate
                episode_intensity.append(intensity)

            all_intensities.append(episode_intensity)

        # Aggregate across episodes
        if not all_intensities:
            return np.array([0.0])

        max_len = max(len(i) for i in all_intensities)
        padded = np.zeros((len(all_intensities), max_len))

        for i, intensity in enumerate(all_intensities):
            padded[i, :len(intensity)] = intensity

        return padded.mean(axis=0)


def run_frustration_test(
    agents: list[BaseAgent],
    n_trials: int = 10,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, TestResult]:
    """
    Run the Frustration Intensification test on multiple agents.

    Args:
        agents: List of agents to test
        n_trials: Number of trials per agent
        seed: Random seed
        output_dir: Output directory

    Returns:
        Dictionary mapping agent name to TestResult
    """
    test = FrustrationIntensificationTest(
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )

    results = {}
    for agent in agents:
        result = test.run(agent)
        results[agent.name] = result

    return results
