"""
Failure analysis for identifying and understanding agent failures.

Provides tools for systematic analysis of failure modes,
error patterns, and edge cases.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import EpisodeLog


@dataclass
class FailureMode:
    """Identified failure mode."""

    name: str
    description: str
    frequency: float
    example_episodes: list[int]
    characteristics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "frequency": self.frequency,
            "n_examples": len(self.example_episodes),
            "characteristics": self.characteristics,
        }


@dataclass
class FailureAnalysisResult:
    """Result from failure analysis."""

    total_episodes: int
    failure_episodes: int
    failure_rate: float
    identified_modes: list[FailureMode]
    common_patterns: dict[str, int]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "failure_episodes": self.failure_episodes,
            "failure_rate": self.failure_rate,
            "identified_modes": [m.to_dict() for m in self.identified_modes],
            "common_patterns": self.common_patterns,
            "recommendations": self.recommendations,
        }


class FailureModeAnalyzer:
    """
    Analyzer for identifying and categorizing failure modes.

    Failure types:
    - Energy depletion: Agent runs out of energy
    - Stuck behavior: Agent gets stuck in loops
    - Suboptimal paths: Agent takes inefficient routes
    - Missed opportunities: Agent ignores visible resources
    """

    def __init__(
        self,
        success_threshold: float = 0.5,
        stuck_threshold: int = 20,
    ):
        """
        Initialize failure analyzer.

        Args:
            success_threshold: Minimum survival rate for success
            stuck_threshold: Positions to check for stuck detection
        """
        self.success_threshold = success_threshold
        self.stuck_threshold = stuck_threshold

    def analyze_episodes(
        self,
        episodes: list[EpisodeLog],
    ) -> FailureAnalysisResult:
        """
        Analyze episodes for failure modes.

        Args:
            episodes: Episodes to analyze

        Returns:
            FailureAnalysisResult
        """
        failure_episodes = []
        failure_types = []

        for i, episode in enumerate(episodes):
            failures = self._classify_episode_failure(episode)
            if failures:
                failure_episodes.append(i)
                failure_types.extend(failures)

        # Count failure types
        type_counts = Counter(failure_types)
        n_failures = len(failure_episodes)
        total = len(episodes)

        # Identify distinct failure modes
        modes = self._identify_failure_modes(episodes, failure_episodes, type_counts)

        # Common patterns
        patterns = self._extract_common_patterns(episodes, failure_episodes)

        # Generate recommendations
        recommendations = self._generate_recommendations(modes, type_counts)

        return FailureAnalysisResult(
            total_episodes=total,
            failure_episodes=n_failures,
            failure_rate=n_failures / total if total > 0 else 0.0,
            identified_modes=modes,
            common_patterns=patterns,
            recommendations=recommendations,
        )

    def _classify_episode_failure(
        self,
        episode: EpisodeLog,
    ) -> list[str]:
        """Classify failures in a single episode."""
        failures = []

        # Energy depletion
        if episode.final_energy <= 0:
            failures.append("energy_depletion")

        # Check for stuck behavior
        if self._is_stuck(episode):
            failures.append("stuck_behavior")

        # Check for missed resources
        if self._missed_resources(episode):
            failures.append("missed_resources")

        # Low reward efficiency
        if episode.total_reward < 0:
            failures.append("negative_reward")

        return failures

    def _is_stuck(self, episode: EpisodeLog) -> bool:
        """Check if agent got stuck in a loop."""
        trajectory = episode.get_trajectory()

        if len(trajectory) < self.stuck_threshold:
            return False

        # Check last N positions for repetition
        recent = trajectory[-self.stuck_threshold:]
        recent_tuples = [tuple(p) for p in recent]
        unique = set(recent_tuples)

        # Stuck if visiting very few unique positions
        return len(unique) < self.stuck_threshold // 4

    def _missed_resources(self, episode: EpisodeLog) -> bool:
        """Check if agent missed visible resources."""
        for step in episode.steps:
            # Check if resource was visible but agent moved away
            if step.info.get("on_resource") and step.action != 5:  # Not consume
                return True
        return False

    def _identify_failure_modes(
        self,
        episodes: list[EpisodeLog],
        failure_indices: list[int],
        type_counts: Counter,
    ) -> list[FailureMode]:
        """Identify distinct failure modes."""
        modes = []

        for failure_type, count in type_counts.most_common():
            # Find examples
            examples = []
            for i in failure_indices:
                failures = self._classify_episode_failure(episodes[i])
                if failure_type in failures:
                    examples.append(i)
                    if len(examples) >= 5:
                        break

            # Analyze characteristics
            characteristics = self._analyze_failure_characteristics(
                [episodes[i] for i in examples], failure_type
            )

            # Create mode description
            descriptions = {
                "energy_depletion": "Agent runs out of energy before finding resources",
                "stuck_behavior": "Agent gets stuck in repetitive behavior patterns",
                "missed_resources": "Agent fails to consume visible resources",
                "negative_reward": "Agent accumulates negative total reward",
            }

            modes.append(FailureMode(
                name=failure_type,
                description=descriptions.get(failure_type, "Unknown failure mode"),
                frequency=count / len(failure_indices) if failure_indices else 0,
                example_episodes=examples,
                characteristics=characteristics,
            ))

        return modes

    def _analyze_failure_characteristics(
        self,
        episodes: list[EpisodeLog],
        failure_type: str,
    ) -> dict[str, Any]:
        """Analyze characteristics of a failure type."""
        if not episodes:
            return {}

        # Collect statistics
        final_energies = [e.final_energy for e in episodes]
        step_counts = [e.total_steps for e in episodes]
        rewards = [e.total_reward for e in episodes]

        characteristics = {
            "mean_final_energy": float(np.mean(final_energies)),
            "mean_steps": float(np.mean(step_counts)),
            "mean_reward": float(np.mean(rewards)),
        }

        # Type-specific analysis
        if failure_type == "energy_depletion":
            # When did energy run out?
            depletion_steps = []
            for episode in episodes:
                energies = episode.get_energy_curve()
                if len(energies) > 0:
                    zero_idx = np.where(energies <= 0)[0]
                    if len(zero_idx) > 0:
                        depletion_steps.append(zero_idx[0])

            if depletion_steps:
                characteristics["mean_depletion_step"] = float(np.mean(depletion_steps))

        elif failure_type == "stuck_behavior":
            # What positions did they get stuck at?
            stuck_positions = []
            for episode in episodes:
                trajectory = episode.get_trajectory()
                if len(trajectory) >= self.stuck_threshold:
                    recent = trajectory[-self.stuck_threshold:]
                    most_common = Counter([tuple(p) for p in recent]).most_common(1)
                    if most_common:
                        stuck_positions.append(most_common[0][0])

            characteristics["common_stuck_positions"] = stuck_positions[:5]

        return characteristics

    def _extract_common_patterns(
        self,
        episodes: list[EpisodeLog],
        failure_indices: list[int],
    ) -> dict[str, int]:
        """Extract common patterns in failure episodes."""
        patterns = Counter()

        for i in failure_indices:
            episode = episodes[i]

            # Analyze action sequences
            actions = episode.get_action_sequence()
            if len(actions) > 10:
                # Check for action repetition
                unique_ratio = len(set(actions)) / len(actions)
                if unique_ratio < 0.3:
                    patterns["repetitive_actions"] += 1

                # Check for excessive staying
                stay_ratio = np.mean(actions == 4)
                if stay_ratio > 0.5:
                    patterns["excessive_staying"] += 1

                # Check for no consumption
                consume_count = np.sum(actions == 5)
                if consume_count == 0:
                    patterns["no_consumption"] += 1

            # Check energy trajectory
            energies = episode.get_energy_curve()
            if len(energies) > 10:
                # Monotonic decrease
                if np.all(np.diff(energies) <= 0):
                    patterns["monotonic_energy_decrease"] += 1

        return dict(patterns)

    def _generate_recommendations(
        self,
        modes: list[FailureMode],
        type_counts: Counter,
    ) -> list[str]:
        """Generate recommendations based on failure analysis."""
        recommendations = []

        most_common = type_counts.most_common(1)
        if most_common:
            main_failure = most_common[0][0]

            if main_failure == "energy_depletion":
                recommendations.extend([
                    "Consider reducing energy decay rate",
                    "Increase resource density or energy values",
                    "Improve exploration strategy to find resources faster",
                ])
            elif main_failure == "stuck_behavior":
                recommendations.extend([
                    "Add exploration bonus or entropy regularization",
                    "Check for environment obstacles blocking paths",
                    "Increase action diversity in training",
                ])
            elif main_failure == "missed_resources":
                recommendations.extend([
                    "Verify consume action is properly rewarded",
                    "Check agent's resource detection in observations",
                    "Ensure agent has learned resource value",
                ])
            elif main_failure == "negative_reward":
                recommendations.extend([
                    "Review reward shaping for unintended penalties",
                    "Check for reward scale issues",
                    "Consider curriculum learning",
                ])

        return recommendations


def identify_edge_cases(
    episodes: list[EpisodeLog],
    percentile: float = 5.0,
) -> dict[str, list[int]]:
    """
    Identify edge case episodes.

    Args:
        episodes: Episodes to analyze
        percentile: Percentile for extreme cases

    Returns:
        Dictionary of edge case type -> episode indices
    """
    rewards = np.array([e.total_reward for e in episodes])
    steps = np.array([e.total_steps for e in episodes])
    energies = np.array([e.final_energy for e in episodes])

    edge_cases = {
        "very_low_reward": np.where(rewards < np.percentile(rewards, percentile))[0].tolist(),
        "very_high_reward": np.where(rewards > np.percentile(rewards, 100 - percentile))[0].tolist(),
        "very_short": np.where(steps < np.percentile(steps, percentile))[0].tolist(),
        "very_long": np.where(steps > np.percentile(steps, 100 - percentile))[0].tolist(),
        "zero_energy": np.where(energies <= 0)[0].tolist(),
        "max_energy": np.where(energies >= np.percentile(energies, 100 - percentile))[0].tolist(),
    }

    return edge_cases


def run_failure_analysis(
    episodes: list[EpisodeLog],
    agent_name: str = "unknown",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive failure analysis.

    Args:
        episodes: Episodes to analyze
        agent_name: Name of the agent
        output_dir: Directory to save results

    Returns:
        Analysis results
    """
    analyzer = FailureModeAnalyzer()
    result = analyzer.analyze_episodes(episodes)

    edge_cases = identify_edge_cases(episodes)

    output = {
        "agent_name": agent_name,
        "failure_analysis": result.to_dict(),
        "edge_cases": edge_cases,
        "summary": {
            "total_episodes": result.total_episodes,
            "failure_rate": result.failure_rate,
            "primary_failure_mode": result.identified_modes[0].name if result.identified_modes else None,
            "n_recommendations": len(result.recommendations),
        },
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_dir / f"failure_analysis_{agent_name}.json", "w") as f:
            json.dump(output, f, indent=2)

    return output
