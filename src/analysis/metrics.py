"""
Metrics computation for behavioral analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.tests.base_test import TestResult
from src.utils.logging import EpisodeLog


def compute_frustration_score(
    baseline_intensity: np.ndarray,
    post_blockage_intensity: np.ndarray,
) -> float:
    """
    Compute the Frustration Intensification (FI) score.

    FI = max(post_blockage_intensity) / mean(baseline_intensity)

    Args:
        baseline_intensity: Action intensity during baseline period
        post_blockage_intensity: Action intensity after path blockage

    Returns:
        FI score (>1 indicates increased activity)
    """
    baseline_mean = np.mean(baseline_intensity)
    if baseline_mean < 0.01:
        baseline_mean = 0.01

    max_post = np.max(post_blockage_intensity)
    return float(max_post / baseline_mean)


def compute_means_end_score(
    optimal_path_length: int,
    actual_reach_times: List[int],
) -> float:
    """
    Compute the Means-End Decoupling (MED) score.

    MED = optimal_path_length / mean(actual_reach_time)

    Args:
        optimal_path_length: Shortest possible path to goal
        actual_reach_times: List of actual times to reach goal

    Returns:
        MED score (1.0 = optimal, <1 = suboptimal)
    """
    if not actual_reach_times:
        return 0.0

    mean_time = np.mean(actual_reach_times)
    if mean_time <= 0:
        return 1.0

    score = optimal_path_length / mean_time
    return float(np.clip(score, 0.0, 1.0))


def compute_extinction_resistance(
    episodes: List[EpisodeLog],
    resource_depletion_step: int,
    window_size: int = 50,
) -> Dict[str, float]:
    """
    Compute extinction resistance metrics.

    Measures how long agent continues seeking behavior after
    resources are depleted.

    Args:
        episodes: Episode logs
        resource_depletion_step: Step when resources were depleted
        window_size: Window for computing seeking intensity

    Returns:
        Dictionary with extinction metrics
    """
    seeking_durations = []
    intensity_decay_rates = []

    for episode in episodes:
        actions = episode.get_action_sequence()

        if len(actions) <= resource_depletion_step:
            continue

        # Post-depletion actions
        post_actions = actions[resource_depletion_step:]

        # Seeking = movement actions (not STAY=4, not CONSUME=5)
        seeking = (post_actions >= 0) & (post_actions <= 3)

        # Find when seeking stops (consecutive STAYs)
        consecutive_stays = 0
        seeking_end = len(post_actions)

        for i, is_seeking in enumerate(seeking):
            if not is_seeking:
                consecutive_stays += 1
                if consecutive_stays >= 10:  # 10 consecutive non-seeking
                    seeking_end = i - 10
                    break
            else:
                consecutive_stays = 0

        seeking_durations.append(seeking_end)

        # Compute intensity decay
        if len(post_actions) >= 2 * window_size:
            early_intensity = np.mean(seeking[:window_size])
            late_intensity = np.mean(seeking[-window_size:])
            if early_intensity > 0:
                decay_rate = (early_intensity - late_intensity) / early_intensity
                intensity_decay_rates.append(decay_rate)

    return {
        "mean_seeking_duration": float(np.mean(seeking_durations)) if seeking_durations else 0.0,
        "std_seeking_duration": float(np.std(seeking_durations)) if seeking_durations else 0.0,
        "mean_decay_rate": float(np.mean(intensity_decay_rates)) if intensity_decay_rates else 0.0,
        "extinction_resistance": float(np.mean(seeking_durations)) if seeking_durations else 0.0,
    }


def compute_action_entropy(action_sequence: np.ndarray) -> float:
    """
    Compute entropy of action distribution.

    Higher entropy = more random/exploratory behavior.

    Args:
        action_sequence: Sequence of actions

    Returns:
        Entropy value
    """
    if len(action_sequence) == 0:
        return 0.0

    # Count action frequencies
    unique, counts = np.unique(action_sequence, return_counts=True)
    probs = counts / len(action_sequence)

    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return float(entropy)


def compute_goal_directedness(
    trajectory: np.ndarray,
    goal_position: Tuple[int, int],
) -> float:
    """
    Compute goal-directedness of a trajectory.

    Measures what fraction of movements decreased distance to goal.

    Args:
        trajectory: Sequence of positions
        goal_position: Target position

    Returns:
        Score in [0, 1] where 1 = perfectly goal-directed
    """
    if len(trajectory) < 2:
        return 0.0

    goal = np.array(goal_position)
    distances = np.linalg.norm(trajectory - goal, axis=1)

    # Count movements toward goal
    toward_goal = np.sum(np.diff(distances) < 0)
    total_moves = len(distances) - 1

    if total_moves == 0:
        return 0.0

    return float(toward_goal / total_moves)


def compare_agents(
    results: Dict[str, TestResult],
    test_name: str = "frustration_intensification",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Statistical comparison of agent test results.

    Uses Kruskal-Wallis test for overall comparison and
    Mann-Whitney U for pairwise comparisons with Bonferroni correction.

    Args:
        results: Dictionary of agent_name -> TestResult
        test_name: Name of test being compared
        alpha: Significance level

    Returns:
        Dictionary with statistical comparison results
    """
    agent_names = list(results.keys())
    n_agents = len(agent_names)

    if n_agents < 2:
        return {"error": "Need at least 2 agents for comparison"}

    # Extract trial scores for each agent
    agent_scores = {}
    for name, result in results.items():
        if "trial_scores" in result.raw_data:
            agent_scores[name] = result.raw_data["trial_scores"]
        else:
            agent_scores[name] = [result.score]

    # Kruskal-Wallis test (non-parametric ANOVA)
    score_groups = [agent_scores[name] for name in agent_names]
    try:
        kw_stat, kw_pvalue = stats.kruskal(*score_groups)
    except ValueError:
        kw_stat, kw_pvalue = 0.0, 1.0

    # Pairwise Mann-Whitney U tests with Bonferroni correction
    n_comparisons = n_agents * (n_agents - 1) // 2
    corrected_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    pairwise_results = {}
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            name1, name2 = agent_names[i], agent_names[j]
            scores1, scores2 = agent_scores[name1], agent_scores[name2]

            try:
                u_stat, u_pvalue = stats.mannwhitneyu(
                    scores1, scores2, alternative='two-sided'
                )
            except ValueError:
                u_stat, u_pvalue = 0.0, 1.0

            # Effect size (rank-biserial correlation)
            n1, n2 = len(scores1), len(scores2)
            effect_size = 1 - (2 * u_stat) / (n1 * n2) if n1 * n2 > 0 else 0.0

            pairwise_results[f"{name1}_vs_{name2}"] = {
                "u_statistic": float(u_stat),
                "p_value": float(u_pvalue),
                "significant": u_pvalue < corrected_alpha,
                "effect_size": float(effect_size),
                "mean_diff": float(np.mean(scores1) - np.mean(scores2)),
            }

    # Summary statistics per agent
    summary_stats = {}
    for name, scores in agent_scores.items():
        summary_stats[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "n": len(scores),
        }

    return {
        "test_name": test_name,
        "kruskal_wallis": {
            "statistic": float(kw_stat),
            "p_value": float(kw_pvalue),
            "significant": kw_pvalue < alpha,
        },
        "pairwise_comparisons": pairwise_results,
        "summary_stats": summary_stats,
        "alpha": alpha,
        "corrected_alpha": corrected_alpha,
    }


def compute_survival_rate(episodes: List[EpisodeLog]) -> float:
    """Compute the fraction of episodes where agent survived."""
    if not episodes:
        return 0.0
    survived = sum(1 for e in episodes if e.survived)
    return survived / len(episodes)


def compute_mean_reward(episodes: List[EpisodeLog]) -> float:
    """Compute mean total reward across episodes."""
    if not episodes:
        return 0.0
    return float(np.mean([e.total_reward for e in episodes]))


def compute_mean_steps(episodes: List[EpisodeLog]) -> float:
    """Compute mean episode length."""
    if not episodes:
        return 0.0
    return float(np.mean([e.total_steps for e in episodes]))


def aggregate_metrics(episodes: List[EpisodeLog]) -> Dict[str, float]:
    """
    Compute aggregate metrics across a set of episodes.

    Args:
        episodes: List of episode logs

    Returns:
        Dictionary of metric name -> value
    """
    if not episodes:
        return {}

    # Basic statistics
    total_rewards = [e.total_reward for e in episodes]
    total_steps = [e.total_steps for e in episodes]
    final_energies = [e.final_energy for e in episodes]

    # Action distribution
    all_actions = np.concatenate([e.get_action_sequence() for e in episodes])
    action_entropy = compute_action_entropy(all_actions)

    # Survival
    survival_rate = compute_survival_rate(episodes)

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_steps": float(np.mean(total_steps)),
        "std_steps": float(np.std(total_steps)),
        "mean_final_energy": float(np.mean(final_energies)),
        "survival_rate": survival_rate,
        "action_entropy": action_entropy,
        "n_episodes": len(episodes),
    }
