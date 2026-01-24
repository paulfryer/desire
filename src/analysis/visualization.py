"""
Visualization utilities for behavioral analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from src.utils.logging import EpisodeLog
from src.tests.base_test import TestResult


# Color scheme for agent types
AGENT_COLORS = {
    "heuristic": "#2ecc71",      # Green
    "rl_ppo": "#3498db",         # Blue
    "active_inference": "#e74c3c",  # Red
    "default": "#95a5a6",        # Gray
}


def get_agent_color(agent_name: str) -> str:
    """Get color for an agent type."""
    for key, color in AGENT_COLORS.items():
        if key in agent_name.lower():
            return color
    return AGENT_COLORS["default"]


def plot_trajectories(
    episodes: List[EpisodeLog],
    grid_size: int = 20,
    title: str = "Agent Trajectories",
    ax: Optional[plt.Axes] = None,
    show_start_end: bool = True,
    alpha: float = 0.5,
    max_episodes: int = 10,
) -> plt.Figure:
    """
    Plot agent trajectories on the grid.

    Args:
        episodes: List of episode logs
        grid_size: Size of the grid
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        show_start_end: Whether to mark start and end points
        alpha: Trajectory transparency
        max_episodes: Maximum episodes to plot

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Plot grid
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot trajectories
    episodes_to_plot = episodes[:max_episodes]

    for i, episode in enumerate(episodes_to_plot):
        trajectory = episode.get_trajectory()
        if len(trajectory) == 0:
            continue

        color = get_agent_color(episode.agent_type)

        # Plot path
        ax.plot(
            trajectory[:, 1], trajectory[:, 0],
            color=color, alpha=alpha,
            linewidth=1, label=episode.agent_type if i == 0 else None
        )

        if show_start_end:
            # Start point
            ax.scatter(
                trajectory[0, 1], trajectory[0, 0],
                color=color, marker='o', s=100,
                edgecolor='white', linewidth=2, zorder=5
            )
            # End point
            ax.scatter(
                trajectory[-1, 1], trajectory[-1, 0],
                color=color, marker='X', s=100,
                edgecolor='white', linewidth=2, zorder=5
            )

    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.invert_yaxis()

    return fig


def plot_internal_states(
    episodes: List[EpisodeLog],
    title: str = "Internal State Evolution",
    ax: Optional[plt.Axes] = None,
    max_episodes: int = 5,
    dimensionality_reduction: str = "pca",
) -> plt.Figure:
    """
    Visualize internal state evolution over time.

    Args:
        episodes: List of episode logs
        title: Plot title
        ax: Matplotlib axes
        max_episodes: Maximum episodes to plot
        dimensionality_reduction: Method for reducing dimensions (pca, tsne)

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    episodes_to_plot = episodes[:max_episodes]

    for episode in episodes_to_plot:
        internal_states = episode.get_internal_states()

        if internal_states is None or len(internal_states) == 0:
            continue

        # Reduce dimensionality if needed
        if internal_states.shape[1] > 2:
            if dimensionality_reduction == "pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(internal_states)
            else:
                # Just take first two dimensions
                reduced = internal_states[:, :2]
        else:
            reduced = internal_states

        color = get_agent_color(episode.agent_type)
        steps = np.arange(len(reduced))

        # Plot as colored line
        points = reduced.reshape(-1, 1, 2)
        from matplotlib.collections import LineCollection
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', alpha=0.7)
        lc.set_array(steps[:-1])
        ax.add_collection(lc)

        # Mark start and end
        ax.scatter(reduced[0, 0], reduced[0, 1], color=color, marker='o', s=100, zorder=5)
        ax.scatter(reduced[-1, 0], reduced[-1, 1], color=color, marker='X', s=100, zorder=5)

    ax.autoscale()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)

    return fig


def plot_learning_curves(
    training_data: Dict[str, List[float]],
    title: str = "Learning Curves",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    ax: Optional[plt.Axes] = None,
    smoothing: int = 10,
) -> plt.Figure:
    """
    Plot learning curves for different agents.

    Args:
        training_data: Dict mapping agent_name to list of rewards
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        ax: Matplotlib axes
        smoothing: Window size for smoothing

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    for agent_name, rewards in training_data.items():
        color = get_agent_color(agent_name)
        rewards = np.array(rewards)

        # Smooth the curve
        if smoothing > 1 and len(rewards) >= smoothing:
            kernel = np.ones(smoothing) / smoothing
            smoothed = np.convolve(rewards, kernel, mode='valid')
            x = np.arange(smoothing - 1, len(rewards))
        else:
            smoothed = rewards
            x = np.arange(len(rewards))

        ax.plot(x, smoothed, color=color, label=agent_name, linewidth=2)

        # Add confidence band
        if len(rewards) >= smoothing * 2:
            std = np.std(rewards)
            ax.fill_between(x, smoothed - std, smoothed + std, color=color, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_predicate_scores(
    results: Dict[str, TestResult],
    title: str = "Predicate Test Scores",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing predicate test scores across agents.

    Args:
        results: Dict mapping agent_name to TestResult
        title: Plot title
        ax: Matplotlib axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    agent_names = list(results.keys())
    scores = [results[name].score for name in agent_names]
    colors = [get_agent_color(name) for name in agent_names]

    # Get standard deviations if available
    stds = []
    for name in agent_names:
        raw_data = results[name].raw_data
        if "std_score" in raw_data:
            stds.append(raw_data["std_score"])
        elif "trial_scores" in raw_data:
            stds.append(np.std(raw_data["trial_scores"]))
        else:
            stds.append(0)

    x = np.arange(len(agent_names))
    width = 0.6

    bars = ax.bar(x, scores, width, color=colors, edgecolor='black', linewidth=1)

    # Add error bars
    ax.errorbar(x, scores, yerr=stds, fmt='none', color='black', capsize=5)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f'{score:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10
        )

    ax.set_xlabel("Agent Type")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_frustration_response(
    baseline_intensity: np.ndarray,
    post_blockage_intensity: np.ndarray,
    blockage_step: int,
    agent_name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot frustration intensification response.

    Args:
        baseline_intensity: Intensity during baseline
        post_blockage_intensity: Intensity after blockage
        blockage_step: Step when blockage occurred
        agent_name: Name of the agent
        ax: Matplotlib axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    color = get_agent_color(agent_name)

    # Plot baseline
    x_baseline = np.arange(len(baseline_intensity))
    ax.plot(x_baseline, baseline_intensity, color=color, alpha=0.5, label='Baseline')
    ax.axhline(
        y=np.mean(baseline_intensity),
        color=color, linestyle='--', alpha=0.7,
        label=f'Baseline mean ({np.mean(baseline_intensity):.2f})'
    )

    # Plot post-blockage
    x_post = np.arange(blockage_step, blockage_step + len(post_blockage_intensity))
    ax.plot(x_post, post_blockage_intensity, color='red', linewidth=2, label='Post-blockage')

    # Mark peak
    peak_idx = np.argmax(post_blockage_intensity)
    peak_val = post_blockage_intensity[peak_idx]
    ax.scatter(
        [blockage_step + peak_idx], [peak_val],
        color='red', s=100, zorder=5, marker='^',
        label=f'Peak ({peak_val:.2f})'
    )

    # Mark blockage
    ax.axvline(x=blockage_step, color='black', linestyle=':', label='Blockage')

    ax.set_xlabel("Step")
    ax.set_ylabel("Action Intensity")
    ax.set_title(f"Frustration Response - {agent_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_comparison_grid(
    results: Dict[str, Dict[str, TestResult]],
    test_names: List[str],
    agent_names: List[str],
    title: str = "Agent Comparison Across Tests",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create a grid showing all agents across all tests.

    Args:
        results: Nested dict: results[test_name][agent_name] = TestResult
        test_names: List of test names
        agent_names: List of agent names
        title: Overall title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_tests = len(test_names)
    n_agents = len(agent_names)

    fig, axes = plt.subplots(1, n_tests, figsize=figsize)
    if n_tests == 1:
        axes = [axes]

    for i, test_name in enumerate(test_names):
        ax = axes[i]

        if test_name in results:
            test_results = results[test_name]
            scores = [test_results.get(a, TestResult(test_name, a, 0.0)).score for a in agent_names]
            colors = [get_agent_color(a) for a in agent_names]

            x = np.arange(n_agents)
            ax.bar(x, scores, color=colors, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(agent_names, rotation=45, ha='right')
            ax.set_title(test_name.replace('_', ' ').title())
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int = 150,
    formats: List[str] = ["png", "pdf"],
) -> None:
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        dpi: Resolution for raster formats
        formats: List of formats to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(
            path.with_suffix(f".{fmt}"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
        )


def create_summary_report(
    results: Dict[str, TestResult],
    output_path: Path,
    training_data: Optional[Dict[str, List[float]]] = None,
) -> None:
    """
    Create a comprehensive visual summary report.

    Args:
        results: Test results for all agents
        output_path: Directory to save figures
        training_data: Optional training curves data
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Predicate score comparison
    fig_scores = plot_predicate_scores(results, title="Behavioral Predicate Scores")
    save_figure(fig_scores, output_path / "predicate_scores")
    plt.close(fig_scores)

    # 2. Learning curves (if available)
    if training_data:
        fig_learning = plot_learning_curves(training_data)
        save_figure(fig_learning, output_path / "learning_curves")
        plt.close(fig_learning)

    print(f"Summary report saved to {output_path}")
