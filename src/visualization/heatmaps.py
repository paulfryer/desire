"""
Heatmap visualizations for agent behavior analysis.

Provides activation heatmaps, visitation heatmaps, and comparison views.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.utils.logging import EpisodeLog


def create_activation_heatmap(
    episodes: List[EpisodeLog],
    grid_size: int,
    title: str = "Activation Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create heatmap of internal state activations across the grid.

    Args:
        episodes: Episode logs with internal states
        grid_size: Size of the environment grid
        title: Plot title
        figsize: Figure size
        cmap: Colormap to use
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    # Initialize activation grid
    activation_sum = np.zeros((grid_size, grid_size))
    activation_count = np.zeros((grid_size, grid_size))

    # Aggregate activations by position
    for episode in episodes:
        for step in episode.steps:
            if step.internal_state is not None:
                x, y = int(step.position[0]), int(step.position[1])
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    activation = np.linalg.norm(step.internal_state)
                    activation_sum[x, y] += activation
                    activation_count[x, y] += 1

    # Compute mean activation
    with np.errstate(divide='ignore', invalid='ignore'):
        activation_mean = np.divide(activation_sum, activation_count)
        activation_mean[~np.isfinite(activation_mean)] = 0

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(activation_mean, cmap=cmap, origin='lower')

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Mean Activation")

    ax.set_title(title)
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_visitation_heatmap(
    episodes: List[EpisodeLog],
    grid_size: int,
    title: str = "Visitation Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    log_scale: bool = True,
    cmap: str = "hot",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create heatmap of position visitation frequency.

    Args:
        episodes: Episode logs
        grid_size: Size of the environment grid
        title: Plot title
        figsize: Figure size
        log_scale: Whether to use log scale for counts
        cmap: Colormap to use
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    # Count visitations
    visitation_count = np.zeros((grid_size, grid_size))

    for episode in episodes:
        trajectory = episode.get_trajectory()
        for pos in trajectory:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < grid_size and 0 <= y < grid_size:
                visitation_count[x, y] += 1

    # Apply log scale if requested
    if log_scale:
        visitation_count = np.log1p(visitation_count)
        label = "Log(Visits + 1)"
    else:
        label = "Visit Count"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(visitation_count, cmap=cmap, origin='lower')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label=label)

    ax.set_title(title)
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_comparison_heatmap(
    episodes_dict: Dict[str, List[EpisodeLog]],
    grid_size: int,
    metric: str = "activation",
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "viridis",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create side-by-side heatmap comparison across agents.

    Args:
        episodes_dict: Dictionary of agent_name -> episodes
        grid_size: Environment grid size
        metric: Which metric to visualize ("activation" or "visitation")
        figsize: Figure size (auto-calculated if None)
        cmap: Colormap to use
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n_agents = len(episodes_dict)

    if figsize is None:
        figsize = (5 * n_agents, 5)

    fig, axes = plt.subplots(1, n_agents, figsize=figsize)
    if n_agents == 1:
        axes = [axes]

    # Compute heatmaps for each agent
    heatmaps = {}
    for name, episodes in episodes_dict.items():
        if metric == "activation":
            heatmap = _compute_activation_map(episodes, grid_size)
        else:
            heatmap = _compute_visitation_map(episodes, grid_size)
        heatmaps[name] = heatmap

    # Find global min/max for consistent scale
    all_values = np.concatenate([h.flatten() for h in heatmaps.values()])
    vmin, vmax = np.percentile(all_values[all_values > 0], [5, 95]) if np.any(all_values > 0) else (0, 1)

    # Plot each agent
    for ax, (name, heatmap) in zip(axes, heatmaps.items()):
        im = ax.imshow(heatmap, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Y")
        ax.set_ylabel("X")

    # Add shared colorbar
    fig.colorbar(im, ax=axes, label=metric.capitalize(), fraction=0.02, pad=0.04)

    plt.suptitle(f"{metric.capitalize()} Comparison", y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _compute_activation_map(episodes: List[EpisodeLog], grid_size: int) -> np.ndarray:
    """Compute activation map from episodes."""
    activation_sum = np.zeros((grid_size, grid_size))
    activation_count = np.zeros((grid_size, grid_size))

    for episode in episodes:
        for step in episode.steps:
            if step.internal_state is not None:
                x, y = int(step.position[0]), int(step.position[1])
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    activation = np.linalg.norm(step.internal_state)
                    activation_sum[x, y] += activation
                    activation_count[x, y] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(activation_sum, activation_count)
        result[~np.isfinite(result)] = 0

    return result


def _compute_visitation_map(episodes: List[EpisodeLog], grid_size: int) -> np.ndarray:
    """Compute visitation map from episodes."""
    visitation = np.zeros((grid_size, grid_size))

    for episode in episodes:
        trajectory = episode.get_trajectory()
        for pos in trajectory:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < grid_size and 0 <= y < grid_size:
                visitation[x, y] += 1

    return visitation


def create_trajectory_overlay(
    episodes: List[EpisodeLog],
    grid_size: int,
    resource_positions: Optional[List[Tuple[int, int]]] = None,
    max_trajectories: int = 10,
    title: str = "Trajectory Overlay",
    figsize: Tuple[int, int] = (10, 10),
    alpha: float = 0.3,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create overlay of multiple trajectories on the grid.

    Args:
        episodes: Episode logs
        grid_size: Environment grid size
        resource_positions: Optional list of resource positions to mark
        max_trajectories: Maximum trajectories to display
        title: Plot title
        figsize: Figure size
        alpha: Trajectory transparency
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot grid
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)

    # Mark resources
    if resource_positions:
        for rx, ry in resource_positions:
            ax.scatter(ry, rx, s=200, c='green', marker='*', zorder=3, label='Resource')

    # Plot trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, min(len(episodes), max_trajectories)))

    for i, episode in enumerate(episodes[:max_trajectories]):
        trajectory = episode.get_trajectory()
        if len(trajectory) > 0:
            xs = [p[1] for p in trajectory]  # y coordinates for plot x
            ys = [p[0] for p in trajectory]  # x coordinates for plot y

            ax.plot(xs, ys, color=colors[i], alpha=alpha, linewidth=1)
            ax.scatter(xs[0], ys[0], color=colors[i], marker='o', s=50, zorder=2)  # Start
            ax.scatter(xs[-1], ys[-1], color=colors[i], marker='x', s=50, zorder=2)  # End

    ax.set_title(title)
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_distance_activation_profile(
    episodes: List[EpisodeLog],
    max_distance: int = 15,
    title: str = "Activation vs Distance to Goal",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create plot of activation as function of distance to goal.

    Args:
        episodes: Episode logs
        max_distance: Maximum distance to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Collect (distance, activation) pairs
    distance_bins = {i: [] for i in range(max_distance + 1)}

    for episode in episodes:
        for step in episode.steps:
            if step.internal_state is None:
                continue

            distance = step.info.get("nearest_resource_distance")
            if distance is None or distance > max_distance:
                continue

            activation = np.linalg.norm(step.internal_state)
            bin_idx = int(distance)
            distance_bins[bin_idx].append(activation)

    # Compute mean and std for each bin
    distances = []
    means = []
    stds = []

    for d in range(max_distance + 1):
        if distance_bins[d]:
            distances.append(d)
            means.append(np.mean(distance_bins[d]))
            stds.append(np.std(distance_bins[d]))

    distances = np.array(distances)
    means = np.array(means)
    stds = np.array(stds)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(distances, means, 'b-', linewidth=2, label='Mean activation')
    ax.fill_between(distances, means - stds, means + stds, alpha=0.3, label='1 SD')

    ax.set_xlabel("Distance to Nearest Resource")
    ax.set_ylabel("Internal State Activation (norm)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
