"""
Real-time plotting components for live experiment monitoring.

Provides streaming plots for training metrics, episode statistics,
and agent behavior visualization.
"""

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class RealtimePlotter:
    """
    Real-time plotter for streaming data visualization.

    Supports multiple subplots with automatic updating.
    """

    def __init__(
        self,
        n_plots: int = 1,
        plot_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 4),
        max_points: int = 1000,
        update_interval: int = 100,
    ):
        """
        Initialize real-time plotter.

        Args:
            n_plots: Number of subplot panels
            plot_names: Names for each subplot
            figsize: Figure size
            max_points: Maximum points to display
            update_interval: Milliseconds between updates
        """
        self.n_plots = n_plots
        self.max_points = max_points
        self.update_interval = update_interval

        if plot_names is None:
            plot_names = [f"Plot {i+1}" for i in range(n_plots)]
        self.plot_names = plot_names

        # Data storage
        self.data: Dict[str, deque] = {
            name: deque(maxlen=max_points) for name in plot_names
        }
        self.x_data: deque = deque(maxlen=max_points)
        self.step = 0

        # Create figure
        self.fig, self.axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            self.axes = [self.axes]

        self.lines: Dict[str, Any] = {}
        for ax, name in zip(self.axes, plot_names):
            (line,) = ax.plot([], [], 'b-', linewidth=1)
            self.lines[name] = line
            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    def add_point(self, values: Dict[str, float]) -> None:
        """
        Add a data point to the plots.

        Args:
            values: Dictionary of plot_name -> value
        """
        self.step += 1
        self.x_data.append(self.step)

        for name, value in values.items():
            if name in self.data:
                self.data[name].append(value)

    def update(self) -> None:
        """Update the plot display."""
        x = list(self.x_data)

        for ax, name in zip(self.axes, self.plot_names):
            y = list(self.data[name])
            if x and y:
                self.lines[name].set_data(x[-len(y):], y)
                ax.relim()
                ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self) -> None:
        """Display the plot."""
        plt.ion()
        plt.show()

    def save(self, path: str) -> None:
        """Save current figure."""
        self.fig.savefig(path, dpi=150, bbox_inches='tight')


class TrainingMonitor:
    """
    Real-time training monitor for RL agents.

    Displays loss, reward, and other training metrics.
    """

    def __init__(
        self,
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (14, 6),
        window_size: int = 100,
    ):
        """
        Initialize training monitor.

        Args:
            metrics: Metrics to track
            figsize: Figure size
            window_size: Moving average window
        """
        if metrics is None:
            metrics = ["reward", "loss", "entropy"]

        self.metrics = metrics
        self.window_size = window_size

        # Data storage
        self.raw_data: Dict[str, List[float]] = {m: [] for m in metrics}
        self.smoothed_data: Dict[str, List[float]] = {m: [] for m in metrics}

        # Create figure
        n_cols = min(3, len(metrics))
        n_rows = (len(metrics) + n_cols - 1) // n_cols

        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        self.axes = np.atleast_2d(self.axes).flatten()

        self.lines: Dict[str, Tuple[Any, Any]] = {}
        for ax, metric in zip(self.axes[:len(metrics)], metrics):
            (raw_line,) = ax.plot([], [], 'b-', alpha=0.3, label='Raw')
            (smooth_line,) = ax.plot([], [], 'r-', linewidth=2, label='Smoothed')
            self.lines[metric] = (raw_line, smooth_line)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Episode")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    def log(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics for an episode.

        Args:
            metrics: Dictionary of metric_name -> value
        """
        for name, value in metrics.items():
            if name in self.raw_data:
                self.raw_data[name].append(value)

                # Compute smoothed value
                raw = self.raw_data[name]
                if len(raw) >= self.window_size:
                    smoothed = np.mean(raw[-self.window_size:])
                else:
                    smoothed = np.mean(raw)
                self.smoothed_data[name].append(smoothed)

    def update(self) -> None:
        """Update the display."""
        for metric in self.metrics:
            if not self.raw_data[metric]:
                continue

            x = list(range(len(self.raw_data[metric])))
            raw_line, smooth_line = self.lines[metric]

            raw_line.set_data(x, self.raw_data[metric])
            smooth_line.set_data(x[:len(self.smoothed_data[metric])], self.smoothed_data[metric])

            ax = self.axes[self.metrics.index(metric)]
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self) -> None:
        """Display the monitor."""
        plt.ion()
        plt.show()


class EpisodeVisualizer:
    """
    Real-time episode visualization showing agent behavior.

    Displays grid state, trajectory, and internal state.
    """

    def __init__(
        self,
        grid_size: int,
        figsize: Tuple[int, int] = (14, 5),
    ):
        """
        Initialize episode visualizer.

        Args:
            grid_size: Environment grid size
            figsize: Figure size
        """
        self.grid_size = grid_size

        # Create figure with subplots
        self.fig, (self.ax_grid, self.ax_energy, self.ax_internal) = plt.subplots(
            1, 3, figsize=figsize
        )

        # Grid display
        self.grid_img = self.ax_grid.imshow(
            np.zeros((grid_size, grid_size, 3)),
            origin='lower',
        )
        self.ax_grid.set_title("Environment")

        # Energy plot
        self.energy_line, = self.ax_energy.plot([], [], 'g-', linewidth=2)
        self.ax_energy.set_title("Energy")
        self.ax_energy.set_xlabel("Step")
        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.set_xlim(0, 100)
        self.ax_energy.set_ylim(0, 150)
        self.ax_energy.grid(True, alpha=0.3)

        # Internal state plot
        self.internal_line, = self.ax_internal.plot([], [], 'b-', linewidth=1)
        self.ax_internal.set_title("Internal State Norm")
        self.ax_internal.set_xlabel("Step")
        self.ax_internal.set_ylabel("Activation")
        self.ax_internal.set_xlim(0, 100)
        self.ax_internal.grid(True, alpha=0.3)

        plt.tight_layout()

        # Data storage
        self.energy_history: List[float] = []
        self.internal_history: List[float] = []
        self.trajectory: List[Tuple[int, int]] = []

    def update(
        self,
        position: Tuple[int, int],
        energy: float,
        internal_state: Optional[np.ndarray],
        resources: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """
        Update visualization with new state.

        Args:
            position: Agent position
            energy: Agent energy
            internal_state: Agent internal state
            resources: Resource positions
        """
        # Update trajectory
        self.trajectory.append(position)

        # Update energy history
        self.energy_history.append(energy)

        # Update internal state history
        if internal_state is not None:
            self.internal_history.append(np.linalg.norm(internal_state))
        else:
            self.internal_history.append(0)

        # Update grid display
        grid = np.ones((self.grid_size, self.grid_size, 3)) * 0.9

        # Draw trajectory
        for pos in self.trajectory[:-1]:
            grid[pos[0], pos[1]] = [0.8, 0.8, 1.0]

        # Draw resources
        if resources:
            for rx, ry in resources:
                grid[rx, ry] = [0.2, 0.8, 0.2]

        # Draw agent
        grid[position[0], position[1]] = [1.0, 0.2, 0.2]

        self.grid_img.set_array(grid)

        # Update energy plot
        steps = list(range(len(self.energy_history)))
        self.energy_line.set_data(steps, self.energy_history)
        self.ax_energy.set_xlim(0, max(100, len(steps)))

        # Update internal state plot
        self.internal_line.set_data(steps, self.internal_history)
        self.ax_internal.set_xlim(0, max(100, len(steps)))
        if self.internal_history:
            self.ax_internal.set_ylim(0, max(self.internal_history) * 1.1 + 0.1)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def reset(self) -> None:
        """Reset for new episode."""
        self.energy_history = []
        self.internal_history = []
        self.trajectory = []

    def show(self) -> None:
        """Display the visualizer."""
        plt.ion()
        plt.show()


def create_live_comparison(
    agent_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create a live comparison figure for multiple agents.

    Args:
        agent_names: Names of agents to compare
        figsize: Figure size

    Returns:
        Tuple of (figure, update_functions)
    """
    n_agents = len(agent_names)

    fig, axes = plt.subplots(2, n_agents, figsize=figsize)

    # Top row: trajectories
    # Bottom row: metrics

    components = {}

    for i, name in enumerate(agent_names):
        # Trajectory axis
        ax_traj = axes[0, i] if n_agents > 1 else axes[0]
        ax_traj.set_title(f"{name} - Trajectory")
        components[f"{name}_trajectory"] = ax_traj

        # Metrics axis
        ax_metric = axes[1, i] if n_agents > 1 else axes[1]
        ax_metric.set_title(f"{name} - Reward")
        ax_metric.set_xlabel("Episode")
        components[f"{name}_metrics"] = ax_metric

    plt.tight_layout()

    return fig, components
