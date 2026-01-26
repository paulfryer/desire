"""
Learning curve analysis for agent training dynamics.

Provides tools for analyzing how agents learn over time,
identifying learning phases, and comparing learning efficiency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal


@dataclass
class LearningCurveMetrics:
    """Metrics characterizing a learning curve."""

    # Basic statistics
    initial_performance: float
    final_performance: float
    max_performance: float
    total_improvement: float

    # Learning dynamics
    learning_rate: float  # Estimated from curve fit
    time_to_90_percent: int  # Steps to reach 90% of final
    plateau_step: int  # When learning plateaus

    # Stability
    variance_early: float  # Variance in first 10%
    variance_late: float  # Variance in last 10%
    stability_ratio: float  # late/early variance

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_performance": self.initial_performance,
            "final_performance": self.final_performance,
            "max_performance": self.max_performance,
            "total_improvement": self.total_improvement,
            "learning_rate": self.learning_rate,
            "time_to_90_percent": self.time_to_90_percent,
            "plateau_step": self.plateau_step,
            "variance_early": self.variance_early,
            "variance_late": self.variance_late,
            "stability_ratio": self.stability_ratio,
        }


def analyze_learning_curve(
    rewards: list[float],
    window_size: int = 10,
) -> LearningCurveMetrics:
    """
    Analyze a learning curve to extract key metrics.

    Args:
        rewards: List of episode rewards
        window_size: Window for smoothing

    Returns:
        LearningCurveMetrics
    """
    rewards = np.array(rewards)
    n = len(rewards)

    if n < window_size:
        window_size = max(1, n // 2)

    # Smooth the curve
    smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    # Basic statistics
    initial_performance = float(np.mean(rewards[:window_size]))
    final_performance = float(np.mean(rewards[-window_size:]))
    max_performance = float(np.max(smoothed)) if len(smoothed) > 0 else initial_performance
    total_improvement = final_performance - initial_performance

    # Fit exponential learning curve: y = a * (1 - exp(-bx)) + c
    try:
        x = np.arange(len(smoothed))
        y = smoothed

        def exp_curve(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c

        popt, _ = optimize.curve_fit(
            exp_curve, x, y,
            p0=[total_improvement, 0.01, initial_performance],
            maxfev=5000,
        )
        learning_rate = float(popt[1])
    except Exception:
        learning_rate = 0.01

    # Time to 90% of final
    target_90 = initial_performance + 0.9 * total_improvement
    time_to_90 = n
    for i, val in enumerate(smoothed):
        if val >= target_90:
            time_to_90 = i
            break

    # Find plateau (when improvement slows)
    if len(smoothed) > 10:
        gradient = np.gradient(smoothed)
        smoothed_gradient = np.convolve(
            gradient, np.ones(window_size)/window_size, mode='valid'
        )
        # Plateau when gradient consistently near zero
        threshold = 0.01 * np.abs(total_improvement) / n
        plateau_mask = np.abs(smoothed_gradient) < threshold
        plateau_indices = np.where(plateau_mask)[0]
        if len(plateau_indices) > 0:
            plateau_step = int(plateau_indices[0])
        else:
            plateau_step = n
    else:
        plateau_step = n

    # Variance analysis
    cutoff = max(1, n // 10)
    variance_early = float(np.var(rewards[:cutoff]))
    variance_late = float(np.var(rewards[-cutoff:]))
    stability_ratio = variance_late / variance_early if variance_early > 0 else 1.0

    return LearningCurveMetrics(
        initial_performance=initial_performance,
        final_performance=final_performance,
        max_performance=max_performance,
        total_improvement=total_improvement,
        learning_rate=learning_rate,
        time_to_90_percent=time_to_90,
        plateau_step=plateau_step,
        variance_early=variance_early,
        variance_late=variance_late,
        stability_ratio=stability_ratio,
    )


def compare_learning_curves(
    curves: dict[str, list[float]],
    window_size: int = 10,
) -> dict[str, Any]:
    """
    Compare learning curves across multiple agents.

    Args:
        curves: Dictionary of agent_name -> rewards
        window_size: Smoothing window

    Returns:
        Comparison results
    """
    metrics = {}
    for name, rewards in curves.items():
        metrics[name] = analyze_learning_curve(rewards, window_size)

    # Rankings
    final_perf_ranking = sorted(
        metrics.keys(),
        key=lambda x: metrics[x].final_performance,
        reverse=True
    )

    learning_speed_ranking = sorted(
        metrics.keys(),
        key=lambda x: metrics[x].time_to_90_percent
    )

    stability_ranking = sorted(
        metrics.keys(),
        key=lambda x: metrics[x].stability_ratio
    )

    return {
        "metrics": {name: m.to_dict() for name, m in metrics.items()},
        "rankings": {
            "final_performance": final_perf_ranking,
            "learning_speed": learning_speed_ranking,
            "stability": stability_ranking,
        },
        "summary": {
            "best_final_performance": final_perf_ranking[0],
            "fastest_learner": learning_speed_ranking[0],
            "most_stable": stability_ranking[0],
        },
    }


def plot_learning_curves(
    curves: dict[str, list[float]],
    title: str = "Learning Curves",
    figsize: tuple[int, int] = (12, 6),
    window_size: int = 10,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot learning curves with confidence intervals.

    Args:
        curves: Dictionary of agent_name -> rewards
        title: Plot title
        figsize: Figure size
        window_size: Smoothing window
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))

    # Left: Raw curves with smoothing
    for (name, rewards), color in zip(curves.items(), colors, strict=False):
        x = np.arange(len(rewards))

        # Raw data (light)
        ax1.plot(x, rewards, alpha=0.2, color=color)

        # Smoothed
        smoothed = np.convolve(
            rewards, np.ones(window_size)/window_size, mode='valid'
        )
        ax1.plot(np.arange(len(smoothed)), smoothed, label=name, color=color, linewidth=2)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Metrics comparison
    metrics = {name: analyze_learning_curve(rewards) for name, rewards in curves.items()}
    names = list(metrics.keys())

    x = np.arange(len(names))
    width = 0.35

    improvements = [metrics[n].total_improvement for n in names]
    final_perfs = [metrics[n].final_performance for n in names]

    ax2.bar(x - width/2, improvements, width, label='Improvement', alpha=0.7)
    ax2.bar(x + width/2, final_perfs, width, label='Final Performance', alpha=0.7)

    ax2.set_ylabel("Value")
    ax2.set_title("Learning Metrics")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def detect_learning_phases(
    rewards: list[float],
    n_phases: int = 3,
) -> dict[str, Any]:
    """
    Detect distinct learning phases in a curve.

    Args:
        rewards: Reward sequence
        n_phases: Number of phases to detect

    Returns:
        Phase information
    """
    rewards = np.array(rewards)
    n = len(rewards)

    if n < n_phases * 10:
        return {"error": "Insufficient data for phase detection"}

    # Use change point detection (simplified)
    # Divide into segments and compare
    segment_size = n // (n_phases * 2)

    gradient = np.gradient(rewards)
    smoothed_gradient = np.convolve(
        gradient, np.ones(segment_size)/segment_size, mode='valid'
    )

    # Find peaks in absolute gradient (change points)
    peaks, _ = signal.find_peaks(np.abs(smoothed_gradient), distance=segment_size)

    # Define phases
    if len(peaks) >= n_phases - 1:
        boundaries = [0] + list(peaks[:n_phases-1]) + [n]
    else:
        boundaries = list(np.linspace(0, n, n_phases + 1, dtype=int))

    phases = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        phase_rewards = rewards[start:end]

        phases.append({
            "phase": i + 1,
            "start": int(start),
            "end": int(end),
            "mean_reward": float(np.mean(phase_rewards)),
            "std_reward": float(np.std(phase_rewards)),
            "improvement": float(phase_rewards[-1] - phase_rewards[0]) if len(phase_rewards) > 1 else 0,
        })

    return {
        "n_phases": len(phases),
        "phases": phases,
        "boundaries": [int(b) for b in boundaries],
    }


def estimate_convergence(
    rewards: list[float],
    window_size: int = 50,
    threshold: float = 0.01,
) -> dict[str, Any]:
    """
    Estimate if and when learning has converged.

    Args:
        rewards: Reward sequence
        window_size: Window for variance estimation
        threshold: Improvement threshold for convergence

    Returns:
        Convergence analysis
    """
    rewards = np.array(rewards)
    n = len(rewards)

    if n < 2 * window_size:
        return {"converged": False, "reason": "Insufficient data"}

    # Compute rolling statistics
    rolling_mean = np.array([
        np.mean(rewards[max(0, i-window_size):i+1])
        for i in range(n)
    ])

    # Compute improvement rate
    improvements = np.diff(rolling_mean)

    # Check convergence criterion
    converged = False
    convergence_step = n

    for i in range(window_size, n - 1):
        recent_improvement = np.mean(np.abs(improvements[i-window_size:i]))
        if recent_improvement < threshold:
            converged = True
            convergence_step = i
            break

    return {
        "converged": converged,
        "convergence_step": int(convergence_step),
        "final_mean": float(np.mean(rewards[-window_size:])),
        "final_std": float(np.std(rewards[-window_size:])),
        "total_episodes": n,
    }
