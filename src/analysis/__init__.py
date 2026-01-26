"""
Analysis tools for the Desire research project.

Provides metrics computation, curve fitting, and visualization
for analyzing agent behavioral data.
"""

from src.analysis.fitting import (
    compare_fits,
    fit_exponential_decay,
    fit_power_law_decay,
)
from src.analysis.metrics import (
    compare_agents,
    compute_frustration_score,
    compute_means_end_score,
)
from src.analysis.visualization import (
    plot_internal_states,
    plot_learning_curves,
    plot_predicate_scores,
    plot_trajectories,
)

__all__ = [
    "compute_frustration_score",
    "compute_means_end_score",
    "compare_agents",
    "fit_exponential_decay",
    "fit_power_law_decay",
    "compare_fits",
    "plot_trajectories",
    "plot_internal_states",
    "plot_learning_curves",
    "plot_predicate_scores",
]
