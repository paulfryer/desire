"""
Visualization module for DESIRE research.

Provides comprehensive visualization tools including dashboards,
real-time plotting, video comparison, and heatmaps.
"""

from src.visualization.heatmaps import (
    create_activation_heatmap,
    create_visitation_heatmap,
    create_comparison_heatmap,
)

__all__ = [
    "create_activation_heatmap",
    "create_visitation_heatmap",
    "create_comparison_heatmap",
]
