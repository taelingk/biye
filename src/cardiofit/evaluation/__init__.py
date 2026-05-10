"""Evaluation — metrics, Bland-Altman analysis, visualization."""

from .metrics import compute_all_metrics, evaluate_per_subject, format_metrics
from .bland_altman import bland_altman_analysis, plot_bland_altman
from .visualization import plot_prediction_scatter, plot_error_distribution, plot_training_curves

__all__ = [
    "compute_all_metrics",
    "evaluate_per_subject",
    "format_metrics",
    "bland_altman_analysis",
    "plot_bland_altman",
    "plot_prediction_scatter",
    "plot_error_distribution",
    "plot_training_curves",
]
