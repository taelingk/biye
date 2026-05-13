"""Bland-Altman analysis for agreement assessment.

Mirrors the paper's Bland-Altman plots (Figures 3.13-3.32).
"""

import matplotlib.pyplot as plt
import numpy as np


def bland_altman_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Bland-Altman statistics.

    Args:
        y_true: (N,) ground truth.
        y_pred: (N,) predictions.

    Returns:
        Dict with mean_diff, std_diff, loa_lower, loa_upper, ci_lower, ci_upper.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    diffs = y_pred - y_true
    means = (y_true + y_pred) / 2.0

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)

    # Limits of agreement
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    # Confidence intervals for mean difference
    se_mean = std_diff / np.sqrt(n)
    ci_mean = 1.96 * se_mean

    # Confidence intervals for LoA
    se_loa = np.sqrt(3 * std_diff**2 / n)
    ci_loa = 1.96 * se_loa

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "ci_mean_lower": mean_diff - ci_mean,
        "ci_mean_upper": mean_diff + ci_mean,
        "ci_loa_lower_lower": loa_lower - ci_loa,
        "ci_loa_lower_upper": loa_lower + ci_loa,
        "ci_loa_upper_lower": loa_upper - ci_loa,
        "ci_loa_upper_upper": loa_upper + ci_loa,
        "means": means,
        "diffs": diffs,
    }


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Bland-Altman Plot",
    xlabel: str = "Mean (Ground Truth + Prediction) / 2",
    ylabel: str = "Difference (Prediction - Ground Truth)",
    unit: str = "",
    figsize: tuple = (8, 6),
    alpha: float = 0.5,
    save_path: str | None = None,
):
    """Plot Bland-Altman figure matching paper style.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        unit: Unit suffix for annotations.
        figsize: Figure size.
        alpha: Scatter point transparency.
        save_path: If provided, save figure to this path.
    """
    stats = bland_altman_analysis(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        stats["means"],
        stats["diffs"],
        alpha=alpha,
        s=10,
        c="steelblue",
        edgecolors="none",
    )
    ax.axhline(
        y=stats["mean_diff"],
        color="red",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean diff: {stats['mean_diff']:.2f} {unit}",
    )
    ax.axhline(
        y=stats["loa_upper"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"+1.96 SD: {stats['loa_upper']:.2f} {unit}",
    )
    ax.axhline(
        y=stats["loa_lower"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"-1.96 SD: {stats['loa_lower']:.2f} {unit}",
    )

    # CI bands for LoA
    ax.axhline(
        y=stats["ci_loa_upper_lower"], color="gray", linestyle=":", linewidth=0.8
    )
    ax.axhline(
        y=stats["ci_loa_upper_upper"], color="gray", linestyle=":", linewidth=0.8
    )
    ax.axhline(
        y=stats["ci_loa_lower_lower"], color="gray", linestyle=":", linewidth=0.8
    )
    ax.axhline(
        y=stats["ci_loa_lower_upper"], color="gray", linestyle=":", linewidth=0.8
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig, stats
