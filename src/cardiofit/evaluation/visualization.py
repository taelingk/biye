"""Visualization utilities — prediction scatter, error distribution."""

import numpy as np
import matplotlib.pyplot as plt


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    xlabel: str = "Ground Truth",
    ylabel: str = "Prediction",
    unit: str = "",
    figsize: tuple = (6, 6),
    save_path: str | None = None,
):
    """Scatter plot of predicted vs actual values with identity line."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.5, s=10, c="steelblue", edgecolors="none")

    # Identity line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Identity")

    # R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    from scipy.stats import pearsonr
    r, _ = pearsonr(y_true, y_pred)

    ax.text(0.05, 0.95, f"R² = {r2:.4f}\nr = {r:.4f}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel(f"{xlabel} {unit}".strip())
    ax.set_ylabel(f"{ylabel} {unit}".strip())
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Error Distribution",
    unit: str = "",
    figsize: tuple = (8, 5),
    save_path: str | None = None,
):
    """Histogram of prediction errors with normal fit overlay."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    errors = y_pred - y_true

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(errors, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="white")

    # Normal fit
    from scipy.stats import norm
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma), "r-", linewidth=2, label=f"N({mu:.2f}, {sigma:.2f}²)")

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Error {unit}".strip())
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_curves(history, save_path: str | None = None):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    if "lr" in history.history:
        axes[1].plot(history.history["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
