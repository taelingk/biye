"""Evaluation metrics for CO and VO2max prediction.

Mirrors the paper: R², Pearson r, MAE, RMSE, bias, limits of agreement.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics.

    Args:
        y_true: (N,) or (N, 1) ground truth values.
        y_pred: (N,) or (N, 1) predicted values.

    Returns:
        Dict with keys: r2, pearson_r, pearson_p, mae, rmse, bias, loa_lower, loa_upper.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # R²
    r2 = r2_score(y_true, y_pred)

    # Pearson r
    r, p = pearsonr(y_true, y_pred)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Bias (mean error)
    errors = y_pred - y_true
    bias = np.mean(errors)

    # Limits of agreement (Bland-Altman)
    std_errors = np.std(errors)
    loa_lower = bias - 1.96 * std_errors
    loa_upper = bias + 1.96 * std_errors

    return {
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "std_error": std_errors,
        "n_samples": len(y_true),
    }


def evaluate_per_subject(
    signals: np.ndarray,
    clinical: np.ndarray,
    co_true: np.ndarray,
    vo2_true: np.ndarray,
    model,
) -> dict:
    """Evaluate model predictions and compute CO and VO2 metrics.

    Args:
        signals: (N, 125, 3) input windows.
        clinical: (N, 10) clinical features.
        co_true: (N, 1) CO ground truth.
        vo2_true: (N, 1) VO2 ground truth.
        model: Trained Keras model.

    Returns:
        {"co": {...}, "vo2": {...}}
    """
    co_pred, vo2_pred = model.predict([signals, clinical], verbose=0)

    return {
        "co": compute_all_metrics(co_true, co_pred),
        "vo2": compute_all_metrics(vo2_true, vo2_pred),
    }


def compute_vo2max_per_subject(
    vo2_predictions: np.ndarray,
    subject_indices: np.ndarray,
) -> np.ndarray:
    """Aggregate beat-level VO2 predictions to subject-level VO2max.

    VO2max = max VO2 value per subject across all exercise windows.

    Args:
        vo2_predictions: (N,) beat-level VO2 values.
        subject_indices: (N,) subject index for each beat.

    Returns:
        (n_subjects,) VO2max predictions.
    """
    unique_subjects = np.unique(subject_indices)
    vo2max = np.array([
        np.max(vo2_predictions[subject_indices == sid])
        for sid in unique_subjects
    ])
    return vo2max


def format_metrics(metrics: dict, name: str = "") -> str:
    """Format metrics dict as a readable string."""
    lines = [f"--- {name} Metrics ---" if name else "--- Metrics ---"]
    lines.append(f"R²:           {metrics['r2']:.4f}")
    lines.append(f"Pearson r:    {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4f})")
    lines.append(f"MAE:          {metrics['mae']:.4f}")
    lines.append(f"RMSE:         {metrics['rmse']:.4f}")
    lines.append(f"Bias:         {metrics['bias']:.4f}")
    lines.append(f"LoA:          [{metrics['loa_lower']:.4f}, {metrics['loa_upper']:.4f}]")
    lines.append(f"Std Error:    {metrics['std_error']:.4f}")
    lines.append(f"N:            {metrics['n_samples']}")
    return "\n".join(lines)
