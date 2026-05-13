#!/usr/bin/env python3
"""Evaluate trained multimodal model on test set.

Usage:
    python scripts/evaluate_multimodal.py \
        --checkpoint outputs/checkpoints/best_model.keras
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cardiofit.dataset import load_multiple_subjects, load_splits
from src.cardiofit.evaluation.bland_altman import plot_bland_altman
from src.cardiofit.evaluation.metrics import evaluate_per_subject, format_metrics
from src.cardiofit.evaluation.visualization import (
    plot_error_distribution,
    plot_prediction_scatter,
)
from src.cardiofit.models import (
    ResidualSEBlock,
    SEBlock,
    Standardize1D,
    StandardizeClinical,
    StandardizeSignalFlat,
)
from src.cardiofit.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging()

    project_root = Path.cwd()
    data_root = project_root / cfg["paths"]["processed_data"]
    splits_dir = project_root / cfg["paths"]["splits"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load model ---
    model = tf.keras.models.load_model(
        args.checkpoint,
        custom_objects={
            "ResidualSEBlock": ResidualSEBlock,
            "SEBlock": SEBlock,
            "StandardizeSignalFlat": StandardizeSignalFlat,
            "StandardizeClinical": StandardizeClinical,
            "Standardize1D": Standardize1D,
        },
        compile=False,
    )
    logger.info(f"Loaded model from {args.checkpoint}")

    # --- 2. Load test subjects ---
    splits = load_splits(splits_dir)
    test_ids = splits["test"]
    logger.info(f"Test subjects: {test_ids}")

    signals, clinical, co_true, vo2_true = load_multiple_subjects(test_ids, data_root)
    logger.info(f"Test samples: {len(signals)}")

    # --- 3. Evaluate ---
    results = evaluate_per_subject(signals, clinical, co_true, vo2_true, model)

    co_metrics = results["co"]
    vo2_metrics = results["vo2"]

    logger.info("\n" + format_metrics(co_metrics, "CO"))
    logger.info("\n" + format_metrics(vo2_metrics, "VO2"))

    # --- 4. Generate figures ---
    co_pred, vo2_pred = model.predict([signals, clinical], verbose=0)

    # CO figures
    plot_prediction_scatter(
        co_true.ravel(),
        co_pred.ravel(),
        title="CO: Predicted vs Actual",
        xlabel="Ground Truth CO",
        ylabel="Predicted CO",
        unit="L/min",
        save_path=str(output_dir / "co_scatter.png"),
    )
    plot_bland_altman(
        co_true.ravel(),
        co_pred.ravel(),
        title="CO: Bland-Altman Plot",
        xlabel="Mean CO (L/min)",
        ylabel="Difference CO (L/min)",
        unit="L/min",
        save_path=str(output_dir / "co_bland_altman.png"),
    )
    plot_error_distribution(
        co_true.ravel(),
        co_pred.ravel(),
        title="CO: Error Distribution",
        unit="L/min",
        save_path=str(output_dir / "co_error_dist.png"),
    )

    # VO2 figures
    plot_prediction_scatter(
        vo2_true.ravel(),
        vo2_pred.ravel(),
        title="VO2: Predicted vs Actual",
        xlabel="Ground Truth VO2",
        ylabel="Predicted VO2",
        unit="mL/kg/min",
        save_path=str(output_dir / "vo2_scatter.png"),
    )
    plot_bland_altman(
        vo2_true.ravel(),
        vo2_pred.ravel(),
        title="VO2: Bland-Altman Plot",
        xlabel="Mean VO2 (mL/kg/min)",
        ylabel="Difference VO2 (mL/kg/min)",
        unit="mL/kg/min",
        save_path=str(output_dir / "vo2_bland_altman.png"),
    )
    plot_error_distribution(
        vo2_true.ravel(),
        vo2_pred.ravel(),
        title="VO2: Error Distribution",
        unit="mL/kg/min",
        save_path=str(output_dir / "vo2_error_dist.png"),
    )

    # --- 5. Save metrics JSON ---
    metrics_summary = {
        "co": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in co_metrics.items()
        },
        "vo2": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in vo2_metrics.items()
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
