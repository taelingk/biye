#!/usr/bin/env python3
"""Evaluate validation-fitted CO bias correction for Phase 12.

This script runs inference only. It estimates one scalar CO bias per model on
the validation split, then applies that correction to the test split.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_phase11_errors import (
    DEFAULT_MODELS,
    ModelSpec,
    dataframe_to_markdown,
    load_subject_frames,
    predict_model,
)
from src.cardiofit.evaluation.metrics import compute_all_metrics
from src.cardiofit.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def compute_validation_bias(validation_predictions: pd.DataFrame) -> dict[str, float]:
    """Return mean validation error, defined as pred - true, for each model."""
    required = {"model", "co_true", "co_pred"}
    missing = required - set(validation_predictions.columns)
    if missing:
        raise ValueError(f"Missing validation columns: {sorted(missing)}")

    errors = validation_predictions.assign(
        co_error=validation_predictions["co_pred"]
        - validation_predictions["co_true"]
    )
    return {
        model: float(model_df["co_error"].mean())
        for model, model_df in errors.groupby("model", sort=False)
    }


def apply_bias_correction(
    test_predictions: pd.DataFrame, validation_bias: dict[str, float]
) -> pd.DataFrame:
    """Subtract each model's validation bias from test predictions."""
    required = {"model", "co_true", "co_pred"}
    missing = required - set(test_predictions.columns)
    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    corrected = test_predictions.copy()
    corrected["validation_bias"] = corrected["model"].map(validation_bias)
    if corrected["validation_bias"].isna().any():
        missing_models = corrected.loc[
            corrected["validation_bias"].isna(), "model"
        ].unique()
        raise ValueError(f"No validation bias for models: {sorted(missing_models)}")

    corrected["co_pred_corrected"] = (
        corrected["co_pred"] - corrected["validation_bias"]
    )
    corrected["co_error_before"] = corrected["co_pred"] - corrected["co_true"]
    corrected["co_error_corrected"] = (
        corrected["co_pred_corrected"] - corrected["co_true"]
    )
    corrected["co_abs_error_before"] = corrected["co_error_before"].abs()
    corrected["co_abs_error_corrected"] = corrected["co_error_corrected"].abs()
    return corrected


def _metrics_for_model(model_df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    metrics = compute_all_metrics(model_df["co_true"], model_df[pred_col])
    return {
        "r2": float(metrics["r2"]),
        "pearson_r": float(metrics["pearson_r"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "bias": float(metrics["bias"]),
    }


def build_bias_correction_summary(
    validation_predictions: pd.DataFrame, test_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build before/after test metrics using validation-fitted correction."""
    validation_bias = compute_validation_bias(validation_predictions)
    corrected = apply_bias_correction(test_predictions, validation_bias)

    rows = []
    for model, model_df in corrected.groupby("model", sort=False):
        before = _metrics_for_model(model_df, "co_pred")
        after = _metrics_for_model(model_df, "co_pred_corrected")
        rows.append(
            {
                "model": model,
                "validation_bias": validation_bias[model],
                "test_rmse_before": before["rmse"],
                "test_rmse_after": after["rmse"],
                "test_rmse_delta": after["rmse"] - before["rmse"],
                "test_mae_before": before["mae"],
                "test_mae_after": after["mae"],
                "test_mae_delta": after["mae"] - before["mae"],
                "test_bias_before": before["bias"],
                "test_bias_after": after["bias"],
                "test_r2_before": before["r2"],
                "test_r2_after": after["r2"],
                "test_pearson_before": before["pearson_r"],
                "test_pearson_after": after["pearson_r"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("test_rmse_after").reset_index(drop=True)
    return summary, corrected


def load_split_predictions(
    model_specs: tuple[ModelSpec, ...], split_name: str
) -> pd.DataFrame:
    """Run all model specs against one split."""
    with open(model_specs[0].config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from src.cardiofit.dataset import load_splits

    splits = load_splits(Path(cfg["paths"]["splits"]))
    subject_ids = splits[split_name]
    data_root = Path(cfg["paths"]["processed_data"])
    base_frame = load_subject_frames(subject_ids, data_root)
    logger.info("Loaded %s split: %d subjects, %d windows", split_name, len(subject_ids), len(base_frame))

    predictions = []
    for model_spec in model_specs:
        logger.info("Predicting %s on %s", model_spec.name, split_name)
        predictions.append(predict_model(model_spec, base_frame))

    combined = pd.concat(predictions, ignore_index=True)
    combined.insert(1, "split", split_name)
    return combined


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    """Write Phase 12 Markdown report."""
    best = summary.iloc[0]
    baseline = summary[summary["model"] == "phase9_baseline"].iloc[0]
    lines = [
        "# Phase 12 Bias Correction Evaluation",
        "",
        "Run date: 2026-06-03",
        "",
        "This is an inference-only evaluation. One scalar CO correction was fitted per model on validation predictions and applied to test predictions.",
        "",
        "Formula:",
        "",
        "```text",
        "co_pred_corrected = co_pred - mean(validation_pred - validation_true)",
        "```",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary, floatfmt=".6f"),
        "",
        "## Findings",
        "",
        f"- Best corrected test RMSE is `{best['model']}` at {best['test_rmse_after']:.6f} L/min.",
        f"- Phase 9 baseline corrected RMSE is {baseline['test_rmse_after']:.6f} L/min, compared with {baseline['test_rmse_before']:.6f} before correction.",
        "- Pearson r is unchanged by scalar correction; changes come from RMSE/MAE/bias shifts only.",
        "- If corrected RMSE improves but Pearson does not, the next training change should target calibration and subject weighting rather than model capacity.",
        "",
        "## Outputs",
        "",
        "- `validation_predictions.csv`",
        "- `test_predictions_corrected.csv`",
        "- `bias_correction_summary.csv`",
    ]
    (output_dir / "PHASE12_BIAS_CORRECTION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 12 bias correction")
    parser.add_argument("--output-dir", default="outputs/phase12_bias_correction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_predictions = load_split_predictions(DEFAULT_MODELS, "val")
    test_predictions = load_split_predictions(DEFAULT_MODELS, "test")
    summary, corrected = build_bias_correction_summary(
        validation_predictions, test_predictions
    )

    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    corrected.to_csv(output_dir / "test_predictions_corrected.csv", index=False)
    summary.to_csv(output_dir / "bias_correction_summary.csv", index=False)
    write_report(output_dir, summary)

    logger.info("Wrote Phase 12 bias correction outputs to %s", output_dir)
    logger.info("Summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
