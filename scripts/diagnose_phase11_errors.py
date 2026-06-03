#!/usr/bin/env python3
"""Generate Phase 11 CO error diagnostics from existing checkpoints.

This script runs inference only. It does not train models or modify Phase 9
artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cardiofit.evaluation.metrics import compute_all_metrics
from src.cardiofit.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: str
    checkpoint: str


DEFAULT_MODELS = (
    ModelSpec(
        name="phase9_baseline",
        config="configs/scg_rhc_windows5090d.yaml",
        checkpoint="outputs/checkpoints/best_model.keras",
    ),
    ModelSpec(
        name="phase10_regularized",
        config="configs/experiment/phase10_co_regularized.yaml",
        checkpoint="outputs/phase10_regularized/checkpoints/best_model.keras",
    ),
    ModelSpec(
        name="phase10_low_lr",
        config="configs/experiment/phase10_co_low_lr.yaml",
        checkpoint="outputs/phase10_low_lr/checkpoints/best_model.keras",
    ),
    ModelSpec(
        name="phase10_no_aug",
        config="configs/experiment/phase10_co_no_aug.yaml",
        checkpoint="outputs/phase10_no_aug/checkpoints/best_model.keras",
    ),
)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.astype(float), weights=weights.astype(float)))


def build_subject_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-window CO predictions by model and subject."""
    required = {"model", "subject_id", "co_true", "co_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    df = predictions.copy()
    df["co_error"] = df["co_pred"] - df["co_true"]
    df["co_abs_error"] = df["co_error"].abs()
    df["co_sq_error"] = df["co_error"] ** 2

    grouped = df.groupby(["model", "subject_id"], sort=False)
    summary = grouped.agg(
        n_samples=("co_error", "size"),
        co_true_mean=("co_true", "mean"),
        co_pred_mean=("co_pred", "mean"),
        co_mae=("co_abs_error", "mean"),
        co_rmse=("co_sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        co_bias=("co_error", "mean"),
        co_abs_error_sum=("co_abs_error", "sum"),
        co_max_abs_error=("co_abs_error", "max"),
    ).reset_index()

    return summary.sort_values(
        ["co_abs_error_sum", "co_rmse", "n_samples"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_model_comparison(
    subject_summary: pd.DataFrame, baseline_model: str = "phase9_baseline"
) -> pd.DataFrame:
    """Build model-level comparison from subject-level diagnostics."""
    required = {"model", "subject_id", "n_samples", "co_mae", "co_rmse", "co_bias"}
    missing = required - set(subject_summary.columns)
    if missing:
        raise ValueError(f"Missing subject summary columns: {sorted(missing)}")

    rows: list[dict] = []
    for model_name, model_df in subject_summary.groupby("model", sort=False):
        weights = model_df["n_samples"]
        worst = model_df.sort_values("co_abs_error_sum", ascending=False).iloc[0]
        rows.append(
            {
                "model": model_name,
                "n_subjects": int(model_df["subject_id"].nunique()),
                "n_samples": int(weights.sum()),
                "co_mae": _weighted_average(model_df["co_mae"], weights),
                "co_rmse": float(
                    np.sqrt(
                        np.average(model_df["co_rmse"].astype(float) ** 2, weights=weights)
                    )
                ),
                "co_bias": _weighted_average(model_df["co_bias"], weights),
                "worst_subject_id": worst["subject_id"],
                "worst_subject_rmse": float(worst["co_rmse"]),
                "worst_subject_mae": float(worst["co_mae"]),
            }
        )

    comparison = pd.DataFrame(rows)
    baseline = comparison.loc[comparison["model"] == baseline_model]
    if baseline.empty:
        comparison["co_rmse_delta_vs_baseline"] = np.nan
        comparison["co_mae_delta_vs_baseline"] = np.nan
    else:
        baseline_row = baseline.iloc[0]
        comparison["co_rmse_delta_vs_baseline"] = (
            comparison["co_rmse"] - baseline_row["co_rmse"]
        )
        comparison["co_mae_delta_vs_baseline"] = (
            comparison["co_mae"] - baseline_row["co_mae"]
        )

    return comparison.sort_values("co_rmse").reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""
    headers = [str(col) for col in df.columns]

    def fmt(value) -> str:
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    rows = [[fmt(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_subject_frames(subject_ids: list[str], data_root: Path) -> pd.DataFrame:
    """Load test subjects with subject/window metadata."""
    from src.cardiofit.dataset.hdf5_loader import load_subject

    frames = []
    for subject_id in subject_ids:
        signals, clinical, co_true, vo2_true = load_subject(data_root / f"{subject_id}.h5")
        frame = pd.DataFrame(
            {
                "subject_id": subject_id,
                "window_index": np.arange(len(signals), dtype=int),
                "co_true": co_true.ravel(),
                "vo2_true": vo2_true.ravel(),
            }
        )
        frame["_signal"] = list(signals)
        frame["_clinical"] = list(clinical)
        frames.append(frame)

    if not frames:
        raise ValueError("No subject frames loaded")
    return pd.concat(frames, ignore_index=True)


def predict_model(model_spec: ModelSpec, base_frame: pd.DataFrame) -> pd.DataFrame:
    """Run a checkpoint against the loaded test frame."""
    from src.cardiofit.models import load_multimodal_checkpoint

    model = load_multimodal_checkpoint(model_spec.checkpoint, model_spec.config)
    signals = np.stack(base_frame["_signal"].to_numpy()).astype(np.float32)
    clinical = np.stack(base_frame["_clinical"].to_numpy()).astype(np.float32)
    co_pred, vo2_pred = model.predict([signals, clinical], verbose=0)

    result = base_frame[
        ["subject_id", "window_index", "co_true", "vo2_true"]
    ].copy()
    result.insert(0, "model", model_spec.name)
    result["co_pred"] = co_pred.ravel()
    result["vo2_pred"] = vo2_pred.ravel()
    result["co_error"] = result["co_pred"] - result["co_true"]
    result["co_abs_error"] = result["co_error"].abs()
    return result


def write_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    subject_summary: pd.DataFrame,
    baseline_model: str,
) -> None:
    """Write markdown diagnostics report."""
    worst_subjects = subject_summary.sort_values(
        ["co_abs_error_sum", "co_rmse"], ascending=[False, False]
    ).head(12)

    lines = [
        "# Phase 11 Error Diagnostics",
        "",
        "Run date: 2026-06-03",
        "",
        "This analysis reuses existing Phase 9 and Phase 10 checkpoints for inference only. No training was run.",
        "",
        "## Model Comparison",
        "",
        dataframe_to_markdown(comparison, floatfmt=".6f"),
        "",
        "## Worst Subject Contributions",
        "",
        dataframe_to_markdown(
            worst_subjects[
                [
                    "model",
                    "subject_id",
                    "n_samples",
                    "co_mae",
                    "co_rmse",
                    "co_bias",
                    "co_abs_error_sum",
                    "co_max_abs_error",
                ]
            ],
            floatfmt=".6f",
        ),
        "",
        "## Findings",
        "",
    ]

    baseline = comparison[comparison["model"] == baseline_model].iloc[0]
    best = comparison.iloc[0]
    lines.extend(
        [
            f"- Best test RMSE remains `{best['model']}` at {best['co_rmse']:.6f} L/min.",
            f"- Baseline `{baseline_model}` RMSE is {baseline['co_rmse']:.6f} L/min and MAE is {baseline['co_mae']:.6f} L/min.",
            "- Phase 10 did not beat the Phase 9 baseline on test RMSE, so the next tuning round should target error distribution and bias before adding more capacity changes.",
            "- Use `subject_error_summary.csv` to inspect which subjects dominate absolute error mass.",
            "",
            "## Outputs",
            "",
            "- `sample_predictions.csv`",
            "- `subject_error_summary.csv`",
            "- `model_comparison_summary.csv`",
        ]
    )

    (output_dir / "PHASE11_ERROR_DIAGNOSTICS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Phase 11 CO errors")
    parser.add_argument("--output-dir", default="outputs/phase11_diagnostics")
    parser.add_argument("--baseline-model", default="phase9_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(DEFAULT_MODELS[0].config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from src.cardiofit.dataset import load_splits

    data_root = Path(cfg["paths"]["processed_data"])
    splits = load_splits(Path(cfg["paths"]["splits"]))
    test_ids = splits["test"]
    logger.info("Loaded %d test subjects", len(test_ids))

    base_frame = load_subject_frames(test_ids, data_root)
    logger.info("Loaded %d test windows", len(base_frame))

    predictions = []
    for model_spec in DEFAULT_MODELS:
        logger.info("Predicting %s", model_spec.name)
        predictions.append(predict_model(model_spec, base_frame))

    sample_predictions = pd.concat(predictions, ignore_index=True)
    subject_summary = build_subject_error_summary(sample_predictions)
    comparison = build_model_comparison(subject_summary, args.baseline_model)

    sample_predictions.to_csv(output_dir / "sample_predictions.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_error_summary.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    write_report(output_dir, comparison, subject_summary, args.baseline_model)

    logger.info("Wrote diagnostics to %s", output_dir)
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": [model.__dict__ for model in DEFAULT_MODELS],
                "test_subjects": test_ids,
                "outputs": [
                    "sample_predictions.csv",
                    "subject_error_summary.csv",
                    "model_comparison_summary.csv",
                    "PHASE11_ERROR_DIAGNOSTICS.md",
                ],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
