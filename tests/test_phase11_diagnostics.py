import numpy as np
import pandas as pd

from scripts.diagnose_phase11_errors import (
    build_model_comparison,
    build_subject_error_summary,
    dataframe_to_markdown,
)


def test_build_subject_error_summary_groups_metrics_by_model_and_subject():
    predictions = pd.DataFrame(
        {
            "model": ["baseline", "baseline", "baseline", "candidate"],
            "subject_id": ["s1", "s1", "s2", "s1"],
            "co_true": [1.0, 3.0, 4.0, 1.0],
            "co_pred": [2.0, 1.0, 5.0, 1.5],
        }
    )

    summary = build_subject_error_summary(predictions)

    row = summary[
        (summary["model"] == "baseline") & (summary["subject_id"] == "s1")
    ].iloc[0]
    assert row["n_samples"] == 2
    assert row["co_mae"] == 1.5
    assert np.isclose(row["co_rmse"], np.sqrt(2.5))
    assert row["co_bias"] == -0.5
    assert row["co_abs_error_sum"] == 3.0

    worst = summary.iloc[0]
    assert worst["model"] == "baseline"
    assert worst["subject_id"] == "s1"


def test_build_model_comparison_marks_baseline_deltas():
    subject_summary = pd.DataFrame(
        {
            "model": ["baseline", "baseline", "candidate", "candidate"],
            "subject_id": ["s1", "s2", "s1", "s2"],
            "n_samples": [2, 1, 2, 1],
            "co_mae": [1.5, 1.0, 0.5, 2.0],
            "co_rmse": [1.6, 1.0, 0.5, 2.0],
            "co_bias": [-0.5, 1.0, 0.5, -2.0],
            "co_abs_error_sum": [3.0, 1.0, 1.0, 2.0],
        }
    )

    comparison = build_model_comparison(subject_summary, baseline_model="baseline")

    candidate = comparison[comparison["model"] == "candidate"].iloc[0]
    baseline = comparison[comparison["model"] == "baseline"].iloc[0]
    assert baseline["co_rmse_delta_vs_baseline"] == 0.0
    assert np.isclose(candidate["co_rmse"], 1.224744871391589)
    assert np.isclose(candidate["co_rmse_delta_vs_baseline"], -0.20354081431698123)
    assert candidate["worst_subject_id"] == "s2"


def test_dataframe_to_markdown_renders_without_optional_dependencies():
    table = dataframe_to_markdown(
        pd.DataFrame({"model": ["baseline"], "co_rmse": [1.23456789]})
    )

    assert "| model | co_rmse |" in table
    assert "| baseline | 1.234568 |" in table
