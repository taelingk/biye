import numpy as np
import pandas as pd

from scripts.evaluate_phase12_bias_correction import (
    apply_bias_correction,
    build_bias_correction_summary,
    compute_validation_bias,
)


def test_compute_validation_bias_uses_pred_minus_true_mean():
    validation_predictions = pd.DataFrame(
        {
            "model": ["baseline", "baseline", "candidate", "candidate"],
            "co_true": [5.0, 7.0, 5.0, 7.0],
            "co_pred": [4.0, 6.5, 6.0, 8.0],
        }
    )

    corrections = compute_validation_bias(validation_predictions)

    assert corrections == {"baseline": -0.75, "candidate": 1.0}


def test_apply_bias_correction_subtracts_validation_bias_per_model():
    test_predictions = pd.DataFrame(
        {
            "model": ["baseline", "baseline", "candidate"],
            "co_true": [5.0, 7.0, 10.0],
            "co_pred": [4.0, 6.5, 12.0],
        }
    )

    corrected = apply_bias_correction(
        test_predictions, {"baseline": -0.75, "candidate": 1.0}
    )

    assert corrected["co_pred_corrected"].tolist() == [4.75, 7.25, 11.0]
    assert corrected["co_error_corrected"].tolist() == [-0.25, 0.25, 1.0]


def test_build_bias_correction_summary_reports_before_after_metrics():
    validation_predictions = pd.DataFrame(
        {
            "model": ["baseline", "baseline", "candidate", "candidate"],
            "co_true": [5.0, 7.0, 5.0, 7.0],
            "co_pred": [4.0, 6.5, 6.0, 8.0],
        }
    )
    test_predictions = pd.DataFrame(
        {
            "model": ["baseline", "baseline", "candidate", "candidate"],
            "subject_id": ["s1", "s2", "s1", "s2"],
            "co_true": [5.0, 7.0, 5.0, 7.0],
            "co_pred": [4.0, 6.5, 6.0, 8.0],
        }
    )

    summary, corrected = build_bias_correction_summary(
        validation_predictions, test_predictions
    )

    baseline = summary[summary["model"] == "baseline"].iloc[0]
    assert baseline["validation_bias"] == -0.75
    assert np.isclose(baseline["test_bias_before"], -0.75)
    assert np.isclose(baseline["test_bias_after"], 0.0)
    assert baseline["test_rmse_after"] < baseline["test_rmse_before"]
    assert "co_pred_corrected" in corrected.columns
