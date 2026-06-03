# Phase 12 Bias Correction Evaluation

Run date: 2026-06-03

This is an inference-only evaluation. One scalar CO correction was fitted per model on validation predictions and applied to test predictions.

Formula:

```text
co_pred_corrected = co_pred - mean(validation_pred - validation_true)
```

Local detailed outputs are in `outputs/phase12_bias_correction/`:

- `validation_predictions.csv`
- `test_predictions_corrected.csv`
- `bias_correction_summary.csv`
- `PHASE12_BIAS_CORRECTION.md`

## Summary

| model | validation_bias | test_rmse_before | test_rmse_after | test_rmse_delta | test_mae_before | test_mae_after | test_mae_delta | test_bias_before | test_bias_after | test_r2_before | test_r2_after | test_pearson_before | test_pearson_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase9_baseline | -0.147536 | 1.018769 | 1.008770 | -0.009999 | 0.766889 | 0.730244 | -0.036645 | -0.142478 | 0.005058 | 0.534559 | 0.543651 | 0.738181 | 0.738180 |
| phase10_regularized | 0.085232 | 1.156465 | 1.182925 | 0.026460 | 0.764392 | 0.784355 | 0.019963 | -0.320511 | -0.405743 | 0.400240 | 0.372481 | 0.676928 | 0.676928 |
| phase10_low_lr | 0.086533 | 1.191393 | 1.236729 | 0.045336 | 0.898971 | 0.946266 | 0.047295 | -0.592799 | -0.679332 | 0.363465 | 0.314099 | 0.723548 | 0.723548 |
| phase10_no_aug | -0.033877 | 1.260549 | 1.244930 | -0.015619 | 0.981880 | 0.962489 | -0.019391 | -0.594511 | -0.560634 | 0.287423 | 0.304972 | 0.702522 | 0.702522 |

## Findings

- Best corrected test RMSE is `phase9_baseline` at 1.008770 L/min.
- Phase 9 baseline corrected RMSE improves by 0.009999 L/min and MAE improves by 0.036645 L/min.
- Scalar correction nearly removes Phase 9 global test bias (`-0.142478` to `0.005058`).
- Scalar correction does not help Phase 10 regularized or low-lr models; their validation bias direction does not match their test bias direction.
- Pearson r is unchanged by scalar correction; the gain is calibration-only.

## Next Adjustment

Keep Phase 9 baseline as the model checkpoint and apply validation-fitted scalar bias correction as a post-processing calibration candidate. For trained-model changes, prioritize subject-balanced loss or sampling rather than more capacity/dropout changes.
