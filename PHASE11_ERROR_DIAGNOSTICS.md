# Phase 11 Error Diagnostics

Run date: 2026-06-03

This analysis reuses existing Phase 9 and Phase 10 checkpoints for inference only. No training was run.

Local detailed outputs are in `outputs/phase11_diagnostics/`:

- `sample_predictions.csv`
- `subject_error_summary.csv`
- `model_comparison_summary.csv`
- `PHASE11_ERROR_DIAGNOSTICS.md`
- `manifest.json`

## Model Comparison

| model | n_subjects | n_samples | co_mae | co_rmse | co_bias | worst_subject_id | worst_subject_rmse | worst_subject_mae | co_rmse_delta_vs_baseline | co_mae_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase9_baseline | 8 | 14488 | 0.766891 | 1.018767 | -0.142479 | TRM267-RHC1 | 2.150266 | 2.150214 | 0.000000 | 0.000000 |
| phase10_regularized | 8 | 14488 | 0.764389 | 1.156464 | -0.320524 | TRM267-RHC1 | 2.763657 | 2.763579 | 0.137697 | -0.002502 |
| phase10_low_lr | 8 | 14488 | 0.898976 | 1.191395 | -0.592803 | TRM267-RHC1 | 2.693328 | 2.693289 | 0.172628 | 0.132085 |
| phase10_no_aug | 8 | 14488 | 0.981880 | 1.260548 | -0.594512 | TRM267-RHC1 | 2.595157 | 2.595157 | 0.241781 | 0.214989 |

## Worst Subject Contributions

| model | subject_id | n_samples | co_mae | co_rmse | co_bias | co_abs_error_sum | co_max_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase10_regularized | TRM267-RHC1 | 2081 | 2.763579 | 2.763657 | -2.763579 | 5751.008301 | 2.789074 |
| phase10_low_lr | TRM267-RHC1 | 2081 | 2.693289 | 2.693328 | -2.693289 | 5604.733398 | 2.722196 |
| phase10_no_aug | TRM267-RHC1 | 2081 | 2.595157 | 2.595157 | -2.595157 | 5400.521484 | 2.633533 |
| phase9_baseline | TRM267-RHC1 | 2081 | 2.150214 | 2.150266 | -2.150214 | 4474.596191 | 2.184209 |
| phase10_no_aug | TRM127-RHC1 | 2549 | 1.335098 | 1.335381 | -1.335098 | 3403.165039 | 1.782847 |
| phase9_baseline | TRM152-RHC1 | 3064 | 0.938272 | 0.938272 | 0.938272 | 2874.866211 | 0.938272 |
| phase10_no_aug | TRM152-RHC1 | 3064 | 0.905139 | 0.905139 | 0.905139 | 2773.345947 | 0.924635 |
| phase10_regularized | TRM152-RHC1 | 3064 | 0.774424 | 0.774429 | 0.774424 | 2372.834473 | 0.861356 |
| phase10_low_lr | TRM152-RHC1 | 3064 | 0.698725 | 0.698737 | 0.698725 | 2140.891846 | 0.786870 |
| phase10_low_lr | TRM127-RHC1 | 2549 | 0.648687 | 0.652950 | -0.648687 | 1653.504028 | 1.830020 |
| phase9_baseline | TRM159-RHC1 | 1588 | 0.802872 | 0.802877 | 0.802872 | 1274.960083 | 0.811075 |
| phase10_low_lr | TRM208-RHC1 | 1172 | 0.951239 | 0.971222 | -0.921940 | 1114.852539 | 4.579108 |

## Findings

- Best test RMSE remains `phase9_baseline` at 1.018767 L/min.
- Phase 10 regularization slightly improved global MAE by 0.002502 L/min but worsened RMSE by 0.137697 L/min and shifted bias from -0.142479 to -0.320524 L/min.
- `TRM267-RHC1` dominates absolute error mass for every model and has a large negative bias in all runs.
- Phase 10 variants generally improved some positive-bias subjects such as `TRM152-RHC1`, but they paid for it by increasing negative bias on `TRM267-RHC1` and other subjects.
- The next tuning round should target subject-level bias and distribution mismatch before adding more capacity changes.

## Phase 12 Recommendations

1. Add a bias-correction experiment that fits a scalar correction on validation predictions only, then evaluates on test predictions.
2. Add subject-balanced training or loss weighting so subjects with many windows do not dominate optimization.
3. Add diagnostics for label distribution and clinical feature distribution by split, especially for `TRM267-RHC1`.
4. Keep Phase 9 as the deployment baseline until a candidate beats its test RMSE and bias simultaneously.
