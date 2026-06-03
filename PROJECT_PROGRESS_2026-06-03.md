# CardioFit Project Progress - 2026-06-03

## Today's Progress

### 1. Phase 10 CO Tuning Experiments

Pulled the latest `main` commit from Mac Codex:

- `235797a phase10: add tuning analysis and experiment configs`

Then ran the three requested Phase 10 CO experiment configurations on Windows 5090D:

- `configs/experiment/phase10_co_regularized.yaml`
- `configs/experiment/phase10_co_low_lr.yaml`
- `configs/experiment/phase10_co_no_aug.yaml`

Each run was evaluated and exported. Phase 9 was not rerun and Phase 9 status was not changed.

Phase 10 test metrics:

| Experiment | Best Epoch | Val Loss | CO R2 | Pearson | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase10_co_regularized | 19 | 0.601342 | 0.400239 | 0.676925 | 0.764394 | 1.156466 | -0.320461 |
| phase10_co_low_lr | 29 | 0.686003 | 0.363455 | 0.723552 | 0.898999 | 1.191402 | -0.592827 |
| phase10_co_no_aug | 5 | 0.557344 | 0.287423 | 0.702523 | 0.981881 | 1.260548 | -0.594512 |

Result: none of the Phase 10 variants beat the Phase 9 baseline by RMSE.

### 2. Phase 10 Artifact Packaging

Packed Phase 10 artifacts with:

```powershell
python scripts/package_phase9_artifacts.py pack --output phase10_artifacts.zip
```

Local artifact:

- `D:\gly\biye\biye-phase9-main\phase10_artifacts.zip`
- Size: `267,700,013` bytes
- Contents: 58 files

The large archive is ignored by Git so it does not enter the repository history.

### 3. Phase 10 Documentation And Status

Updated Phase 10 documentation after the Windows runs:

- Recorded the three experiment results in `PHASE10_TUNING_ANALYSIS.md`.
- Corrected ONNX export command examples to include `--config`.
- Marked Phase 10 as completed in `.tasks.yaml`.

Pushed commits:

- `1d846de docs: record phase 10 tuning results`
- `b36efd7 chore: ignore phase 10 artifact archive`

### 4. CI Maintenance

The repository also received CI maintenance commits today:

- `297a9eb ci: install TensorFlow test dependencies`
- `98b128c ci: use Node 24 actions and CPU TensorFlow`

Purpose: keep GitHub Actions compatible with the current workflow requirements and CPU test environment.

### 5. Phase 11 Error Diagnostics

Added a Phase 11 diagnostic workflow to compare Phase 9 and Phase 10 checkpoints without retraining.

New files:

- `scripts/diagnose_phase11_errors.py`
- `tests/test_phase11_diagnostics.py`
- `PHASE11_ERROR_DIAGNOSTICS.md`

Generated local diagnostic outputs:

- `outputs/phase11_diagnostics/sample_predictions.csv`
- `outputs/phase11_diagnostics/subject_error_summary.csv`
- `outputs/phase11_diagnostics/model_comparison_summary.csv`
- `outputs/phase11_diagnostics/PHASE11_ERROR_DIAGNOSTICS.md`
- `outputs/phase11_diagnostics/manifest.json`

Key finding:

- Phase 9 baseline remained the best RMSE model.
- `TRM267-RHC1` contributed the largest error mass across models.
- Phase 10 variants generally increased negative CO bias.

Phase 11 baseline comparison:

| Model | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| phase9_baseline | 1.018767 | 0.766891 | -0.142479 |

Pushed commit:

- `91beec9 analysis: add phase 11 error diagnostics`

### 6. Phase 12 Validation-Fitted CO Bias Correction

Added an inference-only scalar bias correction experiment.

Method:

```text
co_pred_corrected = co_pred - mean(validation_pred - validation_true)
```

New files:

- `scripts/evaluate_phase12_bias_correction.py`
- `tests/test_phase12_bias_correction.py`
- `PHASE12_BIAS_CORRECTION.md`

Generated local outputs:

- `outputs/phase12_bias_correction/validation_predictions.csv`
- `outputs/phase12_bias_correction/test_predictions_corrected.csv`
- `outputs/phase12_bias_correction/bias_correction_summary.csv`
- `outputs/phase12_bias_correction/PHASE12_BIAS_CORRECTION.md`

Best current candidate:

| Candidate | RMSE | MAE | Bias | R2 | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 9 baseline + validation-fitted bias correction | 1.008770 | 0.730244 | 0.005058 | 0.543651 | 0.738180 |

Interpretation:

- Bias correction improved the Phase 9 baseline.
- It reduced test bias from `-0.142478` to `0.005058`.
- It improved RMSE from `1.018769` to `1.008770`.
- It improved MAE from `0.766889` to `0.730244`.
- Phase 10 variants did not benefit consistently from the same correction.

Pushed commit:

- `f4c2754 analysis: evaluate phase 12 bias correction`

### 7. Verification

Executed the focused Phase 11 and Phase 12 test suite:

```powershell
pytest tests/test_phase11_diagnostics.py tests/test_phase12_bias_correction.py -q
```

Result:

```text
6 passed
```

## Overall Project Work Contents

### Phase 1 - Project Setup

Established the CardioFit repository structure, task tracking, dependency baseline, and initial project conventions.

### Phase 2 - Data Preprocessing Pipeline

Implemented preprocessing for real CardioFit signals and tabular metadata, including processed dataset generation.

### Phase 3 - Dataset Loader

Built dataset loading utilities for model training and evaluation.

### Phase 4 - Model Definition

Implemented the model architecture and reusable model construction code.

### Phase 5 - Training Pipeline

Added training scripts, experiment configuration support, checkpoint writing, and training metric outputs.

### Phase 6 - Evaluation Pipeline

Added evaluation scripts for checkpoint scoring and metric reporting.

### Phase 7 - ONNX Export And Documentation

Added ONNX export support and documented model export usage.

### Phase 8 - Runtime Verification

Verified the code path can run end to end in the target environment.

### Phase 9 - Real-Data Training

Ran the real-data CO training baseline and packaged Phase 9 artifacts. Phase 9 remains the strongest uncorrected model by RMSE.

### Phase 10 - CO Tuning

Ran three CO tuning experiments:

- regularized training
- lower learning rate
- no augmentation

Conclusion: Phase 10 variants did not outperform the Phase 9 baseline by RMSE.

### Phase 11 - Error Diagnostics

Added model comparison and subject-level diagnostics. The main actionable finding is that `TRM267-RHC1` dominates the residual error pattern.

### Phase 12 - Bias Correction

Added validation-fitted scalar CO bias correction. The current best candidate is Phase 9 baseline plus validation-fitted bias correction.

## Current Best Recommendation

Use the Phase 9 checkpoint as the base model and apply the Phase 12 validation-fitted scalar CO bias correction.

Current best test result:

| Metric | Value |
| --- | ---: |
| RMSE | 1.008770 |
| MAE | 0.730244 |
| Bias | 0.005058 |
| R2 | 0.543651 |
| Pearson | 0.738180 |

## Current Risks And Limitations

- VO2 remains outside the current valid target scope because available labels are placeholder-like and not reliable for training.
- `TRM267-RHC1` has a large negative residual pattern and should be inspected before further model tuning.
- Phase 10 tuning did not improve the best model, so the next improvement should target data balance and residual structure rather than simple hyperparameter changes.
- Large generated outputs and artifact archives are intentionally kept local and ignored by Git.

## Suggested Next Work

1. Add Phase 13 subject-balanced sampling or subject-weighted loss for CO training.
2. Inspect `TRM267-RHC1` label distribution, signal quality, and metadata alignment.
3. Keep Phase 9 as the production baseline until a subject-balanced model beats the Phase 12 corrected metrics.
4. Package Phase 12 corrected predictions and reports if a shareable artifact bundle is needed.

## Git Commits From Today

```text
f4c2754 analysis: evaluate phase 12 bias correction
98b128c ci: use Node 24 actions and CPU TensorFlow
91beec9 analysis: add phase 11 error diagnostics
297a9eb ci: install TensorFlow test dependencies
b36efd7 chore: ignore phase 10 artifact archive
1d846de docs: record phase 10 tuning results
235797a phase10: add tuning analysis and experiment configs
```
