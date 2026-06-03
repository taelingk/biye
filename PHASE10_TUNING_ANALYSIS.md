# Phase 10 Tuning Analysis

Last updated: 2026-06-03

## Artifact Status

Phase 9 artifacts were downloaded from GitHub Release `phase9-artifacts-20260602`, SHA256 verified, and unpacked on Mac.

Verified files:

- `outputs/checkpoints/best_model.keras`
- `outputs/checkpoints/final_model.keras`
- `outputs/checkpoints/standardization_params.pkl`
- `outputs/checkpoints/training_history.csv`
- `outputs/evaluation/metrics.json`
- `outputs/evaluation/*.png`
- `outputs/onnx/scg_rhc_multimodal.onnx`
- `data/splits/subject_splits.json`

No retraining was run on Mac.

## Phase 9 Result Summary

Dataset split:

- Train subjects: 40
- Validation subjects: 9
- Test subjects: 8
- Test windows/samples: 14,488

CO test metrics:

- R2: 0.535
- Pearson r: 0.738
- MAE: 0.767 L/min
- RMSE: 1.019 L/min
- Bias: -0.142 L/min
- 95% limits of agreement: [-2.119, 1.834] L/min

VO2 metrics are not meaningful for this SCG-RHC run because `configs/scg_rhc_windows5090d.yaml` sets:

```yaml
loss_weights:
  co_output: 1.0
  vo2_output: 0.0
```

The imported SCG-RHC labels also use placeholder VO2 arrays, so Phase 10 should optimize CO only unless valid VO2 labels are added.

## Training Curve Diagnosis

Training ran for 48 epochs.

Best validation point:

- Best `val_loss`: 0.474 at epoch 27
- Best `val_co_output_loss`: 0.469 at epoch 27
- Best `val_co_output_mae`: 0.806 at epoch 27

Final epoch:

- Final train `co_output_loss`: 0.025
- Final train `co_output_mae`: 0.137
- Final `val_co_output_loss`: 0.618
- Final `val_co_output_mae`: 1.041

Interpretation:

- Generalization is the bottleneck. Training loss keeps improving after epoch 27, while validation loss worsens.
- Early stopping restored best weights, so `best_model.keras` should be the correct Phase 9 checkpoint.
- The gap between train MAE 0.137 and validation MAE around 0.806 at the best epoch suggests overfitting or subject-distribution mismatch.
- CO performance is promising enough for tuning; it is not a failed baseline.

## Recommended Phase 10 Experiments

Run these on Windows 5090D, one at a time. Each config writes to a separate output directory to avoid overwriting Phase 9 artifacts.

1. Regularized CO model

```bash
python scripts/train_multimodal.py --config configs/experiment/phase10_co_regularized.yaml
python scripts/evaluate_multimodal.py --config configs/experiment/phase10_co_regularized.yaml --checkpoint outputs/phase10_regularized/checkpoints/best_model.keras --output-dir outputs/phase10_regularized/evaluation
python scripts/export_onnx.py --config configs/experiment/phase10_co_regularized.yaml --checkpoint outputs/phase10_regularized/checkpoints/best_model.keras --output outputs/phase10_regularized/onnx/scg_rhc_multimodal.onnx
```

Purpose: reduce overfitting with stronger dropout, stronger L2, lower capacity, and shorter patience.

2. Low learning-rate CO model

```bash
python scripts/train_multimodal.py --config configs/experiment/phase10_co_low_lr.yaml
python scripts/evaluate_multimodal.py --config configs/experiment/phase10_co_low_lr.yaml --checkpoint outputs/phase10_low_lr/checkpoints/best_model.keras --output-dir outputs/phase10_low_lr/evaluation
python scripts/export_onnx.py --config configs/experiment/phase10_co_low_lr.yaml --checkpoint outputs/phase10_low_lr/checkpoints/best_model.keras --output outputs/phase10_low_lr/onnx/scg_rhc_multimodal.onnx
```

Purpose: test whether the validation instability is from too-large early updates.

3. No-augmentation control

```bash
python scripts/train_multimodal.py --config configs/experiment/phase10_co_no_aug.yaml
python scripts/evaluate_multimodal.py --config configs/experiment/phase10_co_no_aug.yaml --checkpoint outputs/phase10_no_aug/checkpoints/best_model.keras --output-dir outputs/phase10_no_aug/evaluation
python scripts/export_onnx.py --config configs/experiment/phase10_co_no_aug.yaml --checkpoint outputs/phase10_no_aug/checkpoints/best_model.keras --output outputs/phase10_no_aug/onnx/scg_rhc_multimodal.onnx
```

Purpose: test whether current time-shift/noise augmentation helps or hurts SCG-RHC generalization.

## Success Criteria

Primary metric:

- CO test RMSE below 1.019 L/min

Secondary metrics:

- CO MAE below 0.767 L/min
- CO R2 above 0.535
- CO Pearson r above 0.738
- Bias magnitude below 0.142 L/min

Treat VO2 metrics as diagnostic only until valid VO2 labels exist.

## Next Handoff Expected From Windows 5090D

After running the three Phase 10 configs, package artifacts again:

```bash
python scripts/package_phase9_artifacts.py pack --output phase10_artifacts.zip
```

The pack script now includes Phase 9 directories plus:

- `outputs/phase10_regularized/`
- `outputs/phase10_low_lr/`
- `outputs/phase10_no_aug/`

Report back:

```text
config name:
best epoch:
best val_loss:
CO metrics.json:
checkpoint path:
notes/errors:
```

## Windows 5090D Phase 10 Results

Run date: 2026-06-03

Artifacts were packaged with:

```bash
python scripts/package_phase9_artifacts.py pack --output phase10_artifacts.zip
```

Packaged file:

- `phase10_artifacts.zip`
- Size: 267,700,013 bytes
- Contains 58 files, including Phase 9 artifacts and all three `outputs/phase10_*` directories.

| Config | Best epoch | Best val_loss | CO R2 | CO Pearson r | CO MAE | CO RMSE | CO bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase10_co_regularized.yaml` | 19 | 0.601342 | 0.400239 | 0.676925 | 0.764394 | 1.156466 | -0.320461 |
| `phase10_co_low_lr.yaml` | 29 | 0.686003 | 0.363455 | 0.723552 | 0.898999 | 1.191402 | -0.592827 |
| `phase10_co_no_aug.yaml` | 5 | 0.557344 | 0.287423 | 0.702523 | 0.981881 | 1.260548 | -0.594512 |

Checkpoints:

- `outputs/phase10_regularized/checkpoints/best_model.keras`
- `outputs/phase10_low_lr/checkpoints/best_model.keras`
- `outputs/phase10_no_aug/checkpoints/best_model.keras`

Notes:

- All three experiments completed training, evaluation, and ONNX export verification.
- None of the Phase 10 runs improved the Phase 9 primary criterion, CO RMSE below 1.019 L/min.
- The regularized run slightly improved CO MAE versus Phase 9 (`0.764394` vs `0.767`), but underperformed Phase 9 on RMSE, R2, Pearson r, and bias.
- ONNX export for Phase 10 configs must pass the matching `--config` because model capacity differs from `configs/default.yaml`.
