# CardioFit Windows 5090D Handoff

Last updated: 2026-06-02

This file is the one-page handoff for continuing CardioFit training on the Windows 5090D machine.

## Current Project State

- Project: CardioFit, multimodal ECG + PPG + SCG prediction for CO and VO2max.
- GitHub repo: https://github.com/taelingk/biye
- Latest pushed branch to use on Windows: `main`
- Also available: `feat/core-pipeline`
- Task source of truth: `.tasks.yaml`
- Phase 1-8: done
- Phase 9: free, next task is real-data preprocessing and training on Windows 5090D.
- Phase 10: free, depends on Phase 9 results.

Note: `AI_CONTEXT.md` may still mention `feat/core-pipeline` as the active branch. As of this handoff, `main` contains the merged core pipeline and `.tasks.yaml` marks Phase 8 as done.

## Start Here On Windows

If the repo is not cloned yet:

```bash
git clone https://github.com/taelingk/biye
cd biye
git checkout main
```

If the repo already exists:

```bash
cd /path/to/biye
git fetch origin
git checkout main
git pull --ff-only
```

Check the workspace before changing anything:

```bash
git status --short --branch
```

Expected clean state:

```text
## main...origin/main
```

## Claim Phase 9 Before Training

Open `.tasks.yaml` and change `phase9-train` from:

```yaml
status: free
handler: ~
branch: ~
```

to:

```yaml
status: in_progress
handler: windows-5090d
branch: main
started: 2026-06-02
```

Then commit and push the claim:

```bash
git add .tasks.yaml
git commit -m "chore: claim phase 9 training"
git push
```

Do not claim Phase 10 yet. Phase 10 starts only after real training outputs exist.

## Environment Setup

Recommended: WSL2 with NVIDIA GPU available.

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-wsl.txt
```

Verify TensorFlow sees the GPU:

```bash
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
PY
```

Expected: at least one GPU device is listed.

## Data Layout Expected By The Code

The default config is `configs/default.yaml`.

Raw data path:

```text
data/raw/
```

Processed HDF5 output path:

```text
data/processed/
```

Each subject should be a folder under `data/raw/`:

```text
data/raw/
  subject_001/
    ecg.csv or ecg.npy
    ppg.csv or ppg.npy
    scg.csv or scg.npy
    co_labels.csv or co_labels.npy
    vo2_labels.csv or vo2_labels.npy
  subject_002/
    ...
```

Signal assumptions from `configs/default.yaml`:

- ECG raw sampling rate: 500 Hz
- PPG raw sampling rate: 100 Hz
- SCG raw sampling rate: 800 Hz
- Target sampling rate: 125 Hz
- Window: 125 samples, aligned to ECG R peaks
- Model input signal shape: `(125, 3)`
- Clinical input shape: `(10,)`

Optional clinical CSV:

```text
subject_id,age,gender,weight_kg,height_cm,hr_bpm,sbp,dbp
subject_001,30,1,70,175,70,120,80
```

The script reads the clinical CSV with the first column as index, so subject IDs must match folder names exactly.

## Run Phase 9

First run preprocessing.

Without clinical CSV:

```bash
python scripts/build_preprocessed_dataset.py --config configs/default.yaml
```

With clinical CSV:

```bash
python scripts/build_preprocessed_dataset.py --config configs/default.yaml --clinical data/clinical.csv
```

Single-subject smoke test:

```bash
python scripts/build_preprocessed_dataset.py --config configs/default.yaml --subject subject_001
```

Confirm HDF5 files were created:

```bash
find data/processed -maxdepth 1 -name "*.h5" | head
```

Then train:

```bash
python scripts/train_multimodal.py --config configs/default.yaml
```

For a shorter first GPU sanity run:

```bash
python scripts/train_multimodal.py --config configs/default.yaml --epochs 3
```

Expected training outputs:

```text
outputs/checkpoints/best_model.keras
outputs/checkpoints/final_model.keras
outputs/checkpoints/standardization_params.pkl
data/splits/
```

## Evaluate And Export

After training:

```bash
python scripts/evaluate_multimodal.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.keras
```

Expected evaluation outputs:

```text
outputs/evaluation/metrics.json
outputs/evaluation/co_scatter.png
outputs/evaluation/co_bland_altman.png
outputs/evaluation/co_error_dist.png
outputs/evaluation/vo2_scatter.png
outputs/evaluation/vo2_bland_altman.png
outputs/evaluation/vo2_error_dist.png
```

Optional ONNX export:

```bash
python scripts/export_onnx.py --checkpoint outputs/checkpoints/best_model.keras
```

Expected ONNX output:

```text
outputs/onnx/cardiofit_multimodal.onnx
```

## What To Send Back To Mac

Copy these back after Phase 9:

```text
outputs/checkpoints/
outputs/logs/
outputs/evaluation/
outputs/onnx/
data/splits/
```

Do not commit large raw data unless explicitly intended. Check `.gitignore` before adding files.

Useful summary to report back:

```text
number of subjects:
number of processed HDF5 files:
train/val/test split sizes:
GPU model:
TensorFlow version:
best val_loss:
CO metrics from metrics.json:
VO2 metrics from metrics.json:
checkpoint path:
```

## Mark Phase 9 Done

After successful preprocessing, training, evaluation, and copying outputs back, update `.tasks.yaml`:

```yaml
status: done
handler: windows-5090d
branch: main
completed: 2026-06-02
```

Add a short note, for example:

```yaml
note: Real-data training completed on Windows 5090D. Checkpoints and evaluation metrics copied back for Phase 10 tuning.
```

Commit and push:

```bash
git add .tasks.yaml
git commit -m "docs: mark phase 9 training complete"
git push
```

## Troubleshooting

If preprocessing says raw data directory is missing:

```text
Raw data directory not found: data/raw
```

Create `data/raw/` and put subject folders inside it.

If training says no HDF5 files were found:

```text
No HDF5 files found in data/processed
```

Run preprocessing first and confirm `.h5` files exist.

If GPU is not visible to TensorFlow, check WSL2 NVIDIA driver and CUDA support first:

```bash
nvidia-smi
python - <<'PY'
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
PY
```

If Git rejects push, pull first:

```bash
git pull --ff-only
git push
```

If `.tasks.yaml` shows another handler already set to `in_progress` for Phase 9, stop and do not overwrite it.
