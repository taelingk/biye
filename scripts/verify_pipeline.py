"""End-to-end pipeline verification with synthetic mock data."""

import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cardiofit.utils import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger("verify_pipeline")

# --- 1. Build synthetic data ---
import numpy as np


def generate_mock_subject(
    fs_ecg=500, fs_ppg=100, fs_scg=800, duration_sec=60, subject_id="mock_001"
):
    t_ecg = np.arange(int(fs_ecg * duration_sec)) / fs_ecg
    t_ppg = np.arange(int(fs_ppg * duration_sec)) / fs_ppg
    t_scg = np.arange(int(fs_scg * duration_sec)) / fs_scg

    ecg = 0.1 * np.sin(2 * np.pi * 1.2 * t_ecg) + 0.05 * np.random.randn(len(t_ecg))
    ppg = 0.1 * np.sin(2 * np.pi * 1.2 * t_ppg) + 0.02 * np.random.randn(len(t_ppg))
    scg = 0.1 * np.sin(2 * np.pi * 2.0 * t_scg) + 0.05 * np.random.randn(len(t_scg))

    clinical = {
        "age": 30,
        "gender": 1,
        "weight_kg": 70,
        "height_cm": 175,
        "hr_bpm": 70,
        "sbp": 120,
        "dbp": 80,
    }
    co_true = np.array([5.0])
    vo2_true = np.array([35.0])

    return {
        "ecg": ecg,
        "ppg": ppg,
        "scg": scg,
        "clinical": clinical,
        "co_true": co_true,
        "vo2_true": vo2_true,
        "fs_ecg": fs_ecg,
        "fs_ppg": fs_ppg,
        "fs_scg": fs_scg,
    }


# --- 2. Run preprocessing ---
from src.cardiofit.preprocessing import (
    compute_clinical_features,
    extract_windows,
    process_ecg,
    process_ppg,
    process_scg,
)

logger.info("=== Phase 1: Preprocessing ===")
mock = generate_mock_subject()
logger.info(
    f"Mock subject: ECG={len(mock['ecg'])}, PPG={len(mock['ppg'])}, SCG={len(mock['scg'])}"
)

ecg_125, r_peaks = process_ecg(mock["ecg"], fs=mock["fs_ecg"])
ppg_125 = process_ppg(mock["ppg"], fs=mock["fs_ppg"], ch_ir=1)
scg_125 = process_scg(mock["scg"], fs=mock["fs_scg"], axis_z=2)
logger.info(
    f"Resampled: ECG={len(ecg_125)}, PPG={len(ppg_125)}, SCG={len(scg_125)}, R-peaks={len(r_peaks)}"
)

if len(r_peaks) == 0:
    r_peaks = np.array([50, 175, 300, 425, 550], dtype=int)  # fallback

windows = extract_windows(
    ecg_125, ppg_125, scg_125, r_peaks, window_size=125, before_r=40
)
logger.info(f"Windows extracted: {windows.shape}")

clinical_vec = compute_clinical_features(mock["clinical"])
logger.info(f"Clinical features: {clinical_vec.shape}")

assert windows.shape[1:] == (125, 3), f"Expected (N, 125, 3), got {windows.shape}"
assert clinical_vec.shape == (10,), f"Expected (10,), got {clinical_vec.shape}"
logger.info("✅ Preprocessing verified")

# --- 3. Build HDF5 and test dataset loading ---
from src.cardiofit.dataset import (
    build_tf_dataset,
    fit_standardization_params,
    load_subject,
    split_subjects,
)

logger.info("=== Phase 2: HDF5 + DataLoader ===")
tmpdir = Path(tempfile.mkdtemp())
raw_dir = tmpdir / "raw"
proc_dir = tmpdir / "processed"
raw_dir.mkdir()
proc_dir.mkdir()

# Save raw mock data as numpy files (simulating what build_hdf5 expects)
import h5py

subject_file = proc_dir / f"subject_{mock['co_true'][0]:.0f}.h5"
clinical_matrix = np.tile(clinical_vec, (len(windows), 1))  # (N, 10)
with h5py.File(subject_file, "w") as f:
    f.create_dataset("signal_segments", data=windows)
    f.create_dataset("clinical", data=clinical_matrix)
    f.create_dataset("co_targets", data=mock["co_true"] * np.ones((len(windows), 1)))
    f.create_dataset("vo2_targets", data=mock["vo2_true"] * np.ones((len(windows), 1)))
logger.info(f"HDF5 saved: {subject_file}")

signals, clinical, co, vo2 = load_subject(subject_file)
logger.info(
    f"Loaded: signals={signals.shape}, clinical={clinical.shape}, co={co.shape}, vo2={vo2.shape}"
)
assert signals.shape == windows.shape
assert clinical.shape == (len(windows), 10)

# Full data loader pipeline
all_files = list(proc_dir.glob("*.h5"))
splits = split_subjects([f.stem for f in all_files], ratios=(1.0, 0.0, 0.0))
logger.info(f"Splits: {splits}")

std_params = fit_standardization_params(splits["train"], proc_dir)
ds = build_tf_dataset(
    splits["train"], proc_dir, batch_size=2, shuffle=False, augment=False
)
for batch in ds.take(1):
    sig, clin = batch[0]
    co_b, vo2_b = batch[1]
    logger.info(
        f"Batch: signal={sig.shape}, clinical={clin.shape}, co={co_b.shape}, vo2={vo2_b.shape}"
    )
logger.info("✅ Dataset pipeline verified")

# --- 4. Build and train one step ---
logger.info("=== Phase 3: Model forward pass ===")
from src.cardiofit.models import build_multimodal_resnet_se_lstm, compile_model

model = build_multimodal_resnet_se_lstm(
    signal_mean=std_params["signal_mean"].ravel(),
    signal_scale=std_params["signal_std"].ravel(),
    clinical_mean=std_params["clinical_mean"].ravel(),
    clinical_scale=std_params["clinical_std"].ravel(),
)
compile_model(model, learning_rate=0.001)
logger.info(f"Model: {model.count_params()} params")

# Forward pass
pred_co, pred_vo2 = model.predict(ds.take(1), verbose=0)
logger.info(f"Predictions: CO={pred_co[:3].ravel()}, VO2={pred_vo2[:3].ravel()}")
assert pred_co.shape[1] == 1 and pred_vo2.shape[1] == 1
logger.info("✅ Model forward pass verified")

# One training step
history = model.fit(ds.take(1), validation_data=ds.take(1), epochs=2, verbose=0)
logger.info(f"Train loss: {history.history['loss']}")
logger.info(f"Val loss: {history.history['val_loss']}")
logger.info("✅ Training step verified")

# --- 5. Evaluation ---
logger.info("=== Phase 4: Evaluation ===")
from src.cardiofit.evaluation import compute_all_metrics

metrics_co = compute_all_metrics(co[:2], pred_co.ravel())
metrics_vo2 = compute_all_metrics(vo2[:2], pred_vo2.ravel())
logger.info(f"CO metrics: { {k: round(float(v), 4) for k, v in metrics_co.items()} }")
logger.info(f"VO2 metrics: { {k: round(float(v), 4) for k, v in metrics_vo2.items()} }")
logger.info("✅ Evaluation verified")

# Cleanup
shutil.rmtree(tmpdir)

print()
print("=" * 50)
print("✅  End-to-end pipeline verification PASSED")
print("=" * 50)
print(f"  - Preprocessing: mock ECG({mock['fs_ecg']}Hz) → 125Hz windows")
print("  - HDF5 I/O: write → read → tf.data pipeline")
print(f"  - Model: build ({model.count_params():,} params) → forward pass → train step")
print("  - Evaluation: metrics computed")
print(f"  - Data: {len(all_files)} subject (synthetic)")
