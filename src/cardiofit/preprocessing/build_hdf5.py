"""Orchestrate preprocessing and write HDF5 per subject.

Interface contract:
    build_subject_hdf5(raw_dir, output_dir, sid, clinical) -> None
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import yaml

from .ecg_processor import process_ecg
from .ppg_processor import process_ppg
from .scg_processor import process_scg
from .sync_and_segment import extract_windows, compute_hr_from_r_peaks

logger = logging.getLogger(__name__)


def compute_clinical_features(
    clinical_raw: dict,
    hr: float | None = None,
) -> np.ndarray:
    """Compute derived clinical features.

    Args:
        clinical_raw: dict with keys: age, gender, weight_kg, height_cm, hr_bpm, sbp, dbp.
        hr: Optional override heart rate from R-peaks.

    Returns:
        features: (10,) array [age, gender, weight, height, BSA, BMI, HR, SBP, DBP, PP]
    """
    age = float(clinical_raw.get("age", 30))
    gender = float(clinical_raw["gender"])  # 1=male, 0=female
    weight = float(clinical_raw["weight_kg"])
    height = float(clinical_raw["height_cm"])

    # BSA: Du Bois formula
    bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)

    # BMI
    height_m = height / 100.0
    bmi = weight / (height_m * height_m)

    # HR
    if hr is None:
        hr = float(clinical_raw.get("hr_bpm", 70))

    # Blood pressure
    sbp = float(clinical_raw.get("sbp", 120))
    dbp = float(clinical_raw.get("dbp", 80))
    pp = sbp - dbp

    return np.array([age, gender, weight, height, bsa, bmi, hr, sbp, dbp, pp], dtype=np.float32)


def compute_co_target(sv: np.ndarray, hr: np.ndarray) -> np.ndarray:
    """CO (L/min) = SV (mL) * HR (bpm) / 1000"""
    return (sv * hr / 1000.0).astype(np.float32)


def build_subject_hdf5(
    raw_dir: Path,
    output_dir: Path,
    subject_id: str,
    clinical_raw: dict,
    config: dict,
) -> None:
    """Process a single subject's raw data and write processed HDF5.

    Expected raw files in raw_dir/subject_id/:
        ecg.csv (or .npy)
        ppg.csv (or .npy)
        scg.csv (or .npy)
        co_labels.csv (or .npy) — optional CO timeseries
        vo2_labels.csv (or .npy) — optional VO2 timeseries

    Args:
        raw_dir: Path to data/raw/.
        output_dir: Path to data/processed/.
        subject_id: Subject folder name (e.g., "subject_001").
        clinical_raw: dict with clinical parameters.
        config: Full YAML config dict.
    """
    subj_raw = raw_dir / subject_id
    sig_cfg = config["signal"]
    target_fs = sig_cfg["sampling_rate_target"]
    window_size = sig_cfg["window_size"]
    before_r = sig_cfg["window_before_r"]

    logger.info(f"Processing subject: {subject_id}")

    # Load raw signals
    ecg_raw = _load_signal(subj_raw / "ecg.csv", subj_raw / "ecg.npy")
    ppg_raw = _load_signal(subj_raw / "ppg.csv", subj_raw / "ppg.npy")
    scg_raw = _load_signal(subj_raw / "scg.csv", subj_raw / "scg.npy")

    # Preprocess each modality
    ecg_cfg = sig_cfg["ecg"]
    ecg_125, r_peaks = process_ecg(
        ecg_raw, fs=ecg_cfg["fs"],
        bp_low=ecg_cfg["bandpass_low"], bp_high=ecg_cfg["bandpass_high"],
        notch_freq=ecg_cfg["notch_freq"], target_fs=target_fs,
    )

    ppg_cfg = sig_cfg["ppg"]
    ppg_125 = process_ppg(
        ppg_raw, fs=ppg_cfg["fs"], ch_ir=ppg_cfg["channel_ir"],
        bp_low=ppg_cfg["bandpass_low"], bp_high=ppg_cfg["bandpass_high"],
        target_fs=target_fs,
    )

    scg_cfg = sig_cfg["scg"]
    scg_125 = process_scg(
        scg_raw, fs=scg_cfg["fs"], axis_z=scg_cfg["axis_z"],
        bp_low=scg_cfg["bandpass_low"], bp_high=scg_cfg["bandpass_high"],
        target_fs=target_fs,
    )

    # Extract windows
    segments = extract_windows(ecg_125, ppg_125, scg_125, r_peaks, window_size, before_r)
    n_windows = len(segments)

    if n_windows == 0:
        logger.error(f"No valid windows for subject {subject_id}, skipping")
        return

    # Clinical features (repeated for each window)
    hr_avg = compute_hr_from_r_peaks(r_peaks, target_fs)
    clinical_raw["hr_bpm"] = hr_avg
    clinical_vec = compute_clinical_features(clinical_raw)
    clinical_matrix = np.tile(clinical_vec, (n_windows, 1))  # (N, 10)

    # Load / interpolate labels
    co_targets, vo2_targets = _load_labels(subj_raw, n_windows)

    # Write HDF5
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{subject_id}.h5"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("signal_segments", data=segments, dtype="float32")
        f.create_dataset("clinical", data=clinical_matrix, dtype="float32")
        f.create_dataset("co_targets", data=co_targets, dtype="float32")
        f.create_dataset("vo2_targets", data=vo2_targets, dtype="float32")

        # Metadata
        f.attrs["subject_id"] = subject_id
        f.attrs["n_windows"] = n_windows
        f.attrs["window_size"] = window_size
        f.attrs["target_fs"] = target_fs
        f.attrs["channels"] = "ECG,PPG_IR,SCG_z"
        for key, val in clinical_raw.items():
            f.attrs[f"clinical_{key}"] = val

    logger.info(f"Written {h5_path}: {n_windows} windows")


def _load_signal(csv_path: Path, npy_path: Path) -> np.ndarray:
    """Load signal from CSV or NPY file."""
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    elif npy_path.exists():
        return np.load(str(npy_path))
    else:
        raise FileNotFoundError(f"No signal file found: {csv_path} or {npy_path}")


def _load_labels(subj_raw: Path, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    """Load CO and VO2 labels, interpolating timeseries to window count if needed."""
    co_file = subj_raw / "co_labels.csv" if (subj_raw / "co_labels.csv").exists() else subj_raw / "co_labels.npy"
    vo2_file = subj_raw / "vo2_labels.csv" if (subj_raw / "vo2_labels.csv").exists() else subj_raw / "vo2_labels.npy"

    if co_file.exists():
        co_data = _load_label_file(co_file)
    else:
        logger.warning(f"No CO labels for {subj_raw.name}, using zeros")
        co_data = np.zeros(n_windows, dtype=np.float32)

    if vo2_file.exists():
        vo2_data = _load_label_file(vo2_file)
    else:
        logger.warning(f"No VO2 labels for {subj_raw.name}, using zeros")
        vo2_data = np.zeros(n_windows, dtype=np.float32)

    # Interpolate or broadcast to window count if length mismatch
    co_data = _match_label_length(co_data, n_windows)
    vo2_data = _match_label_length(vo2_data, n_windows)

    return co_data.reshape(-1, 1).astype(np.float32), vo2_data.reshape(-1, 1).astype(np.float32)


def _match_label_length(data: np.ndarray, target_len: int) -> np.ndarray:
    """Broadcast scalar labels or interpolate time series to the window count."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 0:
        return np.full(target_len, float(data), dtype=np.float32)
    if len(data) != target_len:
        return _interp_to_length(data, target_len)
    return data


def _load_label_file(path: Path) -> np.ndarray:
    """Load label arrays from CSV text or NPY binary files."""
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32).squeeze()
    return np.loadtxt(str(path), delimiter=",", dtype=np.float32).squeeze()


def _interp_to_length(data: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation to match target length."""
    x_orig = np.linspace(0, 1, len(data))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_orig, data).astype(np.float32)
