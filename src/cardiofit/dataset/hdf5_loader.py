"""HDF5 data loader — build tf.data.Dataset from preprocessed HDF5 files.

Interface contract:
    load_subject(filepath) -> (signals (N,125,3), clinical (N,10), co (N,1), vo2 (N,1))
    build_tf_dataset(subject_ids, data_root, batch_size, shuffle, augment) -> tf.data.Dataset
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_subject(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a single subject's HDF5 file.

    Args:
        filepath: Path to subject_XXX.h5.

    Returns:
        signals: (N, 125, 3) float32
        clinical: (N, 10) float32
        co_targets: (N, 1) float32
        vo2_targets: (N, 1) float32
    """
    with h5py.File(filepath, "r") as f:
        signals = f["signal_segments"][:]
        clinical = f["clinical"][:]
        co = f["co_targets"][:]
        vo2 = f["vo2_targets"][:]
    return signals, clinical, co, vo2


def load_multiple_subjects(
    subject_ids: list[str], data_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate multiple subjects.

    Args:
        subject_ids: List of subject IDs (e.g., ["subject_001", "subject_002"]).
        data_root: Path to data/processed/ directory.

    Returns:
        Stacked arrays from all subjects.
    """
    all_s, all_c, all_co, all_vo2 = [], [], [], []
    for sid in subject_ids:
        filepath = data_root / f"{sid}.h5"
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}, skipping")
            continue
        s, c, co, vo2 = load_subject(filepath)
        all_s.append(s)
        all_c.append(c)
        all_co.append(co)
        all_vo2.append(vo2)

    if not all_s:
        raise FileNotFoundError(f"No HDF5 files found for subjects: {subject_ids}")

    return (
        np.concatenate(all_s, axis=0),
        np.concatenate(all_c, axis=0),
        np.concatenate(all_co, axis=0),
        np.concatenate(all_vo2, axis=0),
    )


def augment_segment(signal: np.ndarray, prob: float = 0.3, noise_ratio: float = 0.05) -> np.ndarray:
    """Apply time-shift and Gaussian noise augmentation (mirrors paper).

    Args:
        signal: (125, 3) single window.
        prob: Probability of applying each augmentation.
        noise_ratio: Std of noise relative to signal std.

    Returns:
        Augmented signal of same shape.
    """
    s = signal.copy()
    # Random time shift (±2 samples)
    if np.random.random() < prob:
        shift = np.random.randint(-2, 3)
        s = np.roll(s, shift, axis=0)
    # Gaussian noise
    if np.random.random() < prob:
        noise_std = noise_ratio * np.std(s, axis=0, keepdims=True)
        noise = np.random.normal(0, noise_std, size=s.shape)
        s = s + noise
    return s


def build_tf_dataset(
    subject_ids: list[str],
    data_root: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from HDF5 files.

    Args:
        subject_ids: List of subject IDs to include.
        data_root: Path to data/processed/.
        batch_size: Training batch size.
        shuffle: Whether to shuffle the dataset.
        augment: Whether to apply training-time augmentation.
        seed: Random seed for reproducibility.

    Returns:
        tf.data.Dataset yielding ((signals, clinical), (co, vo2)).
    """
    signals, clinical, co, vo2 = load_multiple_subjects(subject_ids, data_root)
    n_samples = len(signals)
    logger.info(f"Loaded {n_samples} samples from {len(subject_ids)} subjects")

    if augment:
        signals = np.array([augment_segment(x) for x in signals])

    ds = tf.data.Dataset.from_tensor_slices(
        ((signals, clinical), (co, vo2))
    )

    if shuffle:
        ds = ds.shuffle(min(n_samples, 10000), seed=seed)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def fit_standardization_params(
    subject_ids: list[str], data_root: Path
) -> dict:
    """Compute standardization parameters from training subjects.

    Signal: flattened to (N, 375) then StandardScaler.
    Clinical: per-feature (N, 10) StandardScaler.

    Returns:
        {"signal_mean": ..., "signal_std": ..., "clinical_mean": ..., "clinical_std": ...}
    """
    signals, clinical, _, _ = load_multiple_subjects(subject_ids, data_root)

    # Signal standardization: flatten (N, 125, 3) → (N, 375)
    flat = signals.reshape(signals.shape[0], -1)
    sig_mean = flat.mean(axis=0, keepdims=True)
    sig_std = flat.std(axis=0, keepdims=True) + 1e-8

    # Clinical per-feature standardization
    clin_mean = clinical.mean(axis=0, keepdims=True)
    clin_std = clinical.std(axis=0, keepdims=True) + 1e-8

    return {
        "signal_mean": sig_mean.astype(np.float32),
        "signal_std": sig_std.astype(np.float32),
        "clinical_mean": clin_mean.astype(np.float32),
        "clinical_std": clin_std.astype(np.float32),
    }
