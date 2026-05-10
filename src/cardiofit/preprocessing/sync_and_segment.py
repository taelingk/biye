"""Multi-modal sync and window extraction.

Interface contract:
    extract_windows(ecg, ppg, scg, r_peaks, window_size=125, before_r=40) -> (N_windows, 125, 3)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def ensure_same_length(
    ecg: np.ndarray, ppg: np.ndarray, scg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncate all signals to the shortest length among them."""
    min_len = min(len(ecg), len(ppg), len(scg))
    if len(ecg) != min_len or len(ppg) != min_len or len(scg) != min_len:
        logger.info(f"Truncating signals to common length: {min_len}")
    return ecg[:min_len], ppg[:min_len], scg[:min_len]


def extract_windows(
    ecg_125: np.ndarray,
    ppg_125: np.ndarray,
    scg_125: np.ndarray,
    r_peaks: np.ndarray,
    window_size: int = 125,
    before_r: int = 40,
) -> np.ndarray:
    """Extract fixed-length windows centered around R-peaks.

    Each window is stacked as [ECG, PPG, SCG_z] → shape (window_size, 3).

    Args:
        ecg_125: (T,) ECG signal at 125 Hz.
        ppg_125: (T,) PPG signal at 125 Hz.
        scg_125: (T,) SCG_z signal at 125 Hz.
        r_peaks: (K,) indices of R-peaks in the 125 Hz signals.
        window_size: Number of samples per window (125 = 1 second).
        before_r: Samples before R-peak in the window (40 ≈ 320 ms for P-wave).

    Returns:
        segments: (N_valid, window_size, 3) stacked signal windows.
    """
    ecg, ppg, scg = ensure_same_length(ecg_125, ppg_125, scg_125)
    after_r = window_size - before_r

    segments = []
    for r_idx in r_peaks:
        start = r_idx - before_r
        end = r_idx + after_r
        if start < 0 or end > len(ecg):
            continue  # Skip edge windows

        ecg_seg = ecg[start:end]
        ppg_seg = ppg[start:end]
        scg_seg = scg[start:end]

        # Stack as 3 channels: (125, 3)
        stacked = np.stack([ecg_seg, ppg_seg, scg_seg], axis=-1)
        segments.append(stacked)

    if len(segments) == 0:
        logger.warning("No valid windows extracted. Check R-peak positions and signal length.")
        return np.empty((0, window_size, 3), dtype=np.float32)

    result = np.array(segments, dtype=np.float32)
    logger.info(f"Extracted {len(result)} windows of shape {result.shape[1:]}")
    return result


def compute_hr_from_r_peaks(r_peaks: np.ndarray, fs: float = 125.0) -> float:
    """Compute average heart rate from R-peak intervals.

    Args:
        r_peaks: (K,) indices of R-peaks.
        fs: Sampling rate in Hz.

    Returns:
        hr: Mean heart rate in bpm.
    """
    if len(r_peaks) < 2:
        return 70.0  # fallback default
    rr_intervals = np.diff(r_peaks) / fs
    hr = 60.0 / np.mean(rr_intervals)
    return float(np.clip(hr, 30, 220))
