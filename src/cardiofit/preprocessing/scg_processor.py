"""SCG signal preprocessing.

Interface contract:
    process_scg(raw_signal, fs=800, axis_z=2) -> signal_125hz
"""

import logging
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(
    signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


def resample_signal(
    signal: np.ndarray, orig_fs: float, target_fs: float
) -> np.ndarray:
    """Resample with anti-aliasing filter for downsampling."""
    duration = len(signal) / orig_fs
    n_target = int(np.round(duration * target_fs))

    # Anti-aliasing lowpass before downsampling
    if target_fs < orig_fs:
        nyq_target = target_fs / 2.0
        b, a = butter(4, nyq_target / (orig_fs / 2), btype="low")
        signal = filtfilt(b, a, signal)

    t_orig = np.arange(len(signal)) / orig_fs
    t_target = np.linspace(0, duration, n_target, endpoint=False)
    interp = interp1d(t_orig, signal, kind="linear", bounds_error=False, fill_value="extrapolate")
    return interp(t_target)


def select_z_axis(raw_signal: np.ndarray, axis_z: int) -> np.ndarray:
    """Select z-axis from 3-axis SCG accelerometer data (dorso-ventral, primary axis)."""
    if raw_signal.ndim == 1:
        return raw_signal
    if raw_signal.shape[1] <= axis_z:
        logger.warning(f"axis_z={axis_z} out of range for {raw_signal.shape[1]} channels, using channel 0")
        return raw_signal[:, 0]
    return raw_signal[:, axis_z]


def minmax_normalize(signal: np.ndarray) -> np.ndarray:
    s_min = np.min(signal)
    s_max = np.max(signal)
    if s_max - s_min < 1e-8:
        return np.zeros_like(signal)
    return (signal - s_min) / (s_max - s_min)


def process_scg(
    raw_signal: np.ndarray,
    fs: float = 800.0,
    axis_z: int = 2,
    bp_low: float = 1.0,
    bp_high: float = 40.0,
    target_fs: float = 125.0,
) -> np.ndarray:
    """Process raw SCG signal.

    Pipeline: select Z axis → bandpass → resample (with anti-aliasing) → normalize.

    Args:
        raw_signal: (N,) or (N, 3) raw SCG accelerometer samples.
        fs: Original sampling rate in Hz.
        axis_z: Index of z-axis (0-based, default 2 for xyz layout).
        bp_low: Bandpass low cutoff in Hz.
        bp_high: Bandpass high cutoff in Hz.
        target_fs: Target sampling rate after resampling.

    Returns:
        signal_125hz: (M,) filtered and normalized SCG_z signal at target_fs.
    """
    logger.info(f"Processing SCG: {raw_signal.shape} at {fs} Hz")

    # 1. Select z-axis
    signal = select_z_axis(raw_signal, axis_z).astype(np.float64)

    # 2. Bandpass filter (1-40 Hz)
    signal = apply_bandpass_filter(signal, bp_low, bp_high, fs)

    # 3. Resample to target rate (with anti-aliasing for 800→125)
    signal_125hz = resample_signal(signal, fs, target_fs)

    # 4. Normalize to [0, 1]
    signal_125hz = minmax_normalize(signal_125hz)

    logger.info(f"SCG processed: {len(signal_125hz)} samples at {target_fs} Hz")
    return signal_125hz.astype(np.float32)
