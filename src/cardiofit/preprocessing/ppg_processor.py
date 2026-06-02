"""PPG signal preprocessing.

Interface contract:
    process_ppg(raw_signal, fs=100, ch_ir=1) -> signal_125hz
"""

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

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


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample signal to target sampling rate using linear interpolation."""
    duration = len(signal) / orig_fs
    n_target = int(np.round(duration * target_fs))
    t_orig = np.arange(len(signal)) / orig_fs
    t_target = np.linspace(0, duration, n_target, endpoint=False)
    interp = interp1d(
        t_orig, signal, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    return interp(t_target)


def select_ir_channel(raw_signal: np.ndarray, ch_ir: int) -> np.ndarray:
    """Select infrared channel from multi-channel PPG. Usually better SNR than red."""
    if raw_signal.ndim == 1:
        return raw_signal
    if raw_signal.shape[1] <= ch_ir:
        logger.warning(
            f"ch_ir={ch_ir} out of range for {raw_signal.shape[1]} channels, using channel 0"
        )
        return raw_signal[:, 0]
    return raw_signal[:, ch_ir]


def minmax_normalize(signal: np.ndarray) -> np.ndarray:
    s_min = np.min(signal)
    s_max = np.max(signal)
    if s_max - s_min < 1e-8:
        return np.zeros_like(signal)
    return (signal - s_min) / (s_max - s_min)


def process_ppg(
    raw_signal: np.ndarray,
    fs: float = 100.0,
    ch_ir: int = 1,
    bp_low: float = 0.5,
    bp_high: float = 8.0,
    target_fs: float = 125.0,
) -> np.ndarray:
    """Process raw PPG signal.

    Pipeline: select IR channel → bandpass → resample → normalize.

    Args:
        raw_signal: (N,) or (N, C) raw PPG samples.
        fs: Original sampling rate in Hz.
        ch_ir: Index of infrared channel (0-based).
        bp_low: Bandpass low cutoff in Hz.
        bp_high: Bandpass high cutoff in Hz.
        target_fs: Target sampling rate after resampling.

    Returns:
        signal_125hz: (M,) filtered and normalized PPG signal at target_fs.
    """
    logger.info(f"Processing PPG: {raw_signal.shape} at {fs} Hz")

    # 1. Select infrared channel
    signal = select_ir_channel(raw_signal, ch_ir).astype(np.float64)

    # 2. Bandpass filter (0.5-8 Hz)
    signal = apply_bandpass_filter(signal, bp_low, bp_high, fs)

    # 3. Resample to target rate
    signal_125hz = resample_signal(signal, fs, target_fs)

    # 4. Normalize to [0, 1]
    signal_125hz = minmax_normalize(signal_125hz)

    logger.info(f"PPG processed: {len(signal_125hz)} samples at {target_fs} Hz")
    return signal_125hz.astype(np.float32)
