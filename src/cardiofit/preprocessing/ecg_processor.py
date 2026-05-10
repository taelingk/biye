"""ECG signal preprocessing.

Interface contract:
    process_ecg(raw_signal, fs=500) -> (signal_125hz, r_peaks_125hz)
"""

import logging
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(
    signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


def apply_notch_filter(
    signal: np.ndarray, freq: float, fs: float, q: float = 30.0
) -> np.ndarray:
    """Apply IIR notch filter (zero-phase with filtfilt)."""
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, signal)


def baseline_correction(signal: np.ndarray, fs: float) -> np.ndarray:
    """Remove baseline wander using dual-window median filtering."""
    kernel_200ms = int(np.ceil(0.2 * fs))
    if kernel_200ms % 2 == 0:
        kernel_200ms += 1
    baseline_200 = medfilt(signal, kernel_size=kernel_200ms)

    kernel_600ms = int(np.ceil(0.6 * fs))
    if kernel_600ms % 2 == 0:
        kernel_600ms += 1
    baseline_600 = medfilt(baseline_200, kernel_size=kernel_600ms)

    return signal - baseline_600


def resample_signal(
    signal: np.ndarray, orig_fs: float, target_fs: float
) -> np.ndarray:
    """Resample signal to target sampling rate using linear interpolation with anti-aliasing."""
    duration = len(signal) / orig_fs
    n_target = int(np.round(duration * target_fs))
    t_orig = np.arange(len(signal)) / orig_fs
    t_target = np.linspace(0, duration, n_target, endpoint=False)

    # Apply anti-aliasing lowpass at Nyquist of target before downsampling
    if target_fs < orig_fs:
        nyq_target = target_fs / 2.0
        b, a = butter(4, nyq_target / (orig_fs / 2), btype="low")
        signal = filtfilt(b, a, signal)

    interp = interp1d(t_orig, signal, kind="linear", bounds_error=False, fill_value="extrapolate")
    return interp(t_target)


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks using Pan-Tompkins via neurokit2."""
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(signal, sampling_rate=int(fs))
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) == 0:
            logger.warning("No R-peaks found with neurokit2, falling back to simple peak detection")
            r_peaks = _simple_peak_detect(signal, fs)
        return r_peaks
    except Exception:
        logger.warning("neurokit2 unavailable, using simple peak detection")
        return _simple_peak_detect(signal, fs)


def _simple_peak_detect(signal: np.ndarray, fs: float) -> np.ndarray:
    """Simple peak detection fallback."""
    from scipy.signal import find_peaks
    min_dist = int(fs * 0.28)  # ~280ms minimum RR interval (214 bpm max)
    height = 0.5 * np.max(signal)
    peaks, _ = find_peaks(signal, distance=min_dist, height=height)
    return peaks


def minmax_normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to [0, 1]."""
    s_min = np.min(signal)
    s_max = np.max(signal)
    if s_max - s_min < 1e-8:
        return np.zeros_like(signal)
    return (signal - s_min) / (s_max - s_min)


def process_ecg(
    raw_signal: np.ndarray,
    fs: float = 500.0,
    bp_low: float = 0.5,
    bp_high: float = 40.0,
    notch_freq: float = 50.0,
    target_fs: float = 125.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Process raw ECG signal.

    Pipeline: bandpass → notch → baseline correction → resample → normalize → R-peak detection

    Args:
        raw_signal: (N,) raw ECG samples at fs Hz.
        fs: Original sampling rate in Hz.
        bp_low: Bandpass low cutoff in Hz.
        bp_high: Bandpass high cutoff in Hz.
        notch_freq: Notch filter frequency in Hz.
        target_fs: Target sampling rate after resampling.

    Returns:
        signal_125hz: (M,) filtered and normalized signal at target_fs.
        r_peaks_125hz: (K,) indices of R-peaks in the resampled signal.
    """
    logger.info(f"Processing ECG: {len(raw_signal)} samples at {fs} Hz")

    # 1. Bandpass filter (0.5-40 Hz)
    signal = apply_bandpass_filter(raw_signal.astype(np.float64), bp_low, bp_high, fs)
    logger.debug(f"Bandpass {bp_low}-{bp_high} Hz applied")

    # 2. Notch filter (50 Hz)
    if notch_freq > 0:
        signal = apply_notch_filter(signal, notch_freq, fs)
        logger.debug(f"Notch {notch_freq} Hz applied")

    # 3. Baseline correction
    signal = baseline_correction(signal, fs)

    # 4. Resample to target rate
    signal_125hz = resample_signal(signal, fs, target_fs)

    # 5. Normalize to [0, 1]
    signal_125hz = minmax_normalize(signal_125hz)

    # 6. Detect R-peaks on resampled signal
    r_peaks = detect_r_peaks(signal_125hz, target_fs)

    logger.info(f"ECG processed: {len(signal_125hz)} samples, {len(r_peaks)} R-peaks at {target_fs} Hz")
    return signal_125hz.astype(np.float32), r_peaks
