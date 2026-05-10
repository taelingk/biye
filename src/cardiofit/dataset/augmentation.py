"""Training-time data augmentation for physiological signals.

Mirrors the paper's augmentation strategy:
- Random time shift (±2 samples)
- Gaussian noise (5% std)
"""

import numpy as np


def time_shift(signal: np.ndarray, max_shift: int = 2) -> np.ndarray:
    """Randomly shift signal along time axis.

    Args:
        signal: (window_size, channels) or (batch, window_size, channels)
        max_shift: Maximum shift in samples.

    Returns:
        Shifted signal of same shape.
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return signal
    return np.roll(signal, shift, axis=-2)


def gaussian_noise(signal: np.ndarray, std_ratio: float = 0.05) -> np.ndarray:
    """Add Gaussian noise proportional to signal std.

    Args:
        signal: (..., channels)
        std_ratio: Noise std as fraction of per-channel signal std.

    Returns:
        Noisy signal of same shape.
    """
    noise_std = std_ratio * np.std(signal, axis=-2, keepdims=True)
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise


def augment_segment_tf(signal: np.ndarray, shift_prob: float = 0.3, noise_prob: float = 0.3,
                        max_shift: int = 2, noise_ratio: float = 0.05) -> np.ndarray:
    """Apply augmentation to a single segment (NumPy, for use in tf.data pipeline).

    Args:
        signal: (125, 3) float32 array.
        shift_prob: Probability of applying time shift.
        noise_prob: Probability of applying noise.
        max_shift: Maximum shift in samples.
        noise_ratio: Noise std relative to signal std.

    Returns:
        Augmented signal.
    """
    s = signal.copy()
    if np.random.random() < shift_prob:
        s = time_shift(s, max_shift)
    if np.random.random() < noise_prob:
        s = gaussian_noise(s, noise_ratio)
    return s


def tf_augment_wrapper(signal, clinical, co, vo2):
    """TensorFlow py_function wrapper for augmentation.

    Usage:
        ds = ds.map(lambda s, c, co, vo2: tf.py_function(
            tf_augment_wrapper, [s, c, co, vo2],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
        ))
    """
    import tensorflow as tf

    def _augment(sig, clin, c, v):
        sig_np = sig.numpy()
        aug = augment_segment_tf(sig_np)
        return aug.astype(np.float32), clin.numpy(), c.numpy(), v.numpy()

    signal_aug, clinical_out, co_out, vo2_out = tf.py_function(
        _augment,
        [signal, clinical, co, vo2],
        Tout=[tf.float32, tf.float32, tf.float32, tf.float32],
    )
    signal_aug.set_shape(signal.shape)
    clinical_out.set_shape(clinical.shape)
    co_out.set_shape(co.shape)
    vo2_out.set_shape(vo2.shape)
    return (signal_aug, clinical_out), (co_out, vo2_out)
