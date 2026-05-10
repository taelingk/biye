"""Built-in standardization layers for ONNX-exportable models.

These embed the fitted sklearn StandardScaler parameters directly into the Keras graph,
so the exported ONNX is self-contained (no need for external scaler at inference time).

Reference: ~/code/codongxue/src/training/train_svco_model.py (lines 438-484)

Our changes from paper:
    - StandardizeClinical: unified (None, 10) input instead of 11 separate (None, 1) inputs.
    - StandardizeSignalFlat: unchanged except channels=3 (was also 3 in paper: PPG+1st+2nd).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Standardize1D(Layer):
    """Standardize a scalar feature: z = (x - mean) / (scale + eps).

    Input shape: (None, 1) or (None,)
    Output shape: (None, 1)
    """

    def __init__(self, mean, scale, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1)
        self.scale = np.array(scale, dtype=np.float32).reshape(1, 1)
        self.eps = float(eps)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=-1)
        return (x - self.mean) / (self.scale + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "mean": self.mean.reshape(-1).tolist(),
            "scale": self.scale.reshape(-1).tolist(),
            "eps": self.eps,
        })
        return cfg


class StandardizeSignalFlat(Layer):
    """Standardize (None, window_size, channels) by flattening to (None, window_size*channels).

    Exactly mirrors sklearn StandardScaler(is_2d=True):
      reshape (B, T, C) → (B, T*C), z-score per dimension, reshape back.

    Our window_size=125, channels=3 → flattens to (B, 375).
    """

    def __init__(self, mean, scale, window_size: int = 125, channels: int = 3, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, -1)    # (1, 375)
        self.scale = np.array(scale, dtype=np.float32).reshape(1, -1)  # (1, 375)
        self.window_size = int(window_size)
        self.channels = int(channels)
        self.eps = float(eps)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        b = tf.shape(x)[0]
        x2 = tf.reshape(x, (b, -1))  # (B, 375)
        x2 = (x2 - self.mean) / (self.scale + self.eps)
        return tf.reshape(x2, (b, self.window_size, self.channels))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "window_size": self.window_size,
            "channels": self.channels,
            "eps": self.eps,
        })
        return cfg


class StandardizeClinical(Layer):
    """Standardize (None, n_features) clinical matrix — unified version for our project.

    This replaces the paper's 11 separate Standardize1D layers with one layer that handles
    all 10 clinical features at once.

    Input: (None, 10)
    Output: (None, 10)
    """

    def __init__(self, mean, scale, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, -1)  # (1, 10)
        self.scale = np.array(scale, dtype=np.float32).reshape(1, -1)  # (1, 10)
        self.eps = float(eps)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        return (x - self.mean) / (self.scale + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "mean": self.mean.reshape(-1).tolist(),
            "scale": self.scale.reshape(-1).tolist(),
            "eps": self.eps,
        })
        return cfg
