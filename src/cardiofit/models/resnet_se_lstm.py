"""ResNet18-SE-LSTM multimodal model with dual outputs (CO + VO2max).

Ported from Dong Xue's paper implementation with these key changes:
  1. Input: (125, 3) for [ECG, PPG_IR, SCG_z] instead of [PPG, PPG', PPG'']
  2. Clinical: unified (None, 10) input instead of 11 separate (None, 1) inputs
  3. Output: dual heads (co_output, vo2_output) instead of single SV output

Reference: ~/code/codongxue/src/training/train_svco_model.py
"""

import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    Dense, Dropout, LSTM, Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from .se_block import ResidualSEBlock
from .standardize_layers import StandardizeSignalFlat, StandardizeClinical

logger = logging.getLogger(__name__)


def build_multimodal_resnet_se_lstm(
    input_shape: tuple[int, int] = (125, 3),
    n_clinical: int = 10,
    l2_reg: float = 4.11e-05,
    dropout: float = 0.3,
    dropout_shared: float = 0.2,
    dense_units: int = 128,
    lstm_units: int = 64,
    se_reduction: int = 8,
    signal_mean: Optional[np.ndarray] = None,
    signal_scale: Optional[np.ndarray] = None,
    clinical_mean: Optional[np.ndarray] = None,
    clinical_scale: Optional[np.ndarray] = None,
) -> Model:
    """Build ResNet18-SE-LSTM model with built-in standardization and dual outputs.

    Architecture:
        Signal (None, 125, 3)
          → StandardizeSignalFlat
          → Conv1D(64, 7, s=2) → BN → ReLU
          → 4 groups of ResidualSEBlocks (64→128→256→512)
          → LSTM(64)
          → Dense(128) + Dropout

        Clinical (None, 10)
          → StandardizeClinical
          → Concatenate with signal features → (None, 138)
          → Dense(64, relu) + Dropout
          → ├─ Dense(1, name='co_output')
             └─ Dense(1, name='vo2_output')

    Args:
        input_shape: (window_size, channels). Default (125, 3).
        n_clinical: Number of clinical features. Default 10.
        l2_reg: L2 regularization coefficient.
        dropout: Dropout rate after Dense layer.
        dropout_shared: Dropout rate after shared Dense layer.
        dense_units: Units in the feature Dense layer.
        lstm_units: Units in the LSTM layer.
        se_reduction: Reduction ratio in SE blocks.
        signal_mean: (375,) mean for signal standardization (from sklearn StandardScaler).
        signal_scale: (375,) scale for signal standardization.
        clinical_mean: (10,) mean for clinical standardization.
        clinical_scale: (10,) scale for clinical standardization.

    Returns:
        tf.keras.Model with inputs=(signal_input, clinical_input)
        and outputs=(co_output, vo2_output).
    """
    window_size = input_shape[0]
    channels = input_shape[1]
    signal_flat_dim = window_size * channels  # 375

    regularizer = l2(l2_reg) if l2_reg > 0 else None

    # --- Inputs ---
    signal_input = Input(shape=input_shape, name="signal_input", dtype=tf.float32)
    clinical_input = Input(shape=(n_clinical,), name="clinical_input", dtype=tf.float32)

    # --- Built-in Standardization ---
    sig_mean = signal_mean if signal_mean is not None else np.zeros(signal_flat_dim, dtype=np.float32)
    sig_scale = signal_scale if signal_scale is not None else np.ones(signal_flat_dim, dtype=np.float32)
    x = StandardizeSignalFlat(sig_mean, sig_scale, window_size, channels, name="signal_standardize")(signal_input)

    clin_mean = clinical_mean if clinical_mean is not None else np.zeros(n_clinical, dtype=np.float32)
    clin_scale = clinical_scale if clinical_scale is not None else np.ones(n_clinical, dtype=np.float32)
    clin_std = StandardizeClinical(clin_mean, clin_scale, name="clinical_standardize")(clinical_input)

    # --- ResNet18 Backbone ---
    # Initial conv
    x = Conv1D(64, 7, strides=2, padding="same", kernel_regularizer=regularizer, name="conv1")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="act1")(x)

    # Group 1: 64 filters, stride 1, 2 blocks
    for i in range(2):
        x = ResidualSEBlock(64, kernel_regularizer=regularizer, name=f"res_se_block_64_{i}")(x)

    # Group 2: 128 filters, stride 2, 2 blocks
    x = ResidualSEBlock(128, stride=2, use_1x1_conv=True, kernel_regularizer=regularizer, name="res_se_block_128_0")(x)
    for i in range(1):
        x = ResidualSEBlock(128, kernel_regularizer=regularizer, name=f"res_se_block_128_{i+1}")(x)

    # Group 3: 256 filters, stride 2, 2 blocks
    x = ResidualSEBlock(256, stride=2, use_1x1_conv=True, kernel_regularizer=regularizer, name="res_se_block_256_0")(x)
    for i in range(1):
        x = ResidualSEBlock(256, kernel_regularizer=regularizer, name=f"res_se_block_256_{i+1}")(x)

    # Group 4: 512 filters, stride 1, 1 block
    x = ResidualSEBlock(512, stride=1, use_1x1_conv=True, kernel_regularizer=regularizer, name="res_se_block_512")(x)

    # --- LSTM ---
    x = LSTM(lstm_units, return_sequences=False, kernel_regularizer=regularizer, name="lstm")(x)

    # --- Feature Dense ---
    x = Dense(dense_units, activation="relu", kernel_regularizer=regularizer, name="dense_features")(x)
    x = Dropout(dropout, name="dropout_features")(x)

    # --- Concatenate Clinical Features ---
    x = Concatenate(name="concat_clinical")([x, clin_std])  # (None, dense_units + 10) = (None, 138)

    # --- Shared Dense ---
    x = Dense(64, activation="relu", kernel_regularizer=regularizer, name="dense_shared")(x)
    x = Dropout(dropout_shared, name="dropout_shared")(x)

    # --- Dual Output Heads ---
    co_output = Dense(1, name="co_output")(x)
    vo2_output = Dense(1, name="vo2_output")(x)

    model = Model(
        inputs=[signal_input, clinical_input],
        outputs=[co_output, vo2_output],
        name="resnet18_se_lstm_multimodal",
    )

    return model


def compile_model(
    model: Model,
    learning_rate: float = 0.001,
    clipvalue: float = 1.0,
    loss_weights: Optional[dict[str, float]] = None,
) -> Model:
    """Compile the model with paper-consistent settings.

    Dual Huber losses for CO and VO2max, Adam optimizer with gradient clipping.
    """
    resolved_loss_weights = loss_weights or {
        "co_output": 1.0,
        "vo2_output": 1.0,
    }
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipvalue=clipvalue,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
        ),
        loss={
            "co_output": tf.keras.losses.Huber(delta=1.0),
            "vo2_output": tf.keras.losses.Huber(delta=1.0),
        },
        loss_weights=resolved_loss_weights,
        metrics={
            "co_output": ["mae"],
            "vo2_output": ["mae"],
        },
    )
    return model
