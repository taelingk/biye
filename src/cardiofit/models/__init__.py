"""Deep learning models for multi-modal cardiopulmonary assessment.

Core model: ResNet18-SE-LSTM with dual outputs (CO + VO2max).
Ported from Dong Xue's paper and adapted for ECG+PPG+SCG input.

Interface contracts:
    build_multimodal_resnet_se_lstm(...) -> keras.Model
    compile_model(model, lr=0.001, clipvalue=1.0) -> keras.Model
"""

from .resnet_se_lstm import build_multimodal_resnet_se_lstm, compile_model
from .se_block import ResidualSEBlock, SEBlock
from .standardize_layers import (
    Standardize1D,
    StandardizeClinical,
    StandardizeSignalFlat,
)

__all__ = [
    "build_multimodal_resnet_se_lstm",
    "compile_model",
    "SEBlock",
    "ResidualSEBlock",
    "Standardize1D",
    "StandardizeSignalFlat",
    "StandardizeClinical",
]
