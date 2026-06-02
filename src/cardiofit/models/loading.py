"""Utilities for loading trained multimodal checkpoints."""

from pathlib import Path

import joblib
import yaml
from tensorflow.keras.models import Model

from .resnet_se_lstm import build_multimodal_resnet_se_lstm, compile_model


def load_multimodal_checkpoint(
    checkpoint: str | Path,
    config: str | Path,
    standardization_params: str | Path | None = None,
    compile_for_training: bool = False,
) -> Model:
    """Rebuild the multimodal model from config and load checkpoint weights.

    Older ``.keras`` checkpoints may not deserialize directly because custom
    standardization layers require fitted scaler parameters. Training saves
    those parameters separately, so rebuilding the graph is the stable path for
    evaluation and export.
    """
    checkpoint_path = Path(checkpoint)
    config_path = Path(config)
    std_path = (
        Path(standardization_params)
        if standardization_params
        else checkpoint_path.parent / "standardization_params.pkl"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    std_params = joblib.load(std_path)
    model_cfg = cfg["model"]
    model = build_multimodal_resnet_se_lstm(
        input_shape=tuple(model_cfg["input_shape"]),
        n_clinical=model_cfg["n_clinical"],
        l2_reg=model_cfg["l2_reg"],
        dropout=model_cfg["dropout"],
        dropout_shared=model_cfg["dropout_shared"],
        dense_units=model_cfg["dense_units"],
        lstm_units=model_cfg["lstm_units"],
        se_reduction=model_cfg["se_reduction"],
        signal_mean=std_params["signal_mean"].ravel(),
        signal_scale=std_params["signal_std"].ravel(),
        clinical_mean=std_params["clinical_mean"].ravel(),
        clinical_scale=std_params["clinical_std"].ravel(),
    )
    model.load_weights(str(checkpoint_path))

    if compile_for_training:
        train_cfg = cfg["training"]
        compile_model(
            model,
            learning_rate=train_cfg["learning_rate"],
            clipvalue=train_cfg["clipvalue"],
            loss_weights=train_cfg.get("loss_weights"),
        )

    return model
