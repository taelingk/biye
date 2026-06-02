#!/usr/bin/env python3
"""Train the multimodal ResNet18-SE-LSTM model for CO + VO2max prediction.

Usage:
    python scripts/train_multimodal.py
    python scripts/train_multimodal.py --config configs/default.yaml
    python scripts/train_multimodal.py --config configs/default.yaml --epochs 200
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cardiofit.models import build_multimodal_resnet_se_lstm, compile_model
from src.cardiofit.dataset import (
    build_tf_dataset,
    fit_standardization_params,
    split_subjects,
    save_splits,
)
from src.cardiofit.training.callbacks import get_callbacks
from src.cardiofit.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train multimodal CO+VO2max model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging()

    project_root = Path.cwd()
    data_root = project_root / cfg["paths"]["processed_data"]
    splits_dir = project_root / cfg["paths"]["splits"]
    checkpoint_dir = project_root / cfg["paths"]["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["project"]["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # --- 1. Discover subjects ---
    subject_files = sorted(data_root.glob("*.h5"))
    if not subject_files:
        logger.error(f"No HDF5 files found in {data_root}. Run build_preprocessed_dataset.py first.")
        sys.exit(1)
    subject_ids = [f.stem for f in subject_files]
    logger.info(f"Found {len(subject_ids)} subjects: {subject_ids[:5]}...")

    # --- 2. Split subjects ---
    splits = split_subjects(subject_ids, ratios=cfg["data_split"]["ratios"], seed=seed)
    save_splits(splits, splits_dir)
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]
    logger.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # --- 3. Fit standardization params (training set only) ---
    std_params = fit_standardization_params(train_ids, data_root)

    # --- 4. Build datasets ---
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]

    train_ds = build_tf_dataset(
        train_ids, data_root, batch_size=batch_size,
        shuffle=True, augment=True, seed=seed,
    )
    val_ds = build_tf_dataset(
        val_ids, data_root, batch_size=batch_size,
        shuffle=False, augment=False,
    )
    logger.info(f"Datasets ready: train={train_ds.cardinality().numpy()} batches, val={val_ds.cardinality().numpy()} batches")

    # --- 5. Build model ---
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

    compile_model(
        model,
        learning_rate=train_cfg["learning_rate"],
        clipvalue=train_cfg["clipvalue"],
        loss_weights=train_cfg.get("loss_weights"),
    )

    if args.checkpoint:
        logger.info(f"Loading pre-trained weights from {args.checkpoint}")
        model.load_weights(args.checkpoint)

    model.summary()

    # --- 6. Train ---
    epochs = args.epochs or train_cfg["max_epochs"]
    callbacks = get_callbacks(
        checkpoint_dir=str(checkpoint_dir),
        patience=train_cfg["early_stopping_patience"],
        initial_lr=train_cfg["learning_rate"],
        total_epochs=epochs,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # --- 7. Save final model ---
    final_path = checkpoint_dir / "final_model.keras"
    model.save(str(final_path))
    logger.info(f"Model saved to {final_path}")

    # Save standardization params for inference
    import joblib
    joblib.dump(std_params, str(checkpoint_dir / "standardization_params.pkl"))
    logger.info(f"Standardization params saved")

    # Report best metrics
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val_loss = np.min(history.history["val_loss"])
    logger.info(f"Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
