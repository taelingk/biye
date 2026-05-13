"""Training callbacks — cosine annealing lr scheduler, early stopping.

Mirrors the paper's training configuration.
"""

import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def cosine_annealing(epoch: int, lr: float, total_epochs: int = 100) -> float:
    """Cosine annealing: lr(t) = lr_0 * (1 + cos(pi * t / T)) / 2.

    Args:
        epoch: Current epoch (0-indexed).
        lr: Initial learning rate.
        total_epochs: Total number of epochs.

    Returns:
        Annealed learning rate.
    """
    return lr * (1.0 + np.cos(np.pi * epoch / total_epochs)) / 2.0


class CosineAnnealingCallback(tf.keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler."""

    def __init__(
        self, initial_lr: float = 0.001, total_epochs: int = 100, verbose: int = 1
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = cosine_annealing(epoch, self.initial_lr, self.total_epochs)
        self.model.optimizer.learning_rate.assign(new_lr)
        if self.verbose:
            logger.info(f"Epoch {epoch + 1}: learning rate = {new_lr:.6f}")


def get_callbacks(
    checkpoint_dir: str,
    patience: int = 20,
    initial_lr: float = 0.001,
    total_epochs: int = 100,
) -> list:
    """Get standard training callbacks matching the paper's setup.

    Args:
        checkpoint_dir: Directory to save model checkpoints.
        patience: Early stopping patience.
        initial_lr: Initial learning rate.
        total_epochs: Total training epochs.

    Returns:
        List of tf.keras.callbacks.Callback instances.
    """
    callbacks = [
        CosineAnnealingCallback(initial_lr=initial_lr, total_epochs=total_epochs),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{checkpoint_dir}/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(f"{checkpoint_dir}/training_history.csv"),
    ]
    return callbacks
