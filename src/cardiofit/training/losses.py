"""Custom loss functions for multi-output training."""

import tensorflow as tf


def huber_loss(delta: float = 1.0):
    """Standard Huber loss, mirrors paper."""
    return tf.keras.losses.Huber(delta=delta)


def dual_huber_loss(y_true: dict, y_pred: dict, delta: float = 1.0):
    """Compute separate Huber losses for CO and VO2 outputs.

    Args:
        y_true: {"co_output": tensor, "vo2_output": tensor}
        y_pred: {"co_output": tensor, "vo2_output": tensor}
        delta: Huber delta parameter.

    Returns:
        Total loss = Huber(co) + Huber(vo2)
    """
    huber = tf.keras.losses.Huber(delta=delta)
    loss_co = huber(y_true["co_output"], y_pred["co_output"])
    loss_vo2 = huber(y_true["vo2_output"], y_pred["vo2_output"])
    return loss_co + loss_vo2
