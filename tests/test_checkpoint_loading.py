import numpy as np

from src.cardiofit.models.standardize_layers import StandardizeSignalFlat


def test_standardize_signal_flat_config_preserves_scaler_params():
    layer = StandardizeSignalFlat(
        mean=np.arange(6, dtype=np.float32),
        scale=np.ones(6, dtype=np.float32) * 2,
        window_size=2,
        channels=3,
    )

    cfg = layer.get_config()

    assert cfg["mean"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert cfg["scale"] == [2.0] * 6
    assert cfg["window_size"] == 2
    assert cfg["channels"] == 3
