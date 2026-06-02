import numpy as np

from src.cardiofit.preprocessing.build_hdf5 import _load_labels


def test_load_labels_reads_npy_label_files(tmp_path):
    np.save(tmp_path / "co_labels.npy", np.array([7.005], dtype=np.float32))
    np.save(tmp_path / "vo2_labels.npy", np.array([0.0], dtype=np.float32))

    co, vo2 = _load_labels(tmp_path, n_windows=3)

    assert co.shape == (3, 1)
    assert vo2.shape == (3, 1)
    np.testing.assert_allclose(co.ravel(), [7.005, 7.005, 7.005])
    np.testing.assert_allclose(vo2.ravel(), [0.0, 0.0, 0.0])
