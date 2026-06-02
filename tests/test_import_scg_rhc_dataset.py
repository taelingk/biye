import numpy as np
import pandas as pd

from scripts.import_scg_rhc_dataset import (
    build_clinical_row,
    build_label_arrays,
    has_required_signal_columns,
    normalize_record_id,
    select_signal_columns,
)


def test_normalize_record_id_converts_rhc_separator():
    assert normalize_record_id("TRM107.RHC1") == "TRM107-RHC1"
    assert normalize_record_id("TRM107-RHC1") == "TRM107-RHC1"


def test_normalize_record_id_handles_missing_values():
    assert normalize_record_id(np.nan) == ""


def test_select_signal_columns_prefers_patch_ecg_pleth_and_dv_acc():
    signal_names = [
        "patch_ECG",
        "patch_ACC_lat",
        "patch_ACC_hf",
        "patch_ACC_dv",
        "ECG_lead_I",
        "PLETH",
    ]

    selected = select_signal_columns(signal_names)

    assert selected == {"ecg": 0, "ppg": 5, "scg": 3}


def test_has_required_signal_columns_rejects_records_without_pleth():
    signal_names = [
        "patch_ECG",
        "patch_ACC_lat",
        "patch_ACC_hf",
        "patch_ACC_dv",
        "RHC_pressure",
    ]

    assert has_required_signal_columns(signal_names) is False


def test_build_label_arrays_uses_co_l_min_and_zero_vo2_placeholder():
    row = pd.Series({"Avg. COmL/min": 7005.0})

    co, vo2 = build_label_arrays(row, n_samples=4)

    assert co.shape == (4,)
    assert vo2.shape == (4,)
    np.testing.assert_allclose(co, np.array([7.005] * 4, dtype=np.float32))
    np.testing.assert_allclose(vo2, np.zeros(4, dtype=np.float32))


def test_build_clinical_row_maps_json_fields_to_pipeline_schema():
    metadata = {
        "age": 54,
        "gender": "Male",
        "weight": 121.625,
        "height": 190.5,
        "sbp": 166,
        "dbp": 90,
        "maclabMeas": {"PAHR         ": 61.0},
    }

    row = build_clinical_row("TRM107-RHC1", metadata)

    assert row["subject_id"] == "TRM107-RHC1"
    assert row["age"] == 54
    assert row["gender"] == 1
    assert row["weight_kg"] == 121.625
    assert row["height_cm"] == 190.5
    assert row["hr_bpm"] == 61.0
    assert row["sbp"] == 166
    assert row["dbp"] == 90
