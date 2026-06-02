#!/usr/bin/env python3
"""Import the SCG-RHC WFDB dataset into CardioFit raw subject folders."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ROOT = (
    "scg-rhc-wearable-seismocardiogram-signal-and-right-heart-catheter-database-1.0.0"
)
LOGGER = logging.getLogger(__name__)


def normalize_record_id(value: object) -> str:
    """Normalize metadata IDs like TRM107.RHC1 to WFDB record IDs."""
    if not isinstance(value, str):
        return ""
    return value.strip().replace(".RHC", "-RHC")


def select_signal_columns(signal_names: list[str]) -> dict[str, int]:
    """Return ECG, PPG, and SCG column indices from WFDB signal names."""
    lookup = {name: idx for idx, name in enumerate(signal_names)}
    candidates = {
        "ecg": ["patch_ECG", "ECG_lead_II", "ECG_lead_I"],
        "ppg": ["PLETH"],
        "scg": ["patch_ACC_dv", "patch_ACC_hf", "patch_ACC_lat"],
    }
    selected: dict[str, int] = {}
    for key, names in candidates.items():
        for name in names:
            if name in lookup:
                selected[key] = lookup[name]
                break
        if key not in selected:
            raise ValueError(f"Could not find {key} channel in {signal_names}")
    return selected


def has_required_signal_columns(signal_names: list[str]) -> bool:
    """Return whether a record contains all required ECG, PPG, and SCG channels."""
    try:
        select_signal_columns(signal_names)
    except ValueError:
        return False
    return True


def build_label_arrays(
    rhc_row: pd.Series, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build single-target arrays expected by the existing raw-data pipeline."""
    co_ml_min = float(rhc_row["Avg. COmL/min"])
    co_l_min = co_ml_min / 1000.0
    co = np.full(n_samples, co_l_min, dtype=np.float32)
    vo2 = np.zeros(n_samples, dtype=np.float32)
    return co, vo2


def build_clinical_row(record_id: str, metadata: dict) -> dict:
    """Map SCG-RHC JSON clinical fields to CardioFit clinical CSV columns."""
    maclab = metadata.get("maclabMeas", {})
    gender = str(metadata.get("gender", "")).strip().lower()
    return {
        "subject_id": record_id,
        "age": metadata.get("age", 30),
        "gender": 1 if gender.startswith("m") else 0,
        "weight_kg": metadata.get("weight", 70),
        "height_cm": metadata.get("height", 170),
        "hr_bpm": _first_numeric(
            maclab, ["PAHR         ", "RAHR         ", "PCWHR        "], 70
        ),
        "sbp": metadata.get("sbp", 120),
        "dbp": metadata.get("dbp", 80),
    }


def import_dataset(
    source_zip: Path,
    output_raw: Path,
    clinical_csv: Path,
    limit: int | None = None,
    records: list[str] | None = None,
) -> list[str]:
    """Import SCG-RHC records from a zip file into data/raw/<record_id>/."""
    import wfdb

    imported: list[str] = []
    output_raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as zf, tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rhc = _load_rhc_values(zf)
        record_ids = _discover_record_ids(zf)
        if records:
            wanted = {normalize_record_id(record) for record in records}
            record_ids = [record for record in record_ids if record in wanted]
        if limit is not None:
            record_ids = record_ids[:limit]

        clinical_rows = []
        for record_id in record_ids:
            rhc_row = _rhc_row_for_record(rhc, record_id)
            if rhc_row is None:
                continue
            record_dir = tmp_path / record_id
            record_dir.mkdir()
            _extract_record_files(zf, record_id, record_dir)
            record = wfdb.rdrecord(str(record_dir / record_id))
            if not has_required_signal_columns(list(record.sig_name)):
                LOGGER.warning("Skipping %s: missing ECG/PPG/SCG channel", record_id)
                continue
            selected = select_signal_columns(list(record.sig_name))
            destination = output_raw / record_id
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)

            np.save(
                destination / "ecg.npy",
                record.p_signal[:, selected["ecg"]].astype(np.float32),
            )
            np.save(
                destination / "ppg.npy",
                record.p_signal[:, selected["ppg"]].astype(np.float32),
            )
            np.save(
                destination / "scg.npy",
                record.p_signal[:, selected["scg"]].astype(np.float32),
            )
            co, vo2 = build_label_arrays(rhc_row, n_samples=1)
            np.save(destination / "co_labels.npy", co)
            np.save(destination / "vo2_labels.npy", vo2)

            metadata = json.loads(
                (record_dir / f"{record_id}.json").read_text(encoding="utf-8")
            )
            clinical_rows.append(build_clinical_row(record_id, metadata))
            imported.append(record_id)

        if clinical_rows:
            pd.DataFrame(clinical_rows).set_index("subject_id").to_csv(clinical_csv)
    return imported


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Import SCG-RHC WFDB zip into CardioFit raw data."
    )
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--output-raw", default=Path("data/raw"), type=Path)
    parser.add_argument(
        "--clinical-csv", default=Path("data/clinical_scg_rhc.csv"), type=Path
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--record", action="append", dest="records", default=None)
    args = parser.parse_args()

    imported = import_dataset(
        args.source_zip,
        args.output_raw,
        args.clinical_csv,
        limit=args.limit,
        records=args.records,
    )
    print(f"Imported {len(imported)} records")
    for record_id in imported:
        print(record_id)


def _first_numeric(source: dict, keys: list[str], default: float) -> float:
    for key in keys:
        value = source.get(key)
        if value not in (None, "", "nd"):
            try:
                return float(value)
            except ValueError:
                pass
    return float(default)


def _load_rhc_values(zf: zipfile.ZipFile) -> pd.DataFrame:
    with zf.open(f"{DATASET_ROOT}/meta_information/RHC_values.csv") as handle:
        rhc = pd.read_csv(handle)
    rhc["record_id"] = rhc["Study ID"].map(normalize_record_id)
    return rhc.loc[rhc["record_id"] != ""].copy()


def _discover_record_ids(zf: zipfile.ZipFile) -> list[str]:
    with zf.open(f"{DATASET_ROOT}/RECORDS") as handle:
        records = [line.decode("utf-8").strip() for line in handle if line.strip()]
    return [
        Path(record).name for record in records if record.startswith("processed_data/")
    ]


def _rhc_row_for_record(rhc: pd.DataFrame, record_id: str) -> pd.Series | None:
    matches = rhc.loc[rhc["record_id"] == record_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def _extract_record_files(
    zf: zipfile.ZipFile, record_id: str, destination: Path
) -> None:
    for suffix in (".hea", ".dat", ".json"):
        member = f"{DATASET_ROOT}/processed_data/{record_id}{suffix}"
        with (
            zf.open(member) as src,
            (destination / f"{record_id}{suffix}").open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst)


if __name__ == "__main__":
    main()
