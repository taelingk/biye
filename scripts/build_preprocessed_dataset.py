#!/usr/bin/env python3
"""Build preprocessed HDF5 dataset from raw multi-modal signals.

Usage:
    python scripts/build_preprocessed_dataset.py
    python scripts/build_preprocessed_dataset.py --config configs/default.yaml
    python scripts/build_preprocessed_dataset.py --subject subject_001
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cardiofit.preprocessing import build_subject_hdf5
from src.cardiofit.utils.logging import setup_logging


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build preprocessed HDF5 dataset")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Process a single subject (e.g., 'subject_001'). If omitted, process all."
    )
    parser.add_argument(
        "--clinical", type=str, default=None,
        help="Path to clinical registry CSV (subjects x clinical features)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / cfg["paths"]["raw_data"]
    output_dir = project_root / cfg["paths"]["processed_data"]

    if not raw_dir.exists():
        logging.error(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)

    # Load clinical registry
    if args.clinical:
        import pandas as pd
        clinical_df = pd.read_csv(args.clinical, index_col=0)
    else:
        clinical_df = None

    # Determine subjects to process
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = sorted([
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    if not subjects:
        logging.error(f"No subject directories found in {raw_dir}")
        sys.exit(1)

    logging.info(f"Processing {len(subjects)} subjects")

    for sid in subjects:
        # Get clinical data for this subject
        if clinical_df is not None and sid in clinical_df.index:
            row = clinical_df.loc[sid]
            clinical_raw = {
                "age": row.get("age", 30),
                "gender": row.get("gender", 0),
                "weight_kg": row.get("weight_kg", 70),
                "height_cm": row.get("height_cm", 170),
                "hr_bpm": row.get("hr_bpm", 70),
                "sbp": row.get("sbp", 120),
                "dbp": row.get("dbp", 80),
            }
        else:
            # Default clinical parameters (will be overridden by actual measurements)
            clinical_raw = {
                "age": 30, "gender": 0, "weight_kg": 70, "height_cm": 170,
                "hr_bpm": 70, "sbp": 120, "dbp": 80,
            }

        try:
            build_subject_hdf5(raw_dir, output_dir, sid, clinical_raw, cfg)
        except FileNotFoundError as e:
            logging.error(f"Skipping {sid}: {e}")
        except Exception as e:
            logging.error(f"Failed to process {sid}: {e}", exc_info=True)

    logging.info("Preprocessing complete")


if __name__ == "__main__":
    main()
