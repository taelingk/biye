"""Subject-level stratified data splitting.

Interface contract:
    split_subjects(subject_ids, ratios=(0.7,0.15,0.15), seed=42)
        -> {'train': [...], 'val': [...], 'test': [...]}
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def split_subjects(
    subject_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    stratify_labels: Optional[np.ndarray] = None,
) -> dict:
    """Split subjects into train/val/test sets.

    Args:
        subject_ids: List of subject IDs.
        ratios: (train_ratio, val_ratio, test_ratio).
        seed: Random seed.
        stratify_labels: Optional (N_subjects,) array for stratification.

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    rng = np.random.default_rng(seed)
    n = len(subject_ids)
    indices = np.arange(n)

    # Shuffle
    rng.shuffle(indices)

    # Compute split sizes
    train_ratio, val_ratio, test_ratio = ratios
    n_train = int(np.round(n * train_ratio))
    n_val = int(np.round(n * val_ratio))
    n_test = n - n_train - n_val

    if stratify_labels is not None:
        # Stratified split: bin continuous labels into groups
        if stratify_labels.ndim > 1:
            stratify_labels = stratify_labels.ravel()
        from sklearn.model_selection import train_test_split

        # First split: train vs (val+test)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=n_val + n_test,
            stratify=discretize(stratify_labels),
            random_state=seed,
        )
        # Second split: val vs test
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=n_test,
            stratify=(
                discretize(stratify_labels[temp_idx])
                if len(np.unique(discretize(stratify_labels[temp_idx]))) > 1
                else None
            ),
            random_state=seed,
        )
    else:
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

    result = {
        "train": [subject_ids[i] for i in sorted(train_idx)],
        "val": [subject_ids[i] for i in sorted(val_idx)],
        "test": [subject_ids[i] for i in sorted(test_idx)],
    }

    logger.info(
        f"Split {n} subjects: train={len(result['train'])}, "
        f"val={len(result['val'])}, test={len(result['test'])}"
    )
    return result


def discretize(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Discretize continuous values into bins for stratification."""
    bins = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    return np.digitize(values, bins[:-1])


def save_splits(splits: dict, output_dir: Path) -> None:
    """Save split assignments to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "subject_splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved splits to {out_path}")


def load_splits(splits_dir: Path) -> dict:
    """Load split assignments from JSON."""
    path = splits_dir / "subject_splits.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
