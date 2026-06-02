"""Data loading and splitting utilities.

Interface contracts:
    load_subject(filepath) -> (signals (N,125,3), clinical (N,10), co (N,1), vo2 (N,1))
    build_tf_dataset(subject_ids, data_root, ...) -> tf.data.Dataset
    split_subjects(subject_ids, ratios, seed) -> {'train': [...], 'val': [...], 'test': [...]}
"""

from .augmentation import augment_segment_tf, tf_augment_wrapper
from .data_split import load_splits, save_splits, split_subjects
from .hdf5_loader import (
    build_tf_dataset,
    fit_standardization_params,
    load_multiple_subjects,
    load_subject,
)

__all__ = [
    "load_subject",
    "load_multiple_subjects",
    "build_tf_dataset",
    "fit_standardization_params",
    "split_subjects",
    "save_splits",
    "load_splits",
    "augment_segment_tf",
    "tf_augment_wrapper",
]
