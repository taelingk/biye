"""Signal preprocessing pipeline for multi-modal ECG + PPG + SCG data.

Modules:
    ecg_processor: Bandpass (0.5-40Hz) → Notch (50Hz) → Baseline correction →
                   Resample (500→125Hz) → Normalize → R-peak detection.
    ppg_processor: IR channel selection → Bandpass (0.5-8Hz) →
                   Resample (100→125Hz) → Normalize.
    scg_processor: Z-axis selection → Bandpass (1-40Hz) →
                   Resample (800→125Hz) → Normalize.
    sync_and_segment: Truncate to common length → Extract (125,3) windows at R-peaks.
    build_hdf5: Orchestrate full pipeline → Write HDF5 per subject.

Interface contracts (do not change without updating AI_CONTEXT.md):

    process_ecg(raw, fs=500) -> (signal_125hz, r_peaks_125hz)
    process_ppg(raw, fs=100, ch_ir=1) -> signal_125hz
    process_scg(raw, fs=800, axis_z=2) -> signal_125hz
    extract_windows(ecg, ppg, scg, r_peaks, window_size=125, before_r=40) -> (N, 125, 3)
    build_subject_hdf5(raw_dir, output_dir, subject_id, clinical, config) -> None
"""

from .ecg_processor import process_ecg
from .ppg_processor import process_ppg
from .scg_processor import process_scg
from .sync_and_segment import extract_windows, compute_hr_from_r_peaks
from .build_hdf5 import build_subject_hdf5, compute_clinical_features

__all__ = [
    "process_ecg",
    "process_ppg",
    "process_scg",
    "extract_windows",
    "compute_hr_from_r_peaks",
    "build_subject_hdf5",
    "compute_clinical_features",
]
