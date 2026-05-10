"""AI4Pain 2026 data loader.

Reads `data/raw/{train,validation,test}/{Bvp,Eda,Resp,SpO2}/*.csv` and returns
(signals, labels, subjects) numpy arrays.

Data arrives via HIP-A. No data is loaded until that HIP completes.

ANTIPATTERNS rule 5: test labels are blind, never load test split outside HIP-G.
"""
from pathlib import Path
import numpy as np


def load_split(data_root: Path, split: str,
               signals: tuple[str, ...] = ("Bvp", "Eda", "Resp", "SpO2")
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one split from disk.

    Args:
        data_root: project's data/raw/ root
        split: 'train', 'validation', or 'test'
        signals: tuple of signal directory names to load

    Returns:
        signals_array: shape (n_trials, n_channels, n_samples)
        labels: shape (n_trials,) int values 0=NP, 1=HP, 2=AP
        subjects: shape (n_trials,) string subject IDs

    Raises ValueError if split == 'test' is loaded outside HIP-G context.
    """
    raise NotImplementedError
