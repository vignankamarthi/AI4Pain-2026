"""PyTorch Dataset for raw multi-channel physiological windows.

Adapted from `Blood-Pressure-Inference-with-BVP/src/dl_data.py`. Two
substantive changes from BP inference:

  1. **Multi-channel input**: BP inference fed 1- or 2-channel raw PPG / ECG.
     AI4Pain has 4 candidate channels (BVP, EDA, RESP, SpO2) with per-signal
     sampling rates that may not match. The Dataset accepts a per-ablation
     channel list and a fixed window length; the windowing strategy itself
     is parameterized so RT-E can confirm it from real data later.

  2. **3-class integer labels**: BP inference returned a continuous regression
     target. AI4Pain returns an integer label in {0, 1, 2}.

The class supports both real data (loaded from numpy arrays) and synthetic
data (generated in-place from a seed) so the Phase 3 dry-run can exercise the
full DL pipeline without a real release.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# AI4Pain ablation -> ordered channel keys
SIGNAL_CHANNELS = {
    "bvp_eda": ["bvp", "eda"],
    "bvp_eda_resp": ["bvp", "eda", "resp"],
    "all_four": ["bvp", "eda", "resp", "spo2"],
}


class AI4PainWindowDataset(Dataset):
    """Multi-channel raw-signal windows with 3-class pain localization labels.

    Each item returns:
      ``signal`` : float tensor of shape (n_channels, window_length)
      ``label``  : long tensor scalar in {0, 1, 2}

    Parameters
    ----------
    signals : np.ndarray
        Float array of shape ``(n_samples, n_channels, window_length)``.
        Phase 3 takes pre-windowed arrays as input; the windowing pipeline
        from raw subject CSVs to fixed-length windows is finalized at RT-E.
    labels : np.ndarray
        Integer array of shape ``(n_samples,)`` with values in {0, 1, 2}.
    subject_ids : np.ndarray | None
        Optional subject identifiers for diagnostics. Not used by the
        Dataset itself but kept around for downstream slicing.
    scaler_stats : dict | None
        Optional ``{'mean': [...], 'std': [...]}`` per-channel stats for
        z-scoring. Fitted on training subjects only. If None, the signal
        is returned as-is.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        scaler_stats: Optional[dict] = None,
    ):
        if signals.ndim != 3:
            raise ValueError(
                f"signals must be 3D (n_samples, n_channels, window_length), got shape {signals.shape}"
            )
        if labels.ndim != 1 or len(labels) != signals.shape[0]:
            raise ValueError(
                f"labels must be 1D with length {signals.shape[0]}, got shape {labels.shape}"
            )
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.subject_ids = subject_ids
        self.scaler_stats = scaler_stats

        self.n_samples = signals.shape[0]
        self.n_channels = signals.shape[1]
        self.window_length = signals.shape[2]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        signal = self.signals[idx].copy()  # (n_channels, window_length)

        if self.scaler_stats is not None:
            means = self.scaler_stats["mean"]
            stds = self.scaler_stats["std"]
            for ch in range(signal.shape[0]):
                std = stds[ch]
                if std > 1e-8:
                    signal[ch] = (signal[ch] - means[ch]) / std

        return (
            torch.from_numpy(signal),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def compute_scaler_stats(self) -> dict:
        """Compute per-channel mean and std on this dataset (call on training set only).

        Returns ``{'mean': [...], 'std': [...]}`` with one entry per channel.
        Subsequent val/test datasets should be constructed with these stats so
        the StandardScaler invariant (fit on train only) is preserved.
        """
        means = []
        stds = []
        for ch in range(self.n_channels):
            channel_data = self.signals[:, ch, :].ravel().astype(np.float64)
            means.append(float(channel_data.mean()))
            stds.append(float(channel_data.std()))
        return {"mean": means, "std": stds}


def make_synthetic_window_dataset(
    n_subjects: int,
    n_trials_per_subject: int,
    n_channels: int,
    window_length: int = 1024,
    seed: int = 42,
) -> AI4PainWindowDataset:
    """Build a small synthetic windowed dataset for the Phase 3 dry-run.

    Each trial has a slight class-conditional amplitude shift on every channel
    so the classifier has signal to learn from. Subject ids are assigned in
    blocks so subject-disjoint splits remain possible downstream.
    """
    rng = np.random.default_rng(seed)
    n_samples = n_subjects * n_trials_per_subject

    labels = rng.integers(0, 3, size=n_samples)
    subject_ids = np.repeat([f"S{i:03d}" for i in range(n_subjects)], n_trials_per_subject)

    # Base noise plus class-conditional sinusoid amplitude shift
    t = np.linspace(0, 4 * np.pi, window_length, dtype=np.float32)
    signals = np.empty((n_samples, n_channels, window_length), dtype=np.float32)
    for i in range(n_samples):
        cls = labels[i]
        amplitude = 1.0 + 0.5 * (cls - 1)  # 0.5, 1.0, 1.5 for the 3 classes
        for ch in range(n_channels):
            phase = ch * np.pi / 4
            base = amplitude * np.sin(t + phase)
            noise = rng.standard_normal(window_length).astype(np.float32) * 0.3
            signals[i, ch] = base + noise

    return AI4PainWindowDataset(signals=signals, labels=labels, subject_ids=subject_ids)
