"""MINIROCKET feature extractor + RidgeClassifierCV.

Simplified from-scratch implementation of MINIROCKET (Dempster, Schmidt,
Webb 2020, arxiv:2012.08791). Adopted as the 5th seed (FRAMEWORK §9.4).

Design choices and deviations from the reference paper, all intentional
for clarity and our cluster constraints:

  1. **84 fixed kernels** of length 9. Each kernel has 3 positions of value
     +2 and 6 positions of value -1. C(9, 3) = 84 distinct patterns. Sum =
     3*2 + 6*(-1) = 0 (zero-centered). Same as reference.

  2. **Geometric dilation schedule** from 1 to 2^max_exp where
     max_exp = log2((T - 1) / 8). Capped at `max_dilations_per_kernel`.
     Same as reference.

  3. **Bias = 0 (simplification)**. Reference picks bias as a quantile of
     training conv outputs. We use bias=0 for clarity; this slightly weakens
     discriminative power but is robust and removes train-data dependence
     during fit. PPV = mean(conv > 0). Acceptable for first cluster run;
     can be upgraded to quantile-based biases in a later sprint if MINIROCKET
     underperforms expectations.

  4. **Channel-independent multivariate**: MINIROCKET applied per channel
     independently, features concatenated. Reference paper's
     MiniRocketMultivariate does per-kernel channel mixing; we don't.
     Simpler implementation, equally faithful spirit.

  5. **torch.nn.functional.conv1d for dilated conv**. Pure NumPy was too
     slow for 41 subjects x 36 trials x ~1118 timesteps. No SGD anywhere;
     torch is only for the vectorized convolution.

Output: per-channel features of shape (N, 84 * num_dilations). For 4
channels concatenated and 8 dilations on our T~1118 data, that's
84 * 8 * 4 = 2688 features per trial. RidgeClassifierCV handles this
high-dim input with regularization.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifierCV

from ai4pain.data import load_split
from ai4pain.metrics import full_metric_suite
from ai4pain.splits import k_subject_subset
from ai4pain.baselines import (  # reuse helpers
    pad_trials_to_max,
    per_channel_zscore,
    _atomic_write_json,
)


KERNEL_LEN = 9
NUM_KERNELS = 84  # C(9, 3)


class MiniRocket:
    """Simplified MINIROCKET feature extractor for univariate time series.

    Usage:
        mr = MiniRocket(num_features=9996, max_dilations_per_kernel=32,
                        random_state=42)
        mr.fit(X_train)         # X_train: (N, T) float
        feats = mr.transform(X)  # X: (N, T) -> (N, 84 * num_dilations)
    """

    NUM_KERNELS = NUM_KERNELS
    KERNEL_LEN = KERNEL_LEN

    def __init__(self, num_features: int = 9996,
                 max_dilations_per_kernel: int = 32,
                 random_state: int = 42,
                 bias_mode: str = "quantile"):
        self.num_features = int(num_features)
        self.max_dilations_per_kernel = int(max_dilations_per_kernel)
        self.random_state = int(random_state)
        # bias_mode: "quantile" (paper-faithful, biases from train conv quantiles)
        #            "zero" (legacy simplification, bias=0)
        self.bias_mode = bias_mode
        self._dilations: Optional[list[int]] = None
        self._kernels: Optional[torch.Tensor] = None  # (84, 1, 9)
        # When bias_mode="quantile":
        #   _biases is a list (len = num_dilations) of arrays (84, num_biases_per_pair)
        self._biases: Optional[list[np.ndarray]] = None
        self._num_biases_per_pair: int = 1

    def _build_kernels(self) -> torch.Tensor:
        """Return tensor (84, 1, 9) of the 84 fixed MINIROCKET kernels."""
        kernels = np.full((self.NUM_KERNELS, 1, self.KERNEL_LEN), -1.0,
                          dtype=np.float32)
        for ki, positions in enumerate(combinations(range(self.KERNEL_LEN), 3)):
            for p in positions:
                kernels[ki, 0, p] = 2.0
        return torch.from_numpy(kernels)

    def fit(self, X: np.ndarray) -> "MiniRocket":
        """Compute dilation schedule, build kernels, and (if bias_mode='quantile')
        select biases from training conv-output quantiles per (kernel, dilation).

        X: (N, T).
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, T); got shape {X.shape}")
        T = X.shape[1]
        if T < self.KERNEL_LEN:
            self._dilations = [1]
        else:
            max_exp = math.log2(max(1.0, (T - 1) / (self.KERNEL_LEN - 1)))
            n_dilations = min(self.max_dilations_per_kernel,
                              max(1, int(math.floor(max_exp)) + 1))
            raw = np.unique(
                np.floor(2.0 ** np.linspace(0.0, max_exp, n_dilations))
                .astype(np.int64)
            )
            valid = [int(d) for d in raw
                     if T - (self.KERNEL_LEN - 1) * int(d) >= 1]
            self._dilations = valid if valid else [1]
        self._kernels = self._build_kernels()

        if self.bias_mode == "quantile":
            self._fit_biases(X)
        return self

    def _fit_biases(self, X: np.ndarray) -> None:
        """Paper-faithful bias selection: for each (kernel, dilation), pool
        conv outputs across a sample of training trials and select evenly-
        spaced quantiles as biases. Total features per channel =
        84 * num_dilations * num_biases_per_pair, targeted at num_features.
        """
        rng = np.random.RandomState(self.random_state)
        sample_size = min(50, X.shape[0])
        sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_sample = torch.from_numpy(
            X[sample_idx].astype(np.float32)
        ).unsqueeze(1)  # (sample_size, 1, T)

        # Compute features-per-pair so total ~= num_features
        n_pairs = self.NUM_KERNELS * len(self._dilations)
        biases_per_pair = max(1, self.num_features // n_pairs)
        self._num_biases_per_pair = biases_per_pair
        # Quantiles spaced evenly in (0, 1). Skip 0 and 1 exactly.
        quantile_levels = np.linspace(
            1.0 / (biases_per_pair + 1),
            biases_per_pair / (biases_per_pair + 1),
            biases_per_pair,
        )

        self._biases = []
        for d in self._dilations:
            try:
                conv = F.conv1d(X_sample, self._kernels, dilation=int(d),
                                padding=0)  # (sample_size, 84, T_out)
            except RuntimeError:
                # Dilation too large; biases set to zeros (these won't be used
                # because transform also skips bad dilations)
                self._biases.append(np.zeros(
                    (self.NUM_KERNELS, biases_per_pair), dtype=np.float32))
                continue
            if conv.shape[-1] == 0:
                self._biases.append(np.zeros(
                    (self.NUM_KERNELS, biases_per_pair), dtype=np.float32))
                continue
            # Pool over (sample, time): for each kernel, flatten then quantile
            conv_np = conv.numpy()  # (sample_size, 84, T_out)
            biases_for_d = np.zeros((self.NUM_KERNELS, biases_per_pair),
                                     dtype=np.float32)
            for ki in range(self.NUM_KERNELS):
                pooled = conv_np[:, ki, :].flatten()
                biases_for_d[ki] = np.quantile(pooled, quantile_levels).astype(np.float32)
            self._biases.append(biases_for_d)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert X (N, T) to features.

        With bias_mode='quantile' (paper-faithful):
          features per channel = 84 * num_dilations * num_biases_per_pair
        With bias_mode='zero' (legacy):
          features per channel = 84 * num_dilations
        """
        if self._dilations is None or self._kernels is None:
            raise RuntimeError("MiniRocket: call fit() before transform()")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, T); got shape {X.shape}")
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)  # (N, 1, T)
        features: list[torch.Tensor] = []
        for di, d in enumerate(self._dilations):
            try:
                conv = F.conv1d(X_t, self._kernels, dilation=int(d),
                                padding=0)  # (N, 84, T_out)
            except RuntimeError:
                continue
            if conv.shape[-1] == 0:
                continue
            if self.bias_mode == "quantile" and self._biases is not None:
                biases = torch.from_numpy(self._biases[di])  # (84, B)
                # broadcasting: (N, 84, T_out) vs (1, 84, B, 1)
                # for each bias b: PPV = mean((conv - b) > 0, dim=time)
                B = biases.shape[1]
                conv_expanded = conv.unsqueeze(2)  # (N, 84, 1, T_out)
                biases_expanded = biases.unsqueeze(0).unsqueeze(-1)  # (1, 84, B, 1)
                ppv = (conv_expanded > biases_expanded).float().mean(dim=3)  # (N, 84, B)
                features.append(ppv.reshape(ppv.shape[0], -1))  # (N, 84*B)
            else:
                ppv = (conv > 0.0).float().mean(dim=2)  # (N, 84)
                features.append(ppv)
        if not features:
            return np.zeros((X.shape[0], self.NUM_KERNELS), dtype=np.float32)
        out = torch.cat(features, dim=1)
        return out.numpy().astype(np.float32)


def transform_multivariate(X_train: np.ndarray, X_val: np.ndarray,
                           num_features_per_channel: int = 9996,
                           max_dilations_per_kernel: int = 32,
                           random_state: int = 42,
                           bias_mode: str = "quantile"
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Apply MiniRocket per channel, concatenate features.

    X_train, X_val: (N, T, C) float arrays. Returns (train_features,
    val_features), each (N, C * 84 * num_dilations * num_biases_per_pair).
    """
    if X_train.ndim != 3 or X_val.ndim != 3:
        raise ValueError("X_train and X_val must be 3D (N, T, C)")
    C = X_train.shape[2]
    train_parts = []
    val_parts = []
    for c in range(C):
        mr = MiniRocket(
            num_features=num_features_per_channel,
            max_dilations_per_kernel=max_dilations_per_kernel,
            random_state=random_state + c,
            bias_mode=bias_mode,
        )
        mr.fit(X_train[:, :, c])
        train_parts.append(mr.transform(X_train[:, :, c]))
        val_parts.append(mr.transform(X_val[:, :, c]))
    return (np.concatenate(train_parts, axis=1).astype(np.float32),
            np.concatenate(val_parts, axis=1).astype(np.float32))


def train_minirocket(spec: dict, data_root: Path, out_dir: Path) -> dict:
    """End-to-end MINIROCKET + RidgeClassifierCV training run.

    Pipeline:
      1. Load train + val via ai4pain.data.load_split
      2. Optional K-subject subset filter
      3. Pad to global T_max, per-channel z-score (fit on train)
      4. Extract MINIROCKET features per channel, concatenate
      5. Fit RidgeClassifierCV on training features
      6. Score val with full metric suite
      7. Atomic-write result.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = spec.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    signals = tuple(spec.get("data", {}).get("signals",
                                              ["Bvp", "Eda", "Resp", "SpO2"]))
    print(f"[minirocket] loading train from {data_root}", flush=True)
    X_train, y_train, subjects_train = load_split(data_root, "train",
                                                   signals=signals)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    print(f"[minirocket] {len(X_train)} train / {len(X_val)} val trials",
          flush=True)

    data_cfg = spec.get("data", {})
    subset_size = data_cfg.get("subset_size")
    subset_seed = data_cfg.get("subset_seed", 0)
    if subset_size:
        all_train_subjects = sorted(set(subjects_train.tolist()))
        chosen = set(k_subject_subset(all_train_subjects,
                                       k=int(subset_size),
                                       seed=int(subset_seed)))
        mask = np.array([s in chosen for s in subjects_train.tolist()], dtype=bool)
        X_train = [X_train[i] for i in range(len(X_train)) if mask[i]]
        y_train = y_train[mask]
        subjects_train = subjects_train[mask]
        print(f"[minirocket] subset filter: K={subset_size} seed={subset_seed} "
              f"-> {len(X_train)} trials from {len(chosen)} subjects",
              flush=True)

    Xtr = pad_trials_to_max(X_train)
    Xv = pad_trials_to_max(X_val)
    T_max = max(Xtr.shape[1], Xv.shape[1])
    if Xtr.shape[1] < T_max:
        pad = np.zeros((Xtr.shape[0], T_max - Xtr.shape[1], Xtr.shape[2]),
                       dtype=np.float32)
        Xtr = np.concatenate([Xtr, pad], axis=1)
    if Xv.shape[1] < T_max:
        pad = np.zeros((Xv.shape[0], T_max - Xv.shape[1], Xv.shape[2]),
                       dtype=np.float32)
        Xv = np.concatenate([Xv, pad], axis=1)
    Xtr, Xv, _, _ = per_channel_zscore(Xtr, Xv)

    fe_cfg = spec.get("feature_extraction", {}) or {}
    num_features = int(fe_cfg.get("num_features", 9996))
    rs = int(fe_cfg.get("random_state", seed))
    bias_mode = fe_cfg.get("bias_mode", "quantile")
    max_dils = int(fe_cfg.get("max_dilations_per_kernel", 32))

    print(f"[minirocket] extracting features (bias_mode={bias_mode}, "
          f"num_features={num_features}, max_dils={max_dils})...", flush=True)
    t0 = time.time()
    train_feats, val_feats = transform_multivariate(
        X_train=Xtr, X_val=Xv,
        num_features_per_channel=num_features,
        max_dilations_per_kernel=max_dils,
        random_state=rs,
        bias_mode=bias_mode,
    )
    feat_seconds = time.time() - t0
    print(f"[minirocket] features: train {train_feats.shape}, "
          f"val {val_feats.shape} in {feat_seconds:.1f}s", flush=True)

    model_cfg = spec.get("model", {})
    alphas = model_cfg.get("alphas",
                            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    class_weight = model_cfg.get("class_weight", "balanced")
    print(f"[minirocket] fitting RidgeClassifierCV with {len(alphas)} alphas",
          flush=True)
    t1 = time.time()
    clf = RidgeClassifierCV(alphas=alphas, class_weight=class_weight)
    clf.fit(train_feats, y_train)
    train_seconds = time.time() - t1

    t2 = time.time()
    val_preds = clf.predict(val_feats)
    # Pseudo-probabilities via softmax over decision_function for AUC/ECE
    df = clf.decision_function(val_feats)
    if df.ndim == 1:
        # Binary -> turn into 2-col proba via sigmoid
        proba = np.column_stack([1 / (1 + np.exp(df)), 1 / (1 + np.exp(-df))])
    else:
        # Multi-class: softmax
        e = np.exp(df - df.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
    inference_seconds = time.time() - t2

    metrics = full_metric_suite(y_val, val_preds, proba)

    result = {
        "name": spec.get("name", "minirocket"),
        "best_val_metrics": metrics,
        "final_val_metrics": metrics,
        "history": [{"epoch": 0,
                     "train_loss": float("nan"),
                     "train_bal_acc": float("nan"),
                     "val_bal_acc": metrics["balanced_acc"],
                     "val_macro_f1": metrics["macro_f1"]}],
        "param_count": int(train_feats.shape[1] * 3),  # n_features * n_classes for ridge
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "feat_seconds": feat_seconds,
        "device": "cpu",
        "best_alpha": float(getattr(clf, "alpha_", float("nan"))),
        "spec": spec,
    }
    _atomic_write_json(out_dir / "result.json", result)
    print(f"[minirocket] best val bal_acc: {metrics['balanced_acc']:.3f} "
          f"(alpha={result['best_alpha']})", flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    """Read spec.json, train MINIROCKET, write result.json into run_dir."""
    run_dir = Path(run_dir)
    spec_path = run_dir / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"missing spec.json at {spec_path}")
    spec = json.loads(spec_path.read_text())
    return train_minirocket(spec, data_root=Path(data_root), out_dir=run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
