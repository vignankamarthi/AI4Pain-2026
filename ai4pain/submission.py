"""HIP-G test-set submission runner.

The evolutionary loop only ever evaluates on the 12-subject validation split.
A challenge submission needs predictions on the BLINDED 12-subject test split.
`run_submission` trains a chosen spec on the 41 train subjects, early-stops on
the validation split (same protocol the loop used), and at the best-val epoch
runs inference on the test split -- writing `test_predictions.csv`.

Submission budget is hard-capped at 5 (HIP-G); each requires Vignan's explicit
approval and is logged in experiments/SUBMISSIONS.md.

Only the `spectrogram_cnn2d` family is wired here (submission #1). Other
families raise NotImplementedError until needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

from ai4pain.data import load_split
from ai4pain.metrics import full_metric_suite
from ai4pain.baselines import _device, _atomic_write_json
from ai4pain.spectrogram import (SpectrogramCNN2D, compute_spectrogram_stack,
                                  pad_spectrograms_to_max)


def _zscore_fit(stack: np.ndarray):
    """Per-channel mean/std over (F, T'), fit on the given stack."""
    flat = stack.reshape(stack.shape[0], stack.shape[1], -1)
    mu = flat.mean(axis=(0, 2))[None, :, None, None]
    sig = flat.std(axis=(0, 2))[None, :, None, None]
    sig[sig < 1e-6] = 1.0
    return mu, sig


def _align_time(*stacks: np.ndarray) -> list[np.ndarray]:
    """Right-zero-pad a set of (N, C, F, T') stacks to a common max T'."""
    t_max = max(s.shape[-1] for s in stacks)
    out = []
    for s in stacks:
        if s.shape[-1] < t_max:
            pad = np.zeros((*s.shape[:-1], t_max - s.shape[-1]),
                           dtype=np.float32)
            s = np.concatenate([s, pad], axis=-1)
        out.append(s)
    return out


def run_submission(run_dir: Path, data_root: Path) -> dict:
    """Train spec.json's model on train, early-stop on val, predict test.

    Writes `test_predictions.csv` (subject, trial_index, pred_label, p_NP,
    p_AP, p_HP) and `result.json` (val metrics + test prediction summary)
    into run_dir.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads((run_dir / "spec.json").read_text())

    family = spec.get("model", {}).get("family")
    if family != "spectrogram_cnn2d":
        raise NotImplementedError(
            f"submission runner supports spectrogram_cnn2d only, got {family!r}")

    train_cfg = spec.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    signals = tuple(spec.get("data", {}).get(
        "signals", ["Bvp", "Eda", "Resp", "SpO2"]))
    fe = spec.get("feature_extraction", {}) or {}
    tf = dict(fs=int(fe.get("fs", 100)),
              nperseg=int(fe.get("nperseg", 64)),
              noverlap=int(fe.get("noverlap", 32)),
              log_scale=bool(fe.get("log_scale", True)),
              transform=fe.get("transform", "stft"),
              cwt_n_scales=int(fe.get("cwt_n_scales", 48)),
              cwt_time_decim=int(fe.get("cwt_time_decim", 24)),
              cwt_w0=float(fe.get("cwt_w0", 6.0)))

    print(f"[submission] loading splits from {data_root}", flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    X_test, _, subj_test = load_split(data_root, "test", signals=signals)
    print(f"[submission] {len(X_train)} train / {len(X_val)} val / "
          f"{len(X_test)} test trials", flush=True)

    Str = pad_spectrograms_to_max([compute_spectrogram_stack(x, **tf)
                                    for x in X_train])
    Sv = pad_spectrograms_to_max([compute_spectrogram_stack(x, **tf)
                                   for x in X_val])
    Ste = pad_spectrograms_to_max([compute_spectrogram_stack(x, **tf)
                                    for x in X_test])
    Str, Sv, Ste = _align_time(Str, Sv, Ste)
    mu, sig = _zscore_fit(Str)            # fit normalization on train only
    Str = ((Str - mu) / sig).astype(np.float32)
    Sv = ((Sv - mu) / sig).astype(np.float32)
    Ste = ((Ste - mu) / sig).astype(np.float32)
    F = Str.shape[2]

    device = _device()
    mc = spec.get("model", {})
    model = SpectrogramCNN2D(
        in_channels=len(signals), F=F,
        base_channels=int(mc.get("base_channels", 16)),
        depth=int(mc.get("depth", 2)),
        dropout=float(mc.get("dropout", 0.2)),
        use_residual=bool(mc.get("use_residual", False)),
        num_classes=3).to(device)

    epochs = int(train_cfg.get("epochs", 90))
    bs = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    optim_name = train_cfg.get("optimizer", "adam").lower()

    Str_t = torch.from_numpy(Str).to(device)
    ytr_t = torch.from_numpy(y_train).to(device)
    Sv_t = torch.from_numpy(Sv).to(device)
    Ste_t = torch.from_numpy(Ste).to(device)

    counts = np.bincount(y_train, minlength=3)
    class_weights = torch.tensor(
        (counts.sum() / (3 * counts)).astype(np.float32), device=device)
    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    if focal_gamma > 0.0:
        ce_per = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        def loss_fn(logits, y):
            ce = ce_per(logits, y)
            p = torch.softmax(logits, dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
            return ((1.0 - p) ** focal_gamma * ce).mean()
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optim = (torch.optim.AdamW if optim_name == "adamw"
             else torch.optim.Adam)(model.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(Str_t, ytr_t), batch_size=bs, shuffle=True)
    best_val = -math.inf
    best_state = None
    best_val_metrics: dict = {}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(Sv_t)
            vpred = vlogits.argmax(1).cpu().numpy()
            vproba = vlogits.softmax(1).cpu().numpy()
            vm = full_metric_suite(y_val, vpred, vproba)
        if vm["balanced_acc"] > best_val:
            best_val = vm["balanced_acc"]
            best_val_metrics = vm
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[submission] ep {epoch}: val_bal={vm['balanced_acc']:.4f}",
              flush=True)

    # Restore the best-val model and predict the blinded test split.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(Ste_t)
        te_proba = te_logits.softmax(1).cpu().numpy()
        te_pred = te_logits.argmax(1).cpu().numpy()

    label_names = ["NP", "AP", "HP"]
    pred_path = run_dir / "test_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_index", "pred_label", "pred_name",
                    "p_NP", "p_AP", "p_HP"])
        for i in range(len(te_pred)):
            w.writerow([int(subj_test[i]), i, int(te_pred[i]),
                        label_names[int(te_pred[i])],
                        f"{te_proba[i, 0]:.4f}", f"{te_proba[i, 1]:.4f}",
                        f"{te_proba[i, 2]:.4f}"])

    result = {
        "name": spec.get("name", "submission"),
        "submission": True,
        "best_val_metrics": best_val_metrics,
        "test_n_trials": int(len(te_pred)),
        "test_pred_class_counts": {label_names[c]: int((te_pred == c).sum())
                                    for c in range(3)},
        "test_predictions_csv": str(pred_path),
        "train_seconds": time.time() - t0,
        "device": str(device),
        "spec": spec,
    }
    _atomic_write_json(run_dir / "result.json", result)
    print(f"[submission] best val bal_acc: {best_val:.4f}", flush=True)
    print(f"[submission] test predictions -> {pred_path} "
          f"({len(te_pred)} trials)", flush=True)
    print(f"[submission] test class counts: "
          f"{result['test_pred_class_counts']}", flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return run_submission(Path(run_dir), Path(data_root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
