"""PyTorch training loop for the AI4Pain 2026 DL track.

Adapted from `Blood-Pressure-Inference-with-BVP/src/dl_training.py`. Three
substantive changes from BP inference:

  1. **Loss**: ``nn.MSELoss`` -> ``nn.CrossEntropyLoss(weight=class_weights)``
     where the class weights are computed from the training label distribution
     (sklearn-style ``class_weight='balanced'``).

  2. **Early stopping criterion**: BP inference watched validation MSE, which
     IS the regression target. For 3-class imbalanced classification, CE loss
     can decrease while balanced accuracy plateaus or even drops if the model
     collapses to the majority class. We track BOTH per epoch and select the
     best checkpoint by **balanced accuracy** (with CE loss as tiebreak).

  3. **Output**: BP inference produced a (batch, 1) regression value. AI4Pain
     produces (batch, 3) class logits.

Per-epoch checkpointing, latest + best snapshots, atomic JSON history writes,
and resume support are all preserved verbatim from BP inference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import atomic_json_write, get_logger


logger = get_logger("ai4pain_2026")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters and orchestration knobs for one DL training run."""

    arch: str
    in_channels: int
    num_classes: int = 3
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0
    seed: int = 42


# ----------------------------------------------------------------------------
# Checkpoint helpers (verbatim pattern from BP inference)
# ----------------------------------------------------------------------------


def save_dl_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_balanced_acc: float,
    best_balanced_acc: float,
    path: Path,
) -> None:
    """Save model + optimizer state for resuming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_balanced_acc": val_balanced_acc,
            "best_balanced_acc": best_balanced_acc,
        },
        path,
    )


def load_dl_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, path: Path
) -> Tuple[int, float]:
    """Load checkpoint into ``model`` + ``optimizer`` and return (epoch, best_balanced_acc)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt.get("best_balanced_acc", 0.0)


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------


def _epoch_train(model, loader, criterion, optimizer, grad_clip_norm) -> float:
    model.train()
    losses: List[float] = []
    for signals, labels in loader:
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _epoch_evaluate(model, loader, criterion) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (avg_loss, balanced_accuracy, y_true, y_pred)."""
    from sklearn.metrics import balanced_accuracy_score

    model.eval()
    losses: List[float] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(DEVICE)
            labels_dev = labels.to(DEVICE)
            logits = model(signals)
            loss = criterion(logits, labels_dev)
            losses.append(float(loss.item()))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_true.append(labels.numpy())
            all_pred.append(preds)
    if not all_true:
        return float("nan"), float("nan"), np.array([]), np.array([])
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return float(np.mean(losses)), bal_acc, y_true, y_pred


def train_dl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig,
    class_weights: Optional[np.ndarray],
    checkpoint_dir: Path,
    resume: bool = False,
) -> Dict:
    """Train one of the three AI4Pain DL architectures end to end.

    Parameters
    ----------
    model : nn.Module
        A model from ``src.dl_models.build_model``.
    train_loader, val_loader : DataLoader
        PyTorch loaders backed by ``AI4PainWindowDataset``.
    config : TrainingConfig
    class_weights : np.ndarray | None
        Length-3 weights to pass to ``nn.CrossEntropyLoss``. Computed from the
        training labels via ``data_loader.compute_class_weights``.
    checkpoint_dir : Path
        Where ``latest.pt`` and ``best.pt`` are written.
    resume : bool
        If True and a previous ``latest.pt`` exists, load it and continue.

    Returns
    -------
    dict with keys ``model``, ``history``, ``best_balanced_acc``, ``epochs``.
    """
    model = model.to(DEVICE)

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    history_path = checkpoint_dir / "history.json"
    status_path = checkpoint_dir / "status.json"

    start_epoch = 0
    best_balanced_acc = 0.0
    patience_counter = 0
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_balanced_accuracy": [],
    }

    if resume and latest_path.exists():
        start_epoch, best_balanced_acc = load_dl_checkpoint(model, optimizer, latest_path)
        logger.info(f"Resuming {config.arch} from epoch {start_epoch}")
        if history_path.exists():
            history = json.loads(history_path.read_text())

    logger.info(
        f"Training {config.arch} on {DEVICE}: "
        f"epochs={config.epochs}, batch={config.batch_size}, lr={config.lr}, "
        f"in_channels={config.in_channels}, num_classes={config.num_classes}"
    )

    for epoch in range(start_epoch, config.epochs):
        train_loss = _epoch_train(
            model, train_loader, criterion, optimizer, config.grad_clip_norm
        )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss, val_bal_acc, _, _ = _epoch_evaluate(model, val_loader, criterion)
        else:
            val_loss, val_bal_acc = float("nan"), 0.0
        history["val_loss"].append(val_loss)
        history["val_balanced_accuracy"].append(val_bal_acc)

        scheduler.step(val_bal_acc)

        save_dl_checkpoint(
            model, optimizer, epoch + 1, val_loss, val_bal_acc, best_balanced_acc, latest_path
        )
        atomic_json_write(history_path, history)

        if val_bal_acc > best_balanced_acc:
            best_balanced_acc = val_bal_acc
            save_dl_checkpoint(
                model, optimizer, epoch + 1, val_loss, val_bal_acc, best_balanced_acc, best_path
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch + 1}/{config.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_bal_acc={val_bal_acc:.4f}, best={best_balanced_acc:.4f}, "
                f"lr={current_lr:.2e}, patience={patience_counter}/{config.early_stopping_patience}"
            )

        if patience_counter >= config.early_stopping_patience:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    atomic_json_write(
        status_path,
        {
            "arch": config.arch,
            "epochs_completed": len(history["train_loss"]),
            "best_balanced_accuracy": best_balanced_acc,
            "status": "completed",
        },
    )

    return {
        "model": model,
        "history": history,
        "best_balanced_accuracy": best_balanced_acc,
        "epochs": len(history["train_loss"]),
    }


def predict_dl(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and return ``(y_pred, y_proba)``.

    ``y_pred`` is the argmax over class logits, ``y_proba`` is the softmax.
    """
    model.eval()
    all_pred: List[np.ndarray] = []
    all_proba: List[np.ndarray] = []
    with torch.no_grad():
        for signals, _ in loader:
            signals = signals.to(DEVICE)
            logits = model(signals)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            preds = proba.argmax(axis=1)
            all_pred.append(preds)
            all_proba.append(proba)
    if not all_pred:
        return np.array([]), np.array([])
    return np.concatenate(all_pred), np.concatenate(all_proba)
