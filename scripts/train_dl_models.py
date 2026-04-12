"""Deep learning training entry point for AI4Pain 2026.

Adapted from `Blood-Pressure-Inference-with-BVP/scripts/train_dl_models.py`.

Trains one of the three AI4Pain DL architectures on raw multi-channel
physiological signal windows for 3-class pain localization. The architecture
is selected by ``--arch``:

  --arch cnn1d : NN #1 ResNet-1D baseline (Phase 3 implements this)
  --arch nn2   : NN #2 Current SOTA (raises NotImplementedError until Phase 4 +
                 HIP-5 approval)
  --arch nn3   : NN #3 Novel framework (raises NotImplementedError until
                 Phase 5 + HIP-6 approval)

Usage:
    python scripts/train_dl_models.py --arch cnn1d --config bvp_eda --dry-run
    python scripts/train_dl_models.py --arch cnn1d --config bvp_eda
    python scripts/train_dl_models.py --arch cnn1d --config all_four --epochs 50

Dry-run mode generates a small synthetic windowed dataset and runs the full
loop end-to-end without requiring real data. This is the Phase 3 verification
target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/train_dl_models.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data_loader import compute_class_weights  # noqa: E402
from src.dl_data import (  # noqa: E402
    AI4PainWindowDataset,
    SIGNAL_CHANNELS,
    make_synthetic_window_dataset,
)
from src.dl_models import build_model  # noqa: E402
from src.dl_training import TrainingConfig, predict_dl, train_dl_model  # noqa: E402
from src.evaluation import evaluate_model, generate_leaderboard  # noqa: E402
from src.utils import set_all_seeds, setup_logging, timer  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_CONFIG_PATH = REPO_ROOT / "configs" / "ablation_configs.json"
ARCH_TO_RESULTS_DIR = {
    "cnn1d": REPO_ROOT / "results" / "dl_cnn_baseline",
    "nn2": REPO_ROOT / "results" / "dl_nn2_sota",
    "nn3": REPO_ROOT / "results" / "dl_nn3_novel",
}
DEFAULT_CHECKPOINT_BASE = REPO_ROOT / "checkpoints" / "dl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["cnn1d", "nn2", "nn3"])
    parser.add_argument(
        "--config", required=True, choices=["bvp_eda", "bvp_eda_resp", "all_four"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a small synthetic windowed dataset for end-to-end smoke testing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest.pt if present")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_BASE,
                        help="Base checkpoint directory (namespaced by arch + config)")
    return parser.parse_args(argv)


def load_ablation(name: str) -> dict:
    with open(ABLATION_CONFIG_PATH) as f:
        return json.load(f)[name]


def get_synthetic_loaders(
    ablation: dict, batch_size: int, seed: int, logger
) -> tuple[DataLoader, DataLoader, int]:
    n_channels = ablation["dl_channels"]
    logger.info(
        f"DRY-RUN: synthesizing windowed data ({n_channels} channels, 1024 samples per window)"
    )
    train_ds = make_synthetic_window_dataset(
        n_subjects=15,
        n_trials_per_subject=4,
        n_channels=n_channels,
        window_length=1024,
        seed=seed,
    )
    val_ds = make_synthetic_window_dataset(
        n_subjects=6,
        n_trials_per_subject=4,
        n_channels=n_channels,
        window_length=1024,
        seed=seed + 1,
    )
    # Fit per-channel scaler stats on training set only
    stats = train_ds.compute_scaler_stats()
    train_ds.scaler_stats = stats
    val_ds.scaler_stats = stats

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, n_channels


def get_real_loaders(args, ablation: dict, logger):
    """Real-data loader path. Stub for now: blocked on RT-E (windowing strategy).

    Once data arrives and RT-E approves a concrete window length / stride,
    this function will:
      1. Read raw subject CSVs from data/raw/{train,validation}/{Bvp,Eda,Resp,SpO2}/
      2. Resample channels to a common sampling rate
      3. Window into fixed-length segments per the RT-E decision
      4. Construct AI4PainWindowDataset(signals, labels) for each split
      5. Compute scaler_stats on train, share with val
    """
    raise NotImplementedError(
        "Real-data DL loader is gated on RT-E (windowing strategy). "
        "Use --dry-run for now."
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logging(name="ai4pain_2026")
    set_all_seeds(args.seed)

    ablation = load_ablation(args.config)
    arch = args.arch

    results_dir = ARCH_TO_RESULTS_DIR[arch] / args.config
    checkpoint_dir = args.checkpoint_dir / arch / args.config
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Arch: {arch}")
    logger.info(f"Config: {args.config} (channels: {SIGNAL_CHANNELS[args.config]})")
    logger.info(f"Mode: {'DRY-RUN (synthetic)' if args.dry_run else 'real data'}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    with timer("Loading DL data", logger):
        if args.dry_run:
            train_loader, val_loader, n_channels = get_synthetic_loaders(
                ablation, args.batch_size, args.seed, logger
            )
        else:
            train_loader, val_loader, n_channels = get_real_loaders(args, ablation, logger)

    # Class weights from training labels (sklearn-style 'balanced')
    train_labels = np.concatenate([batch[1].numpy() for batch in train_loader])
    cw_dict = compute_class_weights(train_labels)
    class_weights = np.array([cw_dict[i] for i in sorted(cw_dict.keys())], dtype=np.float32)
    logger.info(f"Class weights (NP/HP/AP): {class_weights.tolist()}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = build_model(arch=arch, in_channels=n_channels, num_classes=3)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    config = TrainingConfig(
        arch=arch,
        in_channels=n_channels,
        num_classes=3,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.patience,
        seed=args.seed,
    )

    with timer(f"Training {arch}", logger):
        result = train_dl_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            class_weights=class_weights,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
        )

    logger.info(
        f"Training done. Best val balanced_accuracy: {result['best_balanced_accuracy']:.4f}, "
        f"epochs: {result['epochs']}"
    )

    # ------------------------------------------------------------------
    # 4. Evaluate on validation split
    # ------------------------------------------------------------------
    with timer("Final validation evaluation", logger):
        y_pred, y_proba = predict_dl(result["model"], val_loader)
        y_true = np.concatenate([batch[1].numpy() for batch in val_loader])
        evaluate_model(f"dl_{arch}", y_true, y_pred, y_proba, results_dir)

    leaderboard = generate_leaderboard(results_dir)
    logger.info("=" * 60)
    logger.info(f"DL LEADERBOARD ({arch} / {args.config})")
    logger.info("=" * 60)
    for entry in leaderboard["entries"]:
        flag = " [COLLAPSED]" if entry["single_class_collapse"] else ""
        logger.info(
            f"  {entry['model']:22s}: bal_acc={entry['balanced_accuracy']:.4f}, "
            f"macro_f1={entry['macro_f1']:.4f}, AUC-OVR={entry['auc_ovr']:.4f}{flag}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
