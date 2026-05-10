"""Evaluation harness.

Two roles:

1. Subset-transfer meta-experiment (one-time, gated by HIP-B).
   Train one baseline architecture on subsets of K subjects (K = 5, 10, 15, 20),
   measure correlation between subset val accuracy and full-41 val accuracy.
   Pick smallest K with r >= 0.9. Persisted to ledger.

2. Per-individual fitness evaluation (every iteration).
   Train the rendered program on the chosen K-subject subset (or full 41 if no
   subset has been validated yet, per ANTIPATTERNS rule 19), evaluate on val,
   return the full fitness vector for fitness.py to score.

Spec: FRAMEWORK.md Section 3 (fitness vector), implementation deferred.
"""
from pathlib import Path


def subset_transfer_experiment(baseline_spec: dict, k_grid: list[int],
                               n_seeds: int, out_dir: Path) -> dict:
    """Run subset-transfer experiment, return correlation table.
    Gated by HIP-B for K selection.
    """
    raise NotImplementedError


def evaluate_program(run_dir: Path, k_subject_subset: list[str] | None) -> dict:
    """Train + eval a rendered program. Returns fitness vector.

    Fitness vector keys (per FRAMEWORK.md 3):
      balanced_acc, macro_f1, per_class_pr, confusion_3x3, auc_ovr, ece,
      param_count, train_seconds, inference_seconds, generalization_gap
    """
    raise NotImplementedError
