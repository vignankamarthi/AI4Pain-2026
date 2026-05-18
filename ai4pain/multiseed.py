"""N-seed fitness averaging.

Seed variance on the full-41 AI4Pain task is large (~0.11 absolute: the same
champion config reseeded in iter_0014 produced 0.5486 then 0.4352). Optimizing
on single-seed fitness means tuning to noise. `run_multiseed` wraps any
single-seed `train_fn(spec, data_root, out_dir) -> result_dict` and, when
`spec.training.n_seeds > 1`, trains N times with consecutive seeds, averages
the metrics, and writes ONE aggregated result.json so the ledger-writing path
downstream is unchanged.

Aggregation rules:
  - scalar metrics (balanced_acc, macro_f1, auc_ovr, ece, param_count,
    train_seconds, ...): arithmetic mean over completed seeds
  - per_class_pr: element-wise mean of each [precision, recall] pair
  - confusion_3x3: element-wise SUM (a combined-trials view)
  - extra keys recorded: n_seeds, n_seeds_completed, n_seeds_failed,
    per_seed_balanced_acc, balanced_acc_std

If a seed's train_fn raises (e.g. CUDA OOM), it is skipped; the batch is not
killed. As long as >= 1 seed completes, an aggregated result is produced.
"""
from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate(per_seed_results: list[dict], spec: dict,
               n_seeds: int, n_failed: int) -> dict:
    """Average a list of single-seed result dicts into one aggregated result."""
    bvs = [r["best_val_metrics"] for r in per_seed_results]
    bal_list = [bv["balanced_acc"] for bv in bvs]

    # Scalar metrics: mean
    scalar_keys = ["balanced_acc", "macro_f1", "auc_ovr", "ece",
                    "param_count", "generalization_gap"]
    agg_metrics: dict = {}
    for k in scalar_keys:
        vals = [bv[k] for bv in bvs if k in bv]
        if vals:
            agg_metrics[k] = _mean(vals)

    # per_class_pr: element-wise mean of [precision, recall]
    if all("per_class_pr" in bv for bv in bvs):
        agg_pr: dict = {}
        for cls in bvs[0]["per_class_pr"]:
            ps = [bv["per_class_pr"][cls][0] for bv in bvs]
            rs = [bv["per_class_pr"][cls][1] for bv in bvs]
            agg_pr[cls] = [_mean(ps), _mean(rs)]
        agg_metrics["per_class_pr"] = agg_pr

    # confusion_3x3: element-wise sum
    if all("confusion_3x3" in bv for bv in bvs):
        n = len(bvs[0]["confusion_3x3"])
        summed = [[0] * n for _ in range(n)]
        for bv in bvs:
            for i in range(n):
                for j in range(n):
                    summed[i][j] += bv["confusion_3x3"][i][j]
        agg_metrics["confusion_3x3"] = summed

    # binary block (Pain vs No Pain): mean scalars, sum confusion_2x2.
    if all("binary" in bv for bv in bvs):
        bins = [bv["binary"] for bv in bvs]
        agg_bin: dict = {}
        for k in ("balanced_acc", "f1", "pain_precision", "pain_recall",
                  "auc"):
            vals = [b[k] for b in bins if k in b]
            if vals:
                agg_bin[k] = _mean(vals)
        if all("confusion_2x2" in b for b in bins):
            bsum = [[0, 0], [0, 0]]
            for b in bins:
                for i in range(2):
                    for j in range(2):
                        bsum[i][j] += b["confusion_2x2"][i][j]
            agg_bin["confusion_2x2"] = bsum
        agg_metrics["binary"] = agg_bin

    train_seconds = sum(r.get("train_seconds", 0.0) for r in per_seed_results)
    infer_seconds = _mean([r.get("inference_seconds", 0.0)
                            for r in per_seed_results])

    return {
        "name": spec.get("name", "multiseed"),
        "best_val_metrics": agg_metrics,
        "final_val_metrics": agg_metrics,
        "n_seeds": n_seeds,
        "n_seeds_completed": len(per_seed_results),
        "n_seeds_failed": n_failed,
        "per_seed_balanced_acc": bal_list,
        "balanced_acc_std": (statistics.pstdev(bal_list)
                              if len(bal_list) > 1 else 0.0),
        "train_seconds": train_seconds,
        "inference_seconds": infer_seconds,
        "spec": spec,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def run_multiseed(train_fn, spec: dict, data_root, out_dir) -> dict:
    """Train `spec` with N seeds and average, or pass through if n_seeds <= 1.

    Args:
        train_fn: single-seed trainer, signature (spec, data_root, out_dir) -> dict.
                  Must write its own result.json into out_dir.
        spec: program spec. Reads spec["training"]["n_seeds"] (default 1) and
              spec["training"]["seed"] (default 42).
        data_root: passed through to train_fn.
        out_dir: final output dir. Aggregated result.json lands here.

    Returns: the aggregated (or single-seed) result dict.
    """
    out_dir = Path(out_dir)
    train_cfg = spec.get("training", {})
    n_seeds = int(train_cfg.get("n_seeds", 1))
    base_seed = int(train_cfg.get("seed", 42))

    if n_seeds <= 1:
        return train_fn(spec, data_root, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_results: list[dict] = []
    n_failed = 0
    for s in range(n_seeds):
        seed_spec = copy.deepcopy(spec)
        seed_spec.setdefault("training", {})
        seed_spec["training"]["seed"] = base_seed + s
        # Drop n_seeds in the per-seed spec so train_fn doesn't recurse.
        seed_spec["training"].pop("n_seeds", None)
        seed_dir = out_dir / f"_seed_{s}"
        try:
            r = train_fn(seed_spec, data_root, seed_dir)
            per_seed_results.append(r)
            print(f"[multiseed] seed {base_seed + s}: "
                  f"bal={r['best_val_metrics']['balanced_acc']:.4f}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad seed must not kill the batch
            n_failed += 1
            print(f"[multiseed] seed {base_seed + s} FAILED: {e}", flush=True)

    if not per_seed_results:
        raise RuntimeError(
            f"all {n_seeds} seeds failed for spec {spec.get('name')!r}")

    agg = _aggregate(per_seed_results, spec, n_seeds, n_failed)
    _atomic_write_json(out_dir / "result.json", agg)
    print(f"[multiseed] aggregated {len(per_seed_results)}/{n_seeds} seeds: "
          f"mean_bal={agg['best_val_metrics']['balanced_acc']:.4f} "
          f"std={agg['balanced_acc_std']:.4f}", flush=True)
    return agg
