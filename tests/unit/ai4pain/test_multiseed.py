"""Tests for ai4pain.multiseed (N-seed fitness averaging).

Seed variance on the full-41 task is ~0.11 absolute (discovered iter_0014:
the same champion config reseeded gave 0.5486 then 0.4352). Single-seed
fitness is therefore untrustworthy. `run_multiseed` trains a spec N times
(seed, seed+1, ..., seed+N-1), averages the metrics, and writes one
aggregated result.json so the ledger-writing path is unchanged.
"""
import json
from pathlib import Path
import pytest

from ai4pain import multiseed


def _fake_train_fn(results_by_seed):
    """Build a fake single-seed train fn that returns a canned result per seed
    and writes a result.json into its out_dir (mimics the real train_* fns)."""
    def _fn(spec, data_root, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = int(spec["training"]["seed"])
        bal = results_by_seed[seed]
        result = {
            "name": spec.get("name", "fake"),
            "best_val_metrics": {
                "balanced_acc": bal,
                "macro_f1": bal - 0.02,
                "per_class_pr": {"NP": [0.6, 0.6], "AP": [0.4, 0.4],
                                  "HP": [0.5, 0.5]},
                "confusion_3x3": [[10, 1, 1], [1, 10, 1], [1, 1, 10]],
                "auc_ovr": 0.7,
                "ece": 0.05,
                "param_count": 1000,
            },
            "train_seconds": 100.0,
            "spec": spec,
        }
        (out_dir / "result.json").write_text(json.dumps(result))
        return result
    return _fn


def test_module_imports():
    assert callable(multiseed.run_multiseed)


def test_n_seeds_1_is_passthrough(tmp_path):
    """n_seeds=1 (or absent) -> single call, no averaging wrapper."""
    spec = {"name": "t", "training": {"seed": 42, "n_seeds": 1}}
    fn = _fake_train_fn({42: 0.50})
    result = multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    assert result["best_val_metrics"]["balanced_acc"] == 0.50
    assert (tmp_path / "result.json").exists()


def test_n_seeds_absent_is_passthrough(tmp_path):
    spec = {"name": "t", "training": {"seed": 7}}
    fn = _fake_train_fn({7: 0.48})
    result = multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    assert result["best_val_metrics"]["balanced_acc"] == 0.48


def test_n_seeds_3_trains_three_times_with_offset_seeds(tmp_path):
    """n_seeds=3, base seed 100 -> trains with seeds 100, 101, 102."""
    spec = {"name": "t", "training": {"seed": 100, "n_seeds": 3}}
    fn = _fake_train_fn({100: 0.50, 101: 0.54, 102: 0.46})
    result = multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    # mean of 0.50, 0.54, 0.46 = 0.50
    assert abs(result["best_val_metrics"]["balanced_acc"] - 0.50) < 1e-9


def test_n_seeds_3_reports_per_seed_and_std(tmp_path):
    spec = {"name": "t", "training": {"seed": 100, "n_seeds": 3}}
    fn = _fake_train_fn({100: 0.50, 101: 0.54, 102: 0.46})
    result = multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    assert result["n_seeds"] == 3
    assert sorted(result["per_seed_balanced_acc"]) == [0.46, 0.50, 0.54]
    # population std of {0.46,0.50,0.54} = 0.0326...
    assert result["balanced_acc_std"] > 0.0


def test_n_seeds_3_writes_aggregated_result_json(tmp_path):
    spec = {"name": "t", "training": {"seed": 100, "n_seeds": 3}}
    fn = _fake_train_fn({100: 0.50, 101: 0.54, 102: 0.46})
    multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert abs(persisted["best_val_metrics"]["balanced_acc"] - 0.50) < 1e-9
    assert persisted["n_seeds"] == 3


def test_n_seeds_3_averages_macro_f1_and_per_class(tmp_path):
    spec = {"name": "t", "training": {"seed": 100, "n_seeds": 3}}
    fn = _fake_train_fn({100: 0.50, 101: 0.54, 102: 0.46})
    result = multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    bv = result["best_val_metrics"]
    # macro_f1 = bal - 0.02 for each seed -> mean = 0.48
    assert abs(bv["macro_f1"] - 0.48) < 1e-9
    # per_class_pr averaged (all seeds identical here)
    assert bv["per_class_pr"]["NP"] == [0.6, 0.6]
    # confusion summed across 3 seeds
    assert bv["confusion_3x3"][0][0] == 30


def test_n_seeds_3_handles_one_crashed_seed(tmp_path):
    """If a seed's train fn raises, multiseed averages the survivors and
    records the failure count. A batch shouldn't die because 1 of 3 seeds OOM'd."""
    spec = {"name": "t", "training": {"seed": 100, "n_seeds": 3}}
    def fn(spec, data_root, out_dir):
        seed = int(spec["training"]["seed"])
        if seed == 101:
            raise RuntimeError("simulated CUDA OOM")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bal = {100: 0.50, 102: 0.46}[seed]
        result = {"name": "t", "best_val_metrics": {
            "balanced_acc": bal, "macro_f1": bal,
            "per_class_pr": {"NP": [0.5, 0.5], "AP": [0.5, 0.5],
                              "HP": [0.5, 0.5]},
            "confusion_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "auc_ovr": 0.7, "ece": 0.05, "param_count": 10},
            "spec": spec}
        (out_dir / "result.json").write_text(json.dumps(result))
        return result
    result = multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    # mean of survivors 0.50, 0.46 = 0.48
    assert abs(result["best_val_metrics"]["balanced_acc"] - 0.48) < 1e-9
    assert result["n_seeds_completed"] == 2
    assert result["n_seeds_failed"] == 1


def test_multiseed_aggregates_binary_block(tmp_path):
    """When per-seed results carry a `binary` block, run_multiseed averages
    its scalars and sums its confusion_2x2."""
    spec = {"name": "t", "training": {"seed": 200, "n_seeds": 3}}

    def fn(spec, data_root, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = int(spec["training"]["seed"])
        bal = {200: 0.50, 201: 0.60, 202: 0.40}[seed]
        result = {"name": "t", "best_val_metrics": {
            "balanced_acc": bal, "macro_f1": bal,
            "per_class_pr": {"NP": [0.5, 0.5], "AP": [0.5, 0.5],
                              "HP": [0.5, 0.5]},
            "confusion_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "auc_ovr": 0.7, "ece": 0.05, "param_count": 10,
            "binary": {"balanced_acc": bal + 0.2, "f1": bal,
                        "pain_precision": 0.8, "pain_recall": 0.7,
                        "auc": 0.9, "confusion_2x2": [[2, 1], [1, 2]]}},
            "spec": spec}
        (out_dir / "result.json").write_text(json.dumps(result))
        return result

    result = multiseed.run_multiseed(fn, spec, Path("/data"), tmp_path)
    agg_bin = result["best_val_metrics"]["binary"]
    # mean of 0.70, 0.80, 0.60 = 0.70
    assert abs(agg_bin["balanced_acc"] - 0.70) < 1e-9
    # confusion_2x2 summed across 3 seeds
    assert agg_bin["confusion_2x2"][0][0] == 6
