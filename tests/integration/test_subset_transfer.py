"""Integration: subset-transfer experiment exercises ai4pain.splits + framework.eval.

ANTIPATTERNS rule 19: loop cannot run on K-subject subset until this experiment
selects K via HIP-B. This integration test asserts the generation half of the
workflow (subset_transfer_experiment produces a manifest + 15 experiment dirs).
Actual cluster training is HIP-D / HIP-E / HIP-F. Result aggregation has its
own unit tests in test_eval.py.
"""
from pathlib import Path
from framework import eval as feval
from ai4pain import splits


def test_subset_transfer_experiment_generates_manifest_and_dirs(tmp_path: Path,
                                                                  sample_program_spec):
    manifest = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10, 15, 20, 25],
        n_seeds=3,
        out_dir=tmp_path,
    )
    assert "experiments" in manifest
    assert len(manifest["experiments"]) == 15
    assert (tmp_path / "manifest.json").exists()
    for exp in manifest["experiments"]:
        run_dir = Path(exp["run_dir"])
        assert (run_dir / "spec.json").exists()
        assert (run_dir / "run.py").exists()


def test_subset_subjects_disjoint_from_val():
    """ai4pain.splits is implemented; this is the green-state invariant."""
    all_subjects = [f"S{i:02d}" for i in range(41)]
    train, val = splits.subject_disjoint_split(all_subjects, n_val=5, seed=0)
    chosen = splits.k_subject_subset(train, k=10, seed=0)
    assert set(chosen).isdisjoint(set(val))
    assert set(chosen).issubset(set(train))
