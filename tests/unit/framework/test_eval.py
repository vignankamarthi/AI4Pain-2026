"""TDD scaffold for framework.eval. Spec: FRAMEWORK.md Section 3 + Open Q 2."""
import pytest
from pathlib import Path
from framework import eval as feval


def test_module_imports():
    assert callable(feval.evaluate_program)
    assert callable(feval.subset_transfer_experiment)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_subset_transfer_returns_correlation_table(tmp_path: Path, sample_program_spec):
    table = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10, 15, 20],
        n_seeds=3,
        out_dir=tmp_path,
    )
    assert "correlations" in table
    assert "chosen_k" in table


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_evaluate_program_returns_full_fitness_vector(tmp_path: Path):
    fv = feval.evaluate_program(run_dir=tmp_path, k_subject_subset=None)
    for key in ("balanced_acc", "macro_f1", "ece", "param_count",
                "generalization_gap", "auc_ovr"):
        assert key in fv
