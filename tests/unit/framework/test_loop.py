"""TDD scaffold for framework.loop. Spec: FRAMEWORK.md Section 8."""
import pytest
from framework import loop


def test_module_imports():
    assert loop.IterationOutcome is not None
    assert loop.IterationPaused is not None
    assert loop.IterationCompleted is not None


def test_iteration_paused_carries_hip_label():
    p = loop.IterationPaused(hip="HIP-D", run_id="r1", action_required="rsync up")
    assert p.hip == "HIP-D"


def test_iteration_completed_carries_fitness(sample_fitness_vector):
    c = loop.IterationCompleted(run_id="r1", fitness_vector=sample_fitness_vector)
    assert c.fitness_vector["balanced_acc"] == 0.55


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_advance_one_iteration_runs(tmp_path):
    out = loop.advance_one_iteration(experiments_root=tmp_path / "experiments",
                                     ledger_path=tmp_path / "ledger.db")
    assert isinstance(out, loop.IterationOutcome)
