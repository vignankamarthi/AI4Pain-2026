"""TDD scaffold for framework.fitness. Spec: FRAMEWORK.md Section 3."""
import pytest
import numpy as np
from framework import fitness


def test_module_imports():
    assert callable(fitness.pareto_rank)


def test_confidence_weighted_is_already_implemented():
    """Section 3.3: simple closed form, implemented in stub."""
    s = fitness.confidence_weighted(accuracy=0.8, ece=0.1, lam=1.0)
    assert s == pytest.approx(0.8 * 0.9)


def test_confidence_weighted_zero_ece_unchanged():
    s = fitness.confidence_weighted(accuracy=0.7, ece=0.0, lam=1.0)
    assert s == pytest.approx(0.7)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_pareto_rank_returns_ranks():
    fvs = [
        {"balanced_acc": 0.7, "ece": 0.1, "param_count": 100_000, "generalization_gap": 0.05},
        {"balanced_acc": 0.8, "ece": 0.2, "param_count": 200_000, "generalization_gap": 0.10},
    ]
    ranks = fitness.pareto_rank(fvs, axes=["balanced_acc", "ece", "param_count", "generalization_gap"])
    assert len(ranks) == 2


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_novelty_score_grows_with_distance():
    child = np.array([0.5, 0.3, 0.2])
    pop = [np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.4, 0.2])]
    fitness.novelty_score(child, pop, k=2)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_failure_aware_boost_activates_on_negative_deltas():
    fitness.failure_aware_boost(recent_deltas=[-0.05, -0.04, -0.03], threshold=-0.02)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_scalar_score_combines_components():
    fitness.scalar_score(pareto_rank_value=1, novelty=0.3, accuracy=0.7,
                         ece=0.05, alpha=0.7, lam=1.0)
