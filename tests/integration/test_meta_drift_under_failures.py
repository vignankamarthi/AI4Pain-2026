"""Integration: Level 1 meta-stochastic responds to failure regimes.

FRAMEWORK.md Section 6.2 + 3.4: when recent fitness deltas turn negative,
the failure-aware boost should raise temperature, raise novelty_alpha, raise
tabu_k, tighten lineage_cap. This integration verifies meta.step_meta_state
reads fitness.failure_aware_boost outputs and produces an updated state.
"""
import pytest
from framework import meta, fitness


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_meta_drift_widens_under_negative_deltas():
    base = {"p_lit": 0.5, "novelty_alpha": 0.3, "temperature": 0.5,
            "failure_boost_active": False, "tabu_k": 50, "lineage_cap": 5}
    out = meta.step_meta_state(current_meta=base,
                               recent_deltas=[-0.05, -0.04, -0.03, -0.05])
    assert out["temperature"] > base["temperature"]
    assert out["novelty_alpha"] > base["novelty_alpha"]


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_meta_drift_relaxes_when_improving():
    base = {"p_lit": 0.5, "novelty_alpha": 0.6, "temperature": 0.9,
            "failure_boost_active": True, "tabu_k": 80, "lineage_cap": 3}
    out = meta.step_meta_state(current_meta=base,
                               recent_deltas=[0.04, 0.03, 0.05])
    assert out["temperature"] < base["temperature"]
