"""Integration: full constraint chain.

Verifies that rule_guards + ast_tabu + curriculum_unlock + lineage_cap
chain in the order specified by FRAMEWORK.md Section 4 and that any
single failure aborts the chain.
"""
import pytest
from framework import constraints, render


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_chain_rejects_at_first_violation(sample_program_spec):
    """Spec violates rule_guards (banned import). Chain must stop there."""
    bad = dict(sample_program_spec)
    bad["model"] = {**bad["model"], "uses_import": "transformers.from_pretrained"}
    v_rg = constraints.rule_guards(bad, max_params=10_000_000, max_train_seconds=1800)
    assert v_rg is not None
    assert v_rg.rule == "rule_guards"


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_chain_full_pass(sample_program_spec):
    """Clean spec passes all four constraints in sequence."""
    fp = render.fingerprint_spec(sample_program_spec)
    v1 = constraints.rule_guards(sample_program_spec, max_params=10_000_000, max_train_seconds=1800)
    v2 = constraints.ast_tabu(fp, recent_fingerprints=["other_a", "other_b"])
    v3 = constraints.curriculum_unlock(sample_program_spec, current_stage=2,
                                       threshold_table={"bigru": 0})
    v4 = constraints.lineage_cap(parent_lineage=["p1", "p2"], cap=5)
    assert all(v is None for v in (v1, v2, v3, v4))
