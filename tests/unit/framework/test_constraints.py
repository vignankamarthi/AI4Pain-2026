"""TDD scaffold for framework.constraints. Spec: FRAMEWORK.md Section 4."""
import pytest
from framework import constraints


def test_module_imports():
    assert "transformers.from_pretrained" in constraints.BANNED_IMPORTS


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_rule_guards_rejects_pretrained_loader(sample_program_spec):
    bad = dict(sample_program_spec)
    bad["model"] = {**bad["model"], "uses_import": "transformers.from_pretrained"}
    v = constraints.rule_guards(bad, max_params=10_000_000, max_train_seconds=1800)
    assert v is not None


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_rule_guards_passes_clean_spec(sample_program_spec):
    v = constraints.rule_guards(sample_program_spec, max_params=10_000_000,
                                max_train_seconds=1800)
    assert v is None


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_ast_tabu_rejects_duplicate_fingerprint():
    v = constraints.ast_tabu("hash_x", recent_fingerprints=["hash_a", "hash_x", "hash_b"])
    assert v is not None


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_curriculum_unlock_blocks_advanced_at_stage_zero(sample_program_spec):
    spec = dict(sample_program_spec)
    spec["model"] = {**spec["model"], "family": "transformer"}
    v = constraints.curriculum_unlock(spec, current_stage=0,
                                      threshold_table={"transformer": 1})
    assert v is not None


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_lineage_cap_rejects_inbred_chain():
    chain = ["p", "p", "p", "p", "p", "p"]
    v = constraints.lineage_cap(chain, cap=5)
    assert v is not None
