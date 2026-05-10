"""TDD scaffold for framework.introspect. Spec: FRAMEWORK.md Section 7."""
import pytest
from framework import introspect


def test_module_imports():
    assert "island_count" in introspect.GENOME_RULE_GUARDS


def test_genome_rule_guards_have_sane_bounds():
    lo, hi = introspect.GENOME_RULE_GUARDS["island_count"]
    assert 1 <= lo < hi <= 64
    lo, hi = introspect.GENOME_RULE_GUARDS["introspection_cadence_M"]
    assert lo >= 10


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_assemble_introspection_prompt_returns_structured_blob():
    out = introspect.assemble_introspection_prompt(
        ledger_recent=[],
        current_genome={"island_count": 8},
        m_iter_window=50,
    )
    assert isinstance(out, str)
    assert "## Recent fitness trajectory" in out
    assert "## Current genome" in out


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_validate_genome_mutation_rejects_out_of_bounds():
    err = introspect.validate_genome_mutation(
        proposed={"island_count": 100},
        current={"island_count": 8},
    )
    assert err is not None


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_apply_genome_mutation_returns_new_genome():
    mutation = introspect.GenomeMutation(
        parent_hash="a", child_hash="b", description="test",
        parameter_changes={"island_count": 10},
        operator_changes={},
    )
    new_genome = introspect.apply_genome_mutation(mutation, current_genome={"island_count": 8})
    assert new_genome["island_count"] == 10
