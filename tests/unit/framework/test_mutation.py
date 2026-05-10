"""TDD scaffold for framework.mutation. Spec: FRAMEWORK.md Section 2 + 6."""
import pytest
from framework import mutation


def test_module_imports():
    assert mutation.MetaState is not None


def test_metastate_dataclass_fields():
    m = mutation.MetaState(p_lit=0.5, novelty_alpha=0.3, temperature=0.7,
                           failure_boost_active=False)
    assert m.p_lit == 0.5
    assert m.failure_boost_active is False


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_assemble_mutation_prompt_returns_str(sample_program_spec):
    meta = mutation.MetaState(p_lit=0.5, novelty_alpha=0.3, temperature=0.7,
                              failure_boost_active=False)
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[sample_program_spec],
        recent_failures=[],
        meta=meta,
    )
    assert isinstance(out, str)
    assert "## Parent program" in out
    assert "## Mutation directive" in out
