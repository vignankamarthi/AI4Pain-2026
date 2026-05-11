"""Tests for framework.mutation. Spec: FRAMEWORK.md Section 2 + 6."""
import pytest
from framework import mutation


def test_module_imports():
    assert mutation.MetaState is not None
    assert callable(mutation.assemble_mutation_prompt)


def test_metastate_dataclass_fields():
    m = mutation.MetaState(p_lit=0.5, novelty_alpha=0.3, temperature=0.7,
                           failure_boost_active=False)
    assert m.p_lit == 0.5
    assert m.failure_boost_active is False


def _baseline_meta(**overrides) -> mutation.MetaState:
    defaults = dict(p_lit=0.5, novelty_alpha=0.3, temperature=0.7,
                    failure_boost_active=False)
    defaults.update(overrides)
    return mutation.MetaState(**defaults)


def test_prompt_contains_all_required_sections(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[sample_program_spec],
        recent_failures=[],
        meta=_baseline_meta(),
    )
    assert isinstance(out, str)
    for section in (
        "## Parent program",
        "## Best in island",
        "## Recent rejected programs",
        "## Meta-stochastic state",
        "## Mutation directive",
    ):
        assert section in out


def test_prompt_serializes_parent_spec_as_json(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(),
    )
    assert '"family": "bigru"' in out


def test_directive_aggressive_when_failure_boost_active(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(failure_boost_active=True),
    )
    assert "AGGRESSIVE" in out


def test_directive_literature_bias_when_p_lit_high(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(p_lit=0.8),
    )
    assert "literature-derived" in out


def test_directive_novel_when_p_lit_low(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(p_lit=0.25),
    )
    assert "NOVEL" in out


def test_empty_inputs_do_not_crash(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(),
    )
    assert "(empty)" in out
    assert "(none recorded)" in out


def test_meta_state_values_appear_in_prompt(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(p_lit=0.42, novelty_alpha=0.61, temperature=1.1),
    )
    assert "0.420" in out
    assert "0.610" in out
    assert "1.100" in out
