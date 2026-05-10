"""TDD scaffold for framework.seeds. Spec: FRAMEWORK.md Section 9 Open Q 4."""
import pytest
from framework import seeds


def test_module_imports():
    assert callable(seeds.default_seed_specs)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_default_seed_specs_returns_diverse_list():
    specs = seeds.default_seed_specs()
    families = {s["model"]["family"] for s in specs}
    assert len(families) >= 4


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_diversify_population_distributes_across_islands():
    specs = seeds.default_seed_specs()
    distributed = seeds.diversify_population(specs, island_count=8)
    assert len(distributed) == 8
