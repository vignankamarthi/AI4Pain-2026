"""TDD scaffold for framework.population. Spec: FRAMEWORK.md Section 2."""
import pytest
from framework import population


def test_module_imports():
    assert population.IslandState is not None


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_islands_construction():
    population.Islands(m=8, k=10, reset_cadence=100)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_sample_parent_returns_island_member():
    isl = population.Islands(m=8, k=10, reset_cadence=100)
    rid = isl.sample_parent(island_id=0, tournament_size=5)
    assert isinstance(rid, str)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_genitor_evicts_lowest_fitness():
    isl = population.Islands(m=8, k=10, reset_cadence=100)
    isl.insert_child(island_id=0, child_run_id="r_new", child_fitness={"balanced_acc": 0.7})


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_stagnant_islands_detected():
    isl = population.Islands(m=8, k=10, reset_cadence=100)
    stale = isl.stagnant_islands(current_iter=200, patience=10)
    assert isinstance(stale, list)
