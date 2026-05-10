"""Integration: GENITOR steady-state replacement cycle.

FRAMEWORK.md Section 2: tournament select parent, mutate, evaluate, evict
lowest-fitness island member. This test spans population + ledger.
"""
import pytest
from framework import population, ledger


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_genitor_replacement_keeps_island_size_constant(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    isl = population.Islands(m=4, k=5, reset_cadence=100)
    initial_size = len(isl.sample_parent.__self__.member_run_ids if False else [])  # placeholder
    isl.insert_child(island_id=0, child_run_id="r_new", child_fitness={"balanced_acc": 0.7})


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_periodic_reset_reseeds_from_champion():
    isl = population.Islands(m=4, k=5, reset_cadence=10)
    reset_ids = isl.maybe_reset_islands(current_iter=10, global_champion_run_id="champ")
    assert isinstance(reset_ids, list)
