"""TDD scaffold for framework.ledger. Spec: FRAMEWORK.md Section 10."""
import pytest
from framework import ledger


def test_module_imports():
    assert ledger.DEFAULT_DB_PATH.name == "experiments.db"


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_ledger_init_creates_schema(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_allocate_run_id_returns_unique(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    a, b = led.allocate_run_id(), led.allocate_run_id()
    assert a != b


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_write_experiment_round_trip(tmp_db_path, sample_program_spec, sample_fitness_vector):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    rid = led.allocate_run_id()
    led.write_experiment(rid, sample_program_spec, parent_id=None, island_id=0)
    led.write_result(rid, sample_fitness_vector)
    members = led.get_island_members(0)
    assert rid in [m["run_id"] for m in members]


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_framework_mutation_logged(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    led.write_framework_mutation("hash_a", "hash_b", "raise curriculum_threshold 0.55 -> 0.60")
