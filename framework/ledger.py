"""SQLite ledger for the evolutionary framework.

Tables:
  experiments         run_id, parent_id, island_id, spec_json, fitness_vector, timestamps
  lineage             child_id, parent_id, mutation_type
  islands             island_id, member_run_ids, best_fitness, last_improvement_iter
  critic_population   critic_id, parent_id, critic_genome, fitness
  meta_state          iter, p_lit, novelty_alpha, temperature, failure_boost_state
  framework_mutations parent_genome_hash, child_genome_hash, description, fitness_after_M
  submissions         submission_id, run_id, slot_used, balanced_acc, timestamp

All writes are atomic. Loop must be resumable from any iteration.

Spec: FRAMEWORK.md Section 6 (meta_state), Section 7 (framework_mutations).
"""
from pathlib import Path

DEFAULT_DB_PATH = Path("ledger/experiments.db")


class Ledger:
    """SQLite-backed experiment ledger. See module docstring for schema."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        raise NotImplementedError

    def init_schema(self) -> None:
        raise NotImplementedError

    def allocate_run_id(self) -> str:
        raise NotImplementedError

    def write_experiment(self, run_id: str, spec_json: dict, parent_id, island_id: int) -> None:
        raise NotImplementedError

    def write_result(self, run_id: str, fitness_vector: dict) -> None:
        raise NotImplementedError

    def get_island_members(self, island_id: int) -> list:
        raise NotImplementedError

    def get_recent_iterations(self, n: int) -> list:
        raise NotImplementedError

    def write_framework_mutation(self, parent_hash: str, child_hash: str, desc: str) -> None:
        raise NotImplementedError
