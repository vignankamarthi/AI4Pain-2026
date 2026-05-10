"""FunSearch islands + GENITOR steady-state replacement.

FunSearch (Romera-Paredes et al., Nature 2024): M islands of K programs each,
periodic resets of bottom islands by reseeding from global champion.

GENITOR (Whitley 1988): per-island replacement rule. One child per iteration,
parent picked by rank-based tournament selection, lowest-fitness member evicted.

Spec: FRAMEWORK.md Section 2.
"""
from dataclasses import dataclass


@dataclass
class IslandState:
    island_id: int
    member_run_ids: list
    best_fitness: float
    last_improvement_iter: int


class Islands:
    """Manages M FunSearch islands with GENITOR replacement."""

    def __init__(self, m: int, k: int, reset_cadence: int):
        raise NotImplementedError

    def sample_parent(self, island_id: int, tournament_size: int = 5) -> str:
        """Rank-based tournament selection inside an island. Returns run_id."""
        raise NotImplementedError

    def insert_child(self, island_id: int, child_run_id: str, child_fitness: dict) -> None:
        """GENITOR replacement: evict lowest-fitness member of island."""
        raise NotImplementedError

    def maybe_reset_islands(self, current_iter: int, global_champion_run_id: str) -> list[int]:
        """Periodic reset of bottom islands. Returns list of reset island_ids."""
        raise NotImplementedError

    def stagnant_islands(self, current_iter: int, patience: int) -> list[int]:
        """Islands whose best has not improved in `patience` generations."""
        raise NotImplementedError
