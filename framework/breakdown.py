"""Breakdown layer: unsticking stagnant mechanisms.

Three mechanisms (FRAMEWORK.md Section 5):

  5.1 Inter-island migration
      When an island's best flatlines, mate its champion with a foreign
      island's champion via FunSearch's prompt-based crossover semantic.
  5.2 Coevolutionary critic (Hillis 1990)
      Separate evolving population of (subject_subset, signal_perturbation,
      channel_permutation) triples that maximize program failure rates.
      Real arms race, not just monotone descent.
  5.3 Stagnation -> escalating temperature
      Per-island detector. Raise temp, novelty alpha, force structural
      mutations when patience exceeded.
"""
from dataclasses import dataclass


@dataclass
class CriticIndividual:
    critic_id: str
    subject_subset: list[str]
    signal_perturbation: dict
    channel_permutation: list[int]
    fitness: float


def trigger_migration(stagnant_island_id: int, champion_run_id: str,
                      foreign_champion_run_id: str) -> dict:
    """Construct a crossover spec from two champions. Returns child spec."""
    raise NotImplementedError


class CriticPopulation:
    """Hillis-style coevolutionary critic population. Lives in ledger.critic_population."""

    def __init__(self, size: int):
        raise NotImplementedError

    def evolve_one(self, program_population_failures: dict) -> CriticIndividual:
        """Evolve one critic individual. Fitness = mean failure rate against programs."""
        raise NotImplementedError

    def hardest_critics(self, n: int) -> list[CriticIndividual]:
        """Return top-n critics by failure-induction rate. Used in eval."""
        raise NotImplementedError


def stagnation_escalation(island_id: int, patience: int, current_meta: dict) -> dict:
    """Return updated meta_state with raised temp / alpha for the stagnant island."""
    raise NotImplementedError
