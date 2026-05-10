"""Main loop driver.

Advances the ledger by exactly ONE iteration per call. Designed to be
invoked from a Claude Code session turn, not a daemon. The mutation
operator IS this Claude Code session.

Per-iteration sequence (FRAMEWORK.md Section 8):

  1. Read ledger
  2. Sample parent (FunSearch + GENITOR via population.Islands)
  3. Apply meta-stochastic state (meta.step_meta_state)
  4. Assemble mutation prompt (mutation.assemble_mutation_prompt)
     -> Claude Code reasons in conversation, returns child spec
  5. Constraint check (constraints.*)
     -> reject + regenerate if violated
  6. Render spec to code (render.render_spec_to_code)
  7. Write experiments/<run_id>/
  8. PAUSE for HIP-D: Vignan rsyncs to cluster
  9. PAUSE for HIP-E: Vignan sbatches
 10. PAUSE for HIP-F: Vignan rsyncs results back
 11. Read result.json, update ledger, update meta_state
 12. Critic population evolves (breakdown.CriticPopulation.evolve_one)
 13. Every M iterations: Level 2 introspection (introspect.*)
     -> PAUSE for HIP-H
"""
from pathlib import Path


class IterationOutcome:
    pass


class IterationPaused(IterationOutcome):
    """Raised when the loop hits a HIP. Caller (Claude Code session) handles."""

    def __init__(self, hip: str, run_id: str, action_required: str):
        self.hip = hip
        self.run_id = run_id
        self.action_required = action_required


class IterationCompleted(IterationOutcome):
    def __init__(self, run_id: str, fitness_vector: dict):
        self.run_id = run_id
        self.fitness_vector = fitness_vector


def advance_one_iteration(experiments_root: Path = Path("experiments"),
                          ledger_path: Path = Path("ledger/experiments.db")) -> IterationOutcome:
    """One iteration of the loop. Returns IterationPaused at each HIP, or
    IterationCompleted once all HIPs for this iteration have been satisfied.
    """
    raise NotImplementedError
