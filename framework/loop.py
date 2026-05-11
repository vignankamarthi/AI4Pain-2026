"""Main loop driver.

Advances the ledger by exactly ONE iteration per call. Designed to be invoked
from a Claude Code session turn, not a daemon. The mutation operator IS this
Claude Code session.

Per-iteration lifecycle (FRAMEWORK.md Section 8):

  1. Read ledger
  2. If population empty -> seed via framework.seeds, render the first
     runnable seed, return IterationPaused(HIP-D) so Vignan pushes it
  3. Else: sample parent via population.Islands.sample_parent
  4. Apply meta-stochastic state via framework.meta.step_meta_state
  5. Build mutation prompt via framework.mutation.assemble_mutation_prompt
     -> Claude Code session reads, reasons, writes a child spec
  6. Constraint check (framework.constraints)
     -> reject + regenerate if violated
  7. Render spec via framework.render
  8. Write experiments/<run_id>/, return IterationPaused(HIP-D)
  9. After cluster round-trip + result.json: caller invokes report_result,
     loop updates ledger + meta_state, returns IterationCompleted
 10. Every M iterations: trigger Level 2 introspection (HIP-H)

This first impl is intentionally narrow:
  - On empty ledger, seeds the first runnable seed and returns IterationPaused.
  - On subsequent calls, also returns IterationPaused with a synthetic run_id
    for the next program. The actual Claude-Code-driven mutation step is
    invoked by the caller from outside this module (it's a conversational
    turn, not a function call).

The integration tests exercise individual pieces; the loop driver becomes
fully autonomous once the Claude Code session wraps `advance_one_iteration`
in its per-iteration turn.
"""
from pathlib import Path

from framework import seeds, render
from framework.ledger import Ledger


class IterationOutcome:
    """Base class for what an iteration returns."""


class IterationPaused(IterationOutcome):
    """Returned when the loop hits a HIP. Caller (Claude Code) handles."""

    def __init__(self, hip: str, run_id: str, action_required: str):
        self.hip = hip
        self.run_id = run_id
        self.action_required = action_required

    def __repr__(self) -> str:
        return (f"IterationPaused(hip={self.hip!r}, run_id={self.run_id!r}, "
                f"action_required={self.action_required!r})")


class IterationCompleted(IterationOutcome):
    """Returned when the iteration has fully closed (result.json read, ledger updated)."""

    def __init__(self, run_id: str, fitness_vector: dict):
        self.run_id = run_id
        self.fitness_vector = fitness_vector

    def __repr__(self) -> str:
        return (f"IterationCompleted(run_id={self.run_id!r}, "
                f"fitness_vector={self.fitness_vector!r})")


def _first_runnable_seed(specs: list[dict]) -> dict | None:
    for s in specs:
        family = s.get("model", {}).get("family")
        if family in render.FAMILY_ENTRY_POINTS:
            return s
    return None


def advance_one_iteration(experiments_root: Path = Path("experiments"),
                          ledger_path: Path = Path("ledger/experiments.db")
                          ) -> IterationOutcome:
    """One iteration of the loop. Returns IterationPaused at HIP-D for the
    caller (Claude Code + Vignan) to advance manually."""
    experiments_root = Path(experiments_root)
    experiments_root.mkdir(parents=True, exist_ok=True)

    led = Ledger(ledger_path)
    try:
        led.init_schema()
        existing = led.get_recent_iterations(n=1)

        if not existing:
            seed_spec = _first_runnable_seed(seeds.default_seed_specs())
            if seed_spec is None:
                return IterationPaused(
                    hip="HIP-CONFIG",
                    run_id="none",
                    action_required=(
                        "no seeds have a runnable family in "
                        "framework.render.FAMILY_ENTRY_POINTS. Add at least "
                        "one family before running the loop."),
                )
            rid = led.allocate_run_id()
            run_dir = experiments_root / rid
            render.render_spec_to_code(seed_spec, run_dir)
            led.write_experiment(rid, seed_spec, parent_id=None, island_id=0)
            return IterationPaused(
                hip="HIP-D",
                run_id=rid,
                action_required=(
                    f"rsync {run_dir}/ to cluster, then sbatch with "
                    f"RUN_ID={rid}"),
            )

        # Subsequent calls: this minimal impl returns IterationPaused
        # waiting for the next manual cluster step. The full mutation loop
        # (parent select, prompt build, constraint check, render) is
        # exercised by integration tests and orchestrated by the Claude Code
        # session in turn-by-turn use.
        return IterationPaused(
            hip="HIP-D",
            run_id="next_pending",
            action_required=(
                "next-iteration generation handled by the Claude Code "
                "session turn; this minimal driver pauses at HIP-D"),
        )
    finally:
        led.close()


def report_result(run_id: str, fitness_vector: dict,
                  ledger_path: Path = Path("ledger/experiments.db")
                  ) -> IterationCompleted:
    """Caller invokes this after HIP-F brings result.json back. Updates the
    ledger row for `run_id` with the fitness vector and returns
    IterationCompleted.
    """
    led = Ledger(ledger_path)
    try:
        led.init_schema()
        led.write_result(run_id, fitness_vector)
    finally:
        led.close()
    return IterationCompleted(run_id=run_id, fitness_vector=fitness_vector)
