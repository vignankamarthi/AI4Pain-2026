"""Mutation prompt-context assembler.

NOT an LLM SDK client. The mutation operator is the Claude Code session that
drives this repo. This module assembles a structured Markdown blob that the
assistant reads conversationally each iteration to generate the next program
spec. ANTIPATTERNS rule 11.

Spec: FRAMEWORK.md Section 2 (mutation operator), Section 6 (mix-ratio).
"""
from dataclasses import dataclass
import json


@dataclass
class MetaState:
    p_lit: float                 # literature-vs-novel mix ratio in [0.2, 0.8]
    novelty_alpha: float         # weight on novelty in scoring
    temperature: float           # LLM prompt temperature target
    failure_boost_active: bool   # global failure-aware boost flag


def _format_spec(spec: dict) -> str:
    return "```json\n" + json.dumps(spec, indent=2, sort_keys=True) + "\n```"


def _format_island_bests(specs: list[dict]) -> str:
    if not specs:
        return "(empty)"
    return "\n\n".join(
        f"#{i + 1}\n{_format_spec(s)}" for i, s in enumerate(specs[:5])
    )


def _format_recent_failures(failures: list[dict]) -> str:
    if not failures:
        return "(none recorded)"
    lines = []
    for i, f in enumerate(failures[-10:], start=1):
        if isinstance(f, dict):
            lines.append(f"- {i}. spec={json.dumps(f, sort_keys=True)[:200]}")
        else:
            lines.append(f"- {i}. {f}")
    return "\n".join(lines)


def _mutation_directive(meta: MetaState) -> str:
    if meta.failure_boost_active:
        return (
            "Recent children have degraded fitness vs the parent island's best. "
            "Failure-aware boost is ACTIVE. Propose an AGGRESSIVE structural "
            "mutation, not a hyperparameter tweak. Change the model family, "
            "change the preprocessing pipeline shape, swap a major operator. "
            f"Operate near temperature {meta.temperature:.2f}, novelty_alpha "
            f"{meta.novelty_alpha:.2f}. Eligible model families are those "
            "registered in framework.render.FAMILY_ENTRY_POINTS."
        )
    if meta.p_lit >= 0.5:
        return (
            f"Mix ratio p_lit={meta.p_lit:.2f} biases toward literature-derived "
            "mutations. Recall what the most-cited recent paper on AI4Pain "
            "peripheral signal classification (BVP/EDA/RESP/SpO2 -> pain class) "
            "would do as a structural change from the parent. Apply that "
            "change to the parent spec, leaving the rest intact."
        )
    return (
        f"Mix ratio p_lit={meta.p_lit:.2f} biases toward NOVEL cross-domain "
        "analogy. Draw from physics, chemistry, biology, or a non-ML field. "
        "Propose a structural mutation that would surprise a typical ML "
        f"reviewer. novelty_alpha={meta.novelty_alpha:.2f}, temperature="
        f"{meta.temperature:.2f}."
    )


def assemble_mutation_prompt(parent_spec: dict, island_best_specs: list[dict],
                             recent_failures: list[dict],
                             meta: MetaState) -> str:
    """Return a structured Markdown blob the Claude Code session reads.

    Sections of the returned blob:
      ## Parent program
      ## Best in island (top 5)
      ## Recent rejected programs (so we avoid re-proposing them)
      ## Meta-stochastic state (p_lit, novelty_alpha, temperature, failure_boost)
      ## Mutation directive

    The mutation directive is selected from three regimes based on `meta`:
    failure-boost-active (aggressive), p_lit >= 0.5 (literature-biased),
    else (novel cross-domain).
    """
    return (
        "# Mutation prompt\n\n"
        "Read every section. Output a single JSON spec for the child program. "
        "Constraints are enforced AFTER you propose, so prefer correctness over "
        "minimal change.\n\n"
        "## Parent program\n\n"
        f"{_format_spec(parent_spec)}\n\n"
        "## Best in island (top up to 5, ordered)\n\n"
        f"{_format_island_bests(island_best_specs)}\n\n"
        "## Recent rejected programs (last up to 10)\n\n"
        f"{_format_recent_failures(recent_failures)}\n\n"
        "## Meta-stochastic state\n\n"
        f"- p_lit: {meta.p_lit:.3f}\n"
        f"- novelty_alpha: {meta.novelty_alpha:.3f}\n"
        f"- temperature: {meta.temperature:.3f}\n"
        f"- failure_boost_active: {meta.failure_boost_active}\n\n"
        "## Mutation directive\n\n"
        f"{_mutation_directive(meta)}\n"
    )
