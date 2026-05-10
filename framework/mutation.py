"""Mutation prompt-context assembler.

NOT an LLM SDK client. The mutation operator is the Claude Code session
that drives this repo. This module assembles a structured Markdown blob
that the assistant reads conversationally each iteration to generate the
next program spec.

Spec: FRAMEWORK.md Section 2 (mutation operator), Section 6 (mix-ratio).
"""
from dataclasses import dataclass


@dataclass
class MetaState:
    p_lit: float                 # literature-vs-novel mix ratio in [0.2, 0.8]
    novelty_alpha: float         # weight on novelty in scoring
    temperature: float           # LLM prompt temperature target
    failure_boost_active: bool   # global failure-aware boost flag


def assemble_mutation_prompt(parent_spec: dict, island_best_specs: list,
                             recent_failures: list, meta: MetaState) -> str:
    """Return a structured Markdown blob the Claude Code session reads.

    Sections of the returned blob:
      ## Parent program
      ## Best in island (top 3)
      ## Recent rejected programs (so we avoid re-proposing them)
      ## Meta-stochastic state (p_lit, novelty_alpha, temperature, failure_boost)
      ## Mutation directive
        if p_lit > 0.5: bias toward "what would the most-cited paper for this
                        signal class do as a structural mutation here"
        else:           bias toward "what is a novel structural mutation that
                        would surprise me, drawing from cross-domain analogy"
        if failure_boost_active: explicitly call for aggressive structural change
    """
    raise NotImplementedError
