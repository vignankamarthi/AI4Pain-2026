# EVOLVE

A stochastic, LLM-driven evolutionary discovery framework.

This repository is two things at once.

First, an entry to the **AI4Pain 2026 Challenge**. The task is 3-class pain *localization* (No Pain, Hand Pain, Arm Pain) from peripheral physiological signals (BVP, EDA, RESP, SpO2). Target venue is the PAAIn Workshop at ACII 2026 in Puebla, Mexico. The task is hard: published baselines sit in the 50-60% balanced-accuracy range, partly because pain localization is best inferred from cortical signals (EEG/scalp) and the available channels are autonomic. The benchmark is therefore inherently flawed and any winning model must extract every bit of signal from a low-information channel set.

Second, the **first case study of a stochastic evolutionary discovery framework** that the author is developing as a longer-term research artifact. The framework operates on three levels: Level 0 (a population of candidate programs trained per iteration), Level 1 (FunSearch islands + GENITOR replacement + multi-objective scoring + exploration constraints + breakdown mechanisms + mix-ratio drift), and Level 2 (a self-introspection mechanism that mutates Level 1 on a slow timescale, every ~50 iterations). The mutation operator is a Claude Code session, not a separate API client. The framework is documented in `FRAMEWORK.md` and is not yet a standalone paper.

## Repo layout

```
framework/    Three-level evolutionary framework (model-agnostic)
ai4pain/      AI4Pain-specific data loaders, splits, metrics
scripts/      Manual cluster helpers (Vignan runs by hand)
experiments/  Per-run artifact directories (gitignored)
ledger/       SQLite database (gitignored)
data/         Raw challenge data (gitignored)
papers/       Paper drafts (gitignored)
```

Static identity: `CLAUDE.md`. Dynamic state: `MEMORY.md`. Hard rules: `ANTIPATTERNS.md`. Architecture: `FRAMEWORK.md`. Plan: `~/.claude/plans/first-plan-then-ill-jolly-penguin.md`.

## Status

Scaffold only. All `framework/` and `ai4pain/` modules are stubs. Implementation begins after AI4Pain data is downloaded (HIP-A). See `FRAMEWORK.md` Section 9 for open design questions and `MEMORY.md` for current state.

## Hard constraints

- No external data, no pretrained weights (challenge rule).
- 5 test-set submissions max across project lifetime, gated by HIP-G.
- All cluster operations are manual via HIP-D / HIP-E / HIP-F.
- Mutation operator is the Claude Code session, no external LLM SDKs.
- See `ANTIPATTERNS.md` for the full rule set.
