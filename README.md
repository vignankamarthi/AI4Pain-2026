# EVOLVE

A stochastic, LLM-driven evolutionary discovery framework. First case study:
the AI4Pain 2026 Grand Challenge (3-class pain localisation from peripheral
autonomic signals), targeting the PAAIn Workshop at ACII 2026 in Puebla,
Mexico.

The repository is two things at once.

**First, an AI4Pain 2026 Challenge entry.** 3-class pain *localisation* (No
Pain / Arm Pain / Hand Pain) from BVP, EDA, RESP, SpO2 physiological signals.
20 evolutionary iterations, 151 logged children, every architecture family
tried. All competitive architectures converge to a hard val 3-class plateau
at ~0.530 with tight error bars (sigma 0.007-0.016, n=20 seeds), well above
the organisers' multimodal SVM baseline of 0.398. The plateau is the
channel's information limit, not a tuning failure -- arm-vs-hand pain is
cortical information not strongly represented in peripheral autonomic
signals. Binary Pain-vs-No-Pain detection sits at 0.78-0.82.

**Second, the first case study of a stochastic evolutionary framework** the
author is developing as a longer-term research artifact. The framework
operates on three levels: Level 0 (population of candidate programs trained
per iteration), Level 1 (FunSearch islands + GENITOR replacement + multi-
objective Pareto/novelty/confidence/failure-aware scoring + rule guards +
AST tabu + lineage cap + migration + coevolutionary critic + mix-ratio drift
meta-stochastic), and Level 2 (self-introspection that mutates Level 1 on an
empirical compound-detector cadence). The mutation operator is a Claude Code
session, not an external API client. See `FRAMEWORK.md` for the full design;
`MODELS_FRAMEWORK.html` for per-architecture model details (diagrams + math).

## Final submission (HIP-G, 22 May 2026)

Portfolio of 4 architecturally diverse component models + 1 weighted
prediction ensemble. Each component trained with `n_seeds=20`: 20 independent
trained models per spec, probability tables averaged before argmax, error
bars reported via per-seed bundle ensembling. Final val numbers:

| # | architecture | val 3-class (n=20) | val binary |
|---|--------------|--------------------|------------|
| 1 | spectrogram CNN (2D STFT) | 0.5292 +/- 0.0108 | 0.7801 |
| 2 | multi-stream BiGRU (1D temporal) | 0.5274 +/- 0.0071 | 0.8194 |
| 3 | dual ensemble (joint 2D+1D) | 0.5315 +/- 0.0086 | 0.8032 |
| 4 | residual spectrogram CNN | 0.5248 +/- 0.0080 | 0.7847 |
| **5** | **weighted ensemble [1,1,1,3] (bundle)** | **0.5367 +/- 0.0157** | **0.7874 +/- 0.0175** |

Ensemble #5 beats every individual component on val 3-class. Test 3-class
accuracies pending portal returns (recorded in `SUBMISSIONS.md`).

## Repo layout

```
framework/             Three-level evolutionary framework (model-agnostic)
ai4pain/               AI4Pain-specific model families, data loaders, splits, metrics
  submission.py        Multi-seed train+predict runner (per-seed probability tables)
  ensemble_submission  Weighted soft-vote + per-seed bundle ensembling
  portal_format.py     test_predictions.csv -> emailed TeamName_Version.csv
scripts/               Manual cluster helpers (Vignan runs by hand)
  run_array.slurm      Iteration array job (per-batch evolutionary search)
  run_submission.slurm Single-task submission runner (4h H200, .venv 3.11)
tests/                 Mirror layout: unit/framework/, unit/ai4pain/, integration/
experiments/           Per-run artifact directories (iter_NNNN/, submission_0N/)
ledger/                SQLite database of all child runs (gitignored)
data/                  Raw challenge data (gitignored)
portal_submissions/    Emailable TeamName_Version.csv files (gitignored)
```

Static identity: `CLAUDE.md`. Dynamic memory: `MEMORY.md`. Hard rules:
`ANTIPATTERNS.md`. Architecture: `FRAMEWORK.md`. Project narrative + ledger:
`PLAN.md`. Submission portfolio: `SUBMISSIONS.md`. Per-architecture details:
`MODELS_FRAMEWORK.html`.

## Hard constraints

- No external data (challenge rule).
- 5 test submissions max, each requiring Vignan's explicit approval.
- No subject leakage; per-subject splits only; preprocessing fits on train only.
- No commits or pushes from Claude (`/Commit-Initiation` plans only).
- No programmatic cluster invocations from framework Python (Vignan runs
  `rsync`/`ssh`/`sbatch`/`git pull` by hand).
- No external LLM SDKs (`anthropic`, `openai`, etc. NOT in `requirements.txt`).
  The mutation operator is the Claude Code session driving this repo.
- Strict TDD: failing test in `tests/...` before any implementation.

## Hardware

NEU Explorer cluster, Python 3.11 venv, torch 2.5.1+cu121, 1x H200 GPU per
job, 4h SLURM time limit for submissions, 8 concurrent job slots.
