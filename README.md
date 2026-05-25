# AI4Pain-2026

Entry to the AI4Pain 2026 Grand Challenge (3-class pain localisation from
peripheral autonomic signals), targeting the PAAIn Workshop at ACII 2026 in
Puebla, Mexico. The submission was produced by applying a multi-level
**Stochastic Experiment Loop** framework -- a stochastic, LLM-driven
evolutionary discovery loop -- to this task as its first case study.
The framework itself (under `framework/`, methodology in `FRAMEWORK.html`) is
preserved here as scientific provenance and as a copyable reference for
applying the same loop to future domains/projects.

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
session, not an external API client. See `FRAMEWORK.html` for the full design;
`MODELS_FRAMEWORK.html` for per-architecture model details (diagrams + math).

## Final submission (HIP-G, 22 May 2026)

Portfolio of 4 architecturally diverse component models + 1 weighted
prediction ensemble. Each component trained with `n_seeds=20`: 20 independent
trained models per spec, probability tables averaged before argmax, error
bars reported via per-seed bundle ensembling. Final val numbers:

| # | architecture | val 3-class (n=20) | val binary | test 3-class |
|---|--------------|--------------------|------------|--------------|
| 1 | spectrogram CNN (2D STFT) | 0.5292 +/- 0.0110 | 0.7653 +/- 0.0304 | 0.4884 |
| 2 | multi-stream BiGRU (1D temporal) | 0.5274 +/- 0.0073 | 0.7972 +/- 0.0146 | 0.4653 |
| 3 | dual ensemble (joint 2D+1D) | 0.5315 +/- 0.0088 | 0.7733 +/- 0.0210 | 0.4537 |
| 4 | residual spectrogram CNN | 0.5248 +/- 0.0082 | 0.7707 +/- 0.0129 | 0.4815 |
| **5** | **weighted ensemble [1,1,1,3] (bundle)** | **0.5367 +/- 0.0161** | **0.7874 +/- 0.0179** | **0.4861** |

Ensemble #5 beats every individual component on val 3-class. **5 portal CSVs
emailed to Raul Fernandez-Rojas on 2026-05-22; portal test accuracies returned
2026-05-23** (cluster 0.454-0.488, recorded in `SUBMISSIONS.md`). Sub 1 ranks
first on the portal at 0.4884; the val/test rank flipped at the top by 0.0023
(within the bundle std).

## Paper scope (PAAIn @ ACII 2026)

The AI4Pain 2026 paper leverages the pain detection / localization findings
documented in this repo (binary ~0.77-0.80 val vs 3-class ~0.53 val plateau;
the information-limit thesis). **The paper does NOT discuss the Stochastic
Experiment Loop**; the framework is preserved here only as a research artifact
for application to future projects. `FRAMEWORK.html` and `MODELS_FRAMEWORK.html`
document the framework + models for future reference and as paper-supporting
detail on the architectures, not as the paper's subject. Paper title:
**"Architecturally Diverse Ensembling at the Peripheral-Signal Ceiling for
Pain Localization"**. Drafted 2026-05-24, submission-ready as of 2026-05-25,
PDF at `papers/ai4pain_2026/main.pdf`. Paper deadline: 2026-05-30 via EasyChair.

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
`ANTIPATTERNS.md`. Architecture: `FRAMEWORK.html`. Project narrative + ledger:
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
