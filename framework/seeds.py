"""Multi-seed initialization.

Pulls 4-6 distinct architectures from literature for the initial population
across islands. Gated by HIP-C: Vignan reviews and approves the list before
the first iteration.

Default candidates (placeholder, finalize during Level 1 implementation):
  - 1D-CNN baseline (ResNet-1D-style)
  - BiGRU sequence model
  - Lightweight Transformer encoder
  - Catch22 + gradient-boosted trees (classical ML branch)
  - Multi-stream attention (BVP / EDA / RESP / SpO2 separate then fused)
  - Hybrid CNN + GRU

Each seed becomes one initial population in one island. Remaining islands
seed by mutating these.

Spec: FRAMEWORK.md Section 9, open question 4.
"""


def default_seed_specs() -> list[dict]:
    """Return list of seed program specs (placeholder structures).
    Final list is approved at HIP-C.
    """
    raise NotImplementedError


def diversify_population(seed_specs: list[dict], island_count: int) -> list[list[dict]]:
    """Distribute seeds across islands. Islands with fewer seeds than members
    are filled by single-mutation perturbations of their seed.
    """
    raise NotImplementedError
