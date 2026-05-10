"""Constraint layer.

Four mechanisms (FRAMEWORK.md Section 4):

  4.1 Rule guards
      Auto-reject programs that import pretrained loaders, fetch external data,
      or exceed param/time caps. (Challenge rules, ANTIPATTERNS 1.)
  4.2 AST tabu
      Refuse near-duplicate fingerprints within last K accepted programs.
  4.3 Curriculum unlock
      Early gens locked to simple primitives. Complex unlocks above threshold.
      Gated by HIP-I.
  4.4 Lineage inbreeding cap
      Reject child whose ancestry traces > N consecutive same-parent gens.

Any violation returns a ConstraintViolation; loop regenerates with feedback.
"""
from dataclasses import dataclass


@dataclass
class ConstraintViolation:
    rule: str
    detail: str


BANNED_IMPORTS = {
    "transformers.from_pretrained",
    "torch.hub.load",
    "huggingface_hub",
    "urllib.request",
    "requests.get",
    "wget",
    "gdown",
}


def rule_guards(spec: dict, max_params: int, max_train_seconds: int) -> ConstraintViolation | None:
    """Check banned imports and resource caps. Return violation or None."""
    raise NotImplementedError


def ast_tabu(spec_fingerprint: str, recent_fingerprints: list[str]) -> ConstraintViolation | None:
    """Refuse near-duplicates. recent_fingerprints is the last K accepted hashes."""
    raise NotImplementedError


def curriculum_unlock(spec: dict, current_stage: int, threshold_table: dict) -> ConstraintViolation | None:
    """Reject if spec uses primitives not yet unlocked at current_stage."""
    raise NotImplementedError


def lineage_cap(parent_lineage: list[str], cap: int) -> ConstraintViolation | None:
    """Reject child if lineage shows > cap consecutive same-parent ancestors."""
    raise NotImplementedError
