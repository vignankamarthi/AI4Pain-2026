"""Program-spec to runnable code renderer.

A program spec is a JSON-like AST describing the full pipeline:
  preprocessing -> feature extraction (optional) -> model -> training -> decode

Renderer translates the spec into a runnable Python module that defines:
  - build_model() -> torch.nn.Module
  - get_dataloaders(data_root, split_subjects) -> (train, val)
  - train_one_epoch(model, dl, optim, ...) -> metrics
  - predict(model, dl) -> (logits, labels)

Spec: FRAMEWORK.md Section 2.
"""
from pathlib import Path


def render_spec_to_code(spec: dict, out_dir: Path) -> Path:
    """Render `spec` into a runnable PyTorch + sklearn module at `out_dir/run.py`.

    Returns the path to the generated module.
    """
    raise NotImplementedError


def fingerprint_spec(spec: dict) -> str:
    """Structural hash of a spec, ignoring variable names and literals.
    Used by AST tabu (constraints.py).
    """
    raise NotImplementedError
