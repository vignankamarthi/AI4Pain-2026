"""TDD scaffold for framework.render. Spec: FRAMEWORK.md Section 2."""
import pytest
from pathlib import Path
from framework import render


def test_module_imports():
    assert callable(render.render_spec_to_code)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_render_writes_runnable_module(tmp_path: Path, sample_program_spec):
    out = render.render_spec_to_code(sample_program_spec, tmp_path)
    assert out.exists()
    assert out.suffix == ".py"


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_fingerprint_is_structural(sample_program_spec):
    h1 = render.fingerprint_spec(sample_program_spec)
    spec2 = dict(sample_program_spec)
    spec2["model"] = {**spec2["model"], "hidden_size": 64}  # same structure
    assert render.fingerprint_spec(spec2) == h1
