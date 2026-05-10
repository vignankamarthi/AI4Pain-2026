"""TDD scaffold for ai4pain.data."""
import pytest
from ai4pain import data


def test_module_imports():
    assert callable(data.load_split)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red, awaiting HIP-A")
def test_load_split_returns_signals_and_labels(tmp_path):
    signals, labels, subjects = data.load_split(tmp_path, split="train",
                                                signals=("Bvp", "Eda", "Resp", "SpO2"))
