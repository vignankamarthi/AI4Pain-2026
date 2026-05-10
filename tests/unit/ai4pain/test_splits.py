"""TDD scaffold for ai4pain.splits."""
import pytest
from ai4pain import splits


def test_module_imports():
    assert callable(splits.subject_disjoint_split)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_subject_disjoint_split_no_overlap():
    train, val = splits.subject_disjoint_split(
        all_subjects=[f"S{i:02d}" for i in range(41)],
        n_val=5, seed=0,
    )
    assert set(train).isdisjoint(set(val))


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_k_subject_subset_size():
    chosen = splits.k_subject_subset(
        train_subjects=[f"S{i:02d}" for i in range(41)],
        k=10, seed=0,
    )
    assert len(chosen) == 10
