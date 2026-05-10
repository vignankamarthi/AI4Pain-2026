"""TDD scaffold for ai4pain.metrics."""
import pytest
import numpy as np
from ai4pain import metrics


def test_module_imports():
    assert callable(metrics.full_metric_suite)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_full_metric_suite_keys_present():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    proba = np.eye(3)[y_pred]
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    for k in ("balanced_acc", "macro_f1", "confusion_3x3", "auc_ovr", "ece",
              "per_class_pr"):
        assert k in out


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_balanced_acc_random_baseline_is_one_third():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = rng.integers(0, 3, size=300)
    proba = np.eye(3)[y_pred]
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    assert 0.25 < out["balanced_acc"] < 0.42
