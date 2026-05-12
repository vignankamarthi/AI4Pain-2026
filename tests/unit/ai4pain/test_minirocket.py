"""Tests for ai4pain.minirocket.

Spec: FRAMEWORK.md §9.4 seed 5. Simplified MINIROCKET (Dempster et al. 2020,
arxiv:2012.08791). Channel-independent application, concatenated features.
Tests cover: deterministic given seed, output shape, PPV outputs in [0,1],
fit/transform contract, dispatch via run_from_dir.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ai4pain import minirocket


# ------- MiniRocket class -------


def test_minirocket_class_exists():
    assert hasattr(minirocket, "MiniRocket")


def test_minirocket_constructor_sets_attrs():
    mr = minirocket.MiniRocket(num_features=84, max_dilations_per_kernel=4,
                                random_state=42)
    assert mr.random_state == 42
    assert mr.max_dilations_per_kernel == 4


def test_minirocket_kernel_count_is_84():
    """C(9, 3) = 84 distinct ways to pick 3 positions of value=2 out of 9."""
    mr = minirocket.MiniRocket(num_features=84, random_state=42)
    kernels = mr._build_kernels()
    assert kernels.shape == (84, 1, 9)


def test_minirocket_kernels_sum_to_zero():
    """Each kernel has 3 positions of value=2 and 6 of value=-1.
    Sum: 3*2 + 6*(-1) = 0. Zero-centered."""
    mr = minirocket.MiniRocket(num_features=84, random_state=42)
    kernels = mr._build_kernels()
    sums = kernels.sum(dim=(1, 2))
    assert all(abs(float(s)) < 1e-6 for s in sums)


def test_minirocket_fit_returns_self():
    X = np.random.RandomState(0).randn(20, 200).astype(np.float32)
    mr = minirocket.MiniRocket(num_features=84, max_dilations_per_kernel=4,
                                random_state=42)
    ret = mr.fit(X)
    assert ret is mr
    assert mr._dilations is not None
    assert len(mr._dilations) >= 1


def test_minirocket_transform_output_shape_univariate():
    X = np.random.RandomState(0).randn(15, 200).astype(np.float32)
    # bias_mode="zero" for legacy: features = 84 * num_dilations
    mr = minirocket.MiniRocket(num_features=84, max_dilations_per_kernel=4,
                                random_state=42, bias_mode="zero")
    mr.fit(X)
    feats = mr.transform(X)
    assert feats.shape[0] == 15
    assert feats.shape[1] == 84 * len(mr._dilations)


def test_minirocket_quantile_bias_mode_shape():
    """bias_mode='quantile' gives 84 * num_dilations * num_biases_per_pair features."""
    X = np.random.RandomState(0).randn(15, 1000).astype(np.float32)
    mr = minirocket.MiniRocket(num_features=9996, max_dilations_per_kernel=32,
                                random_state=42, bias_mode="quantile")
    mr.fit(X)
    feats = mr.transform(X)
    expected = 84 * len(mr._dilations) * mr._num_biases_per_pair
    assert feats.shape == (15, expected)
    # And total close to num_features target
    assert abs(expected - 9996) / 9996 < 0.5


def test_minirocket_quantile_biases_diverse():
    """Quantile-based biases should produce varied PPV values, not just 0/1."""
    X = np.random.RandomState(0).randn(20, 500).astype(np.float32)
    mr = minirocket.MiniRocket(num_features=2000, max_dilations_per_kernel=8,
                                random_state=42, bias_mode="quantile")
    mr.fit(X)
    feats = mr.transform(X)
    # Mean PPV should be ~0.5 (since biases are quantile-centered)
    assert 0.3 < float(feats.mean()) < 0.7
    # And distribution should be non-trivial
    assert float(feats.std()) > 0.1


def test_minirocket_quantile_deterministic():
    X = np.random.RandomState(0).randn(20, 500).astype(np.float32)
    mr1 = minirocket.MiniRocket(num_features=1000, max_dilations_per_kernel=4,
                                 random_state=42, bias_mode="quantile")
    mr1.fit(X)
    f1 = mr1.transform(X)
    mr2 = minirocket.MiniRocket(num_features=1000, max_dilations_per_kernel=4,
                                 random_state=42, bias_mode="quantile")
    mr2.fit(X)
    f2 = mr2.transform(X)
    np.testing.assert_array_equal(f1, f2)


def test_minirocket_ppv_values_in_unit_interval():
    """PPV (Proportion of Positive Values) is a proportion -> [0, 1]."""
    X = np.random.RandomState(0).randn(10, 200).astype(np.float32)
    mr = minirocket.MiniRocket(num_features=84, max_dilations_per_kernel=4,
                                random_state=42, bias_mode="zero")
    mr.fit(X)
    feats = mr.transform(X)
    assert (feats >= 0).all()
    assert (feats <= 1).all()


def test_minirocket_deterministic_given_seed():
    X = np.random.RandomState(0).randn(10, 200).astype(np.float32)
    mr1 = minirocket.MiniRocket(num_features=84, max_dilations_per_kernel=4,
                                 random_state=42, bias_mode="zero")
    mr1.fit(X)
    f1 = mr1.transform(X)
    mr2 = minirocket.MiniRocket(num_features=84, max_dilations_per_kernel=4,
                                 random_state=42, bias_mode="zero")
    mr2.fit(X)
    f2 = mr2.transform(X)
    np.testing.assert_array_equal(f1, f2)


def test_minirocket_transform_without_fit_raises():
    mr = minirocket.MiniRocket(num_features=84, random_state=42, bias_mode="zero")
    X = np.random.randn(5, 200).astype(np.float32)
    with pytest.raises(RuntimeError):
        mr.transform(X)


def test_minirocket_handles_short_series():
    """When series is short, dilation schedule may collapse to 1 dilation."""
    X = np.random.RandomState(0).randn(10, 20).astype(np.float32)
    mr = minirocket.MiniRocket(num_features=84, max_dilations_per_kernel=4,
                                random_state=42, bias_mode="zero")
    mr.fit(X)
    feats = mr.transform(X)
    assert feats.shape[0] == 10
    assert feats.shape[1] >= 84  # at least 1 dilation


# ------- Multivariate wrapper -------


def test_multivariate_minirocket_applies_per_channel():
    """transform_multivariate concatenates per-channel features."""
    X = np.random.RandomState(0).randn(8, 200, 4).astype(np.float32)
    feats = minirocket.transform_multivariate(
        X_train=X, X_val=X[:4],
        num_features_per_channel=84, max_dilations_per_kernel=4,
        random_state=42,
    )
    # Returns (train_features, val_features); each (N, 4 * 84 * num_dilations)
    train_f, val_f = feats
    assert train_f.shape[0] == 8
    assert val_f.shape[0] == 4
    assert train_f.shape[1] == val_f.shape[1]
    # 4 channels concatenated, so feature count divisible by 4 (or close)
    assert train_f.shape[1] % 4 == 0


# ------- run_from_dir integration -------


def test_run_from_dir_reads_spec(tmp_path):
    run_dir = tmp_path / "iter_test"
    run_dir.mkdir()
    spec = {
        "name": "test_minirocket",
        "preprocessing": {"normalize": "per_channel_zscore",
                          "padding": "right_zero_to_global_max"},
        "feature_extraction": {"family": "minirocket",
                                "num_features": 84,
                                "per_channel": True,
                                "random_state": 42},
        "model": {"family": "ridge_classifier_cv",
                  "alphas": [0.1, 1.0, 10.0],
                  "class_weight": "balanced"},
        "training": {"loss": "ridge_regression_cv", "seed": 42},
        "decode": {"strategy": "argmax"},
    }
    (run_dir / "spec.json").write_text(json.dumps(spec))
    # Confirm run_from_dir is callable; smoke-test would need real data dir.
    assert callable(minirocket.run_from_dir)
