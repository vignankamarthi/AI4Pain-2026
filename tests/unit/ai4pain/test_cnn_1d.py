"""Tests for ai4pain.cnn_1d (1D-CNN ResNet)."""
import torch
import pytest

from ai4pain import cnn_1d


def test_module_imports():
    assert hasattr(cnn_1d, "CNN1DResNet")
    assert callable(cnn_1d.run_from_dir)


def test_cnn1d_forward_shape():
    model = cnn_1d.CNN1DResNet(in_channels=4, depth=2, base_channels=16,
                                kernel_size=7, use_residual=True,
                                num_classes=3)
    x = torch.randn(8, 200, 4)  # (B, T, C)
    out = model(x)
    assert out.shape == (8, 3)


def test_cnn1d_param_count_scales_with_depth():
    shallow = cnn_1d.CNN1DResNet(in_channels=4, depth=1, base_channels=16)
    deep = cnn_1d.CNN1DResNet(in_channels=4, depth=8, base_channels=16)
    n_shallow = sum(p.numel() for p in shallow.parameters())
    n_deep = sum(p.numel() for p in deep.parameters())
    assert n_deep > n_shallow


def test_cnn1d_residual_toggle_works():
    """use_residual=False should still produce valid forward pass."""
    model = cnn_1d.CNN1DResNet(in_channels=4, depth=2, use_residual=False)
    x = torch.randn(2, 100, 4)
    out = model(x)
    assert out.shape == (2, 3)


def test_cnn1d_handles_variable_length():
    """Global avg pool means any T works."""
    model = cnn_1d.CNN1DResNet(in_channels=4, depth=2, base_channels=16)
    for T in (50, 100, 500):
        out = model(torch.randn(4, T, 4))
        assert out.shape == (4, 3)
