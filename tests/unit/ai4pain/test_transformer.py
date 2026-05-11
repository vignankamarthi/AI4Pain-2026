"""Tests for ai4pain.transformer (lightweight Transformer encoder)."""
import torch

from ai4pain import transformer


def test_module_imports():
    assert hasattr(transformer, "LightweightTransformer")
    assert callable(transformer.run_from_dir)


def test_transformer_forward_shape():
    model = transformer.LightweightTransformer(
        in_channels=4, d_model=64, num_heads=4, num_layers=2,
        ff_dim=128, dropout=0.0, num_classes=3,
    )
    x = torch.randn(8, 200, 4)
    out = model(x)
    assert out.shape == (8, 3)


def test_transformer_positional_encoding_shape():
    pe = transformer._sinusoidal_positional_encoding(T=100, d_model=64)
    assert pe.shape == (100, 64)


def test_transformer_positional_encoding_bounded():
    """sin/cos values are in [-1, 1]."""
    pe = transformer._sinusoidal_positional_encoding(T=200, d_model=32)
    assert (pe >= -1.0).all() and (pe <= 1.0).all()


def test_transformer_handles_short_sequence():
    model = transformer.LightweightTransformer(
        in_channels=4, d_model=32, num_heads=2, num_layers=1,
        ff_dim=64, dropout=0.0, num_classes=3,
    )
    out = model(torch.randn(2, 16, 4))
    assert out.shape == (2, 3)
