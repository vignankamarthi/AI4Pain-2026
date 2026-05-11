"""Tests for ai4pain.multi_stream (Multi-stream BiGRU)."""
import torch
import pytest

from ai4pain import multi_stream


def test_module_imports():
    assert hasattr(multi_stream, "MultiStreamBiGRU")
    assert callable(multi_stream.run_from_dir)


def test_multi_stream_forward_shape():
    model = multi_stream.MultiStreamBiGRU(
        in_channels=4, per_channel_hidden=16, per_channel_layers=1,
        fusion="late_concat", fusion_dropout=0.0, num_classes=3,
    )
    x = torch.randn(8, 200, 4)
    out = model(x)
    assert out.shape == (8, 3)


def test_multi_stream_has_one_encoder_per_channel():
    model = multi_stream.MultiStreamBiGRU(in_channels=4, per_channel_hidden=8)
    assert len(model.encoders) == 4


def test_multi_stream_rejects_unknown_fusion():
    with pytest.raises(ValueError):
        multi_stream.MultiStreamBiGRU(in_channels=4, fusion="early_concat")


def test_multi_stream_handles_short_sequence():
    model = multi_stream.MultiStreamBiGRU(
        in_channels=4, per_channel_hidden=8, per_channel_layers=1,
        fusion="late_concat", fusion_dropout=0.0, num_classes=3,
    )
    out = model(torch.randn(2, 20, 4))
    assert out.shape == (2, 3)
