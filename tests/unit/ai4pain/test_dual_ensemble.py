"""Tests for ai4pain.dual_ensemble (family: dual_ensemble).

iter_0015-0017 produced two co-champions from different architecture
families that plateaued at ~0.536: a 1D multi_stream GRU and a 2D
spectrogram CNN. Their per-class profiles diverge (spectrogram strong on
HP, GRU on AP) -> decorrelated errors -> an ensemble should clear the
plateau. `DualEnsembleNet` trains both sub-models jointly and fuses their
logits with a LEARNED per-class blend weight.

forward(x_seq, x_spec) -> logits. x_seq is the raw (B,T,C) sequence for the
GRU; x_spec is the (B,C,F,T') spectrogram stack for the CNN.
"""
import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
from ai4pain import dual_ensemble


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()


def test_module_imports():
    assert hasattr(dual_ensemble, "DualEnsembleNet")
    assert callable(dual_ensemble.train_dual_ensemble)
    assert callable(dual_ensemble.run_from_dir)


def test_dual_ensemble_forward_shape():
    model = dual_ensemble.DualEnsembleNet(
        in_channels=4, spec_F=33, num_classes=3,
        gru_cfg={"per_channel_hidden": 16},
        cnn_cfg={"base_channels": 8, "depth": 2})
    x_seq = torch.randn(6, 120, 4)
    x_spec = torch.randn(6, 4, 33, 20)
    out = model(x_seq, x_spec)
    assert out.shape == (6, 3)


def test_dual_ensemble_backward_no_nan():
    model = dual_ensemble.DualEnsembleNet(
        in_channels=4, spec_F=33, num_classes=3,
        gru_cfg={"per_channel_hidden": 16},
        cnn_cfg={"base_channels": 8, "depth": 1})
    x_seq = torch.randn(4, 64, 4)
    x_spec = torch.randn(4, 4, 33, 12)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x_seq, x_spec), y)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None or torch.isfinite(p.grad).all()


def test_blend_weight_is_per_class_and_learnable():
    """The fusion weight must be a 3-vector (one per class) and require grad."""
    model = dual_ensemble.DualEnsembleNet(
        in_channels=4, spec_F=33, num_classes=3,
        gru_cfg={"per_channel_hidden": 8}, cnn_cfg={"base_channels": 8})
    assert model.blend.shape == (3,)
    assert model.blend.requires_grad


def test_blend_weights_bounded_0_1():
    """The effective per-class blend (sigmoid of the raw param) is in (0,1)."""
    model = dual_ensemble.DualEnsembleNet(
        in_channels=4, spec_F=33, num_classes=3,
        gru_cfg={"per_channel_hidden": 8}, cnn_cfg={"base_channels": 8})
    w = model.blend_weights()
    assert w.shape == (3,)
    assert torch.all(w > 0.0) and torch.all(w < 1.0)


def test_dual_ensemble_holds_both_submodels():
    model = dual_ensemble.DualEnsembleNet(
        in_channels=4, spec_F=33, num_classes=3,
        gru_cfg={"per_channel_hidden": 8}, cnn_cfg={"base_channels": 8})
    # both component nets are real submodules with parameters
    assert sum(p.numel() for p in model.gru.parameters()) > 0
    assert sum(p.numel() for p in model.cnn.parameters()) > 0


def test_fusion_is_convex_blend_of_component_logits():
    """output = w*cnn_logits + (1-w)*gru_logits, per class. Verify by
    forcing the component outputs via monkeypatched sub-nets."""
    model = dual_ensemble.DualEnsembleNet(
        in_channels=4, spec_F=33, num_classes=3,
        gru_cfg={"per_channel_hidden": 8}, cnn_cfg={"base_channels": 8})
    model.eval()
    gru_out = torch.tensor([[1.0, 2.0, 3.0]])
    cnn_out = torch.tensor([[3.0, 2.0, 1.0]])
    model.gru.forward = lambda x: gru_out
    model.cnn.forward = lambda x: cnn_out
    with torch.no_grad():
        w = model.blend_weights()
        out = model(torch.randn(1, 10, 4), torch.randn(1, 4, 33, 5))
        expected = w * cnn_out + (1 - w) * gru_out
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_smoke_train_dual_ensemble_writes_result(tmp_path: Path):
    spec = {
        "name": "smoke_dual",
        "preprocessing": {"normalize": "per_channel_zscore",
                           "padding": "right_zero_to_global_max"},
        "feature_extraction": {"family": "dual_ensemble", "fs": 100,
                                "nperseg": 64, "noverlap": 32,
                                "log_scale": True},
        "model": {"family": "dual_ensemble",
                   "gru_cfg": {"per_channel_hidden": 8},
                   "cnn_cfg": {"base_channels": 8, "depth": 1}},
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0,
                     "loss": "ce_class_balanced", "optimizer": "adam"},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
        "decode": {"strategy": "argmax"},
    }
    result = dual_ensemble.train_dual_ensemble(
        spec, data_root=DATA_ROOT, out_dir=tmp_path)
    assert (tmp_path / "result.json").exists()
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert 0.0 <= persisted["best_val_metrics"]["balanced_acc"] <= 1.0
