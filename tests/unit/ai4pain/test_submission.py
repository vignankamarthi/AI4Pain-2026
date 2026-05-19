"""Tests for ai4pain.submission (HIP-G test-set submission runner).

`run_submission` trains a spec on the 41 train subjects, early-stops on the
12-subject validation split, and predicts the BLINDED 12-subject test split,
writing test_predictions.csv. Submission budget is hard-capped at 5.
"""
import csv
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ai4pain import submission


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = (DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()
             and (DATA_ROOT / "test" / "Bvp").is_dir())


def test_module_imports():
    assert callable(submission.run_submission)
    assert callable(submission.run_from_dir)


def test_zscore_fit_shapes():
    rng = np.random.default_rng(0)
    stack = rng.standard_normal((10, 4, 33, 20)).astype(np.float32)
    mu, sig = submission._zscore_fit(stack)
    assert mu.shape == (1, 4, 1, 1)
    assert sig.shape == (1, 4, 1, 1)
    assert np.all(sig > 0)  # zero std floored to 1.0


def test_align_time_pads_to_common_max():
    a = np.zeros((3, 4, 33, 10), dtype=np.float32)
    b = np.ones((2, 4, 33, 15), dtype=np.float32)
    c = np.ones((5, 4, 33, 12), dtype=np.float32)
    aa, bb, cc = submission._align_time(a, b, c)
    assert aa.shape[-1] == bb.shape[-1] == cc.shape[-1] == 15
    # original content preserved, pad is zeros
    assert np.allclose(aa[..., :10], a)
    assert np.allclose(aa[..., 10:], 0.0)


def test_run_submission_rejects_unsupported_family(tmp_path):
    spec = {"name": "x", "model": {"family": "multi_stream_bigru"},
            "training": {}, "feature_extraction": {}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(NotImplementedError):
        submission.run_submission(tmp_path, DATA_ROOT)


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain train+test data not present")
def test_smoke_run_submission_writes_predictions(tmp_path):
    spec = {
        "name": "smoke_submission",
        "preprocessing": {"normalize": "per_channel_zscore"},
        "feature_extraction": {"family": "spectrogram", "fs": 100,
                                "nperseg": 64, "noverlap": 32,
                                "log_scale": True, "transform": "stft"},
        "model": {"family": "spectrogram_cnn2d", "base_channels": 8,
                   "depth": 1, "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0,
                     "loss": "ce_class_balanced", "optimizer": "adam"},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
        "decode": {"strategy": "argmax"},
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    result = submission.run_submission(tmp_path, DATA_ROOT)
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "result.json").exists()
    assert result["submission"] is True
    rows = list(csv.DictReader(open(tmp_path / "test_predictions.csv")))
    assert len(rows) == result["test_n_trials"]
    for r in rows:
        assert int(r["pred_label"]) in (0, 1, 2)
