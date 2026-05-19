"""Tests for ai4pain.portal_format -- AI4Pain 2026 portal CSV formatter.

The portal expects an emailed CSV `TeamName_Version.csv` with columns
Subject_ID, Sample, Predicted_label and lowercase string labels
(no_pain / arm_pain / hand_pain). Our test_predictions.csv uses a global
trial_index and NP/AP/HP names, so this module re-indexes Sample per
subject and maps the label strings.
"""
import csv
from pathlib import Path

import pytest

from ai4pain import portal_format as pf


def _write_pred_csv(path: Path, rows: list[tuple]):
    """rows: (subject, trial_index, pred_name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_index", "pred_label", "pred_name",
                    "p_NP", "p_AP", "p_HP"])
        names = ["NP", "AP", "HP"]
        for subj, ti, name in rows:
            lab = names.index(name)
            probs = [0.0, 0.0, 0.0]
            probs[lab] = 1.0
            w.writerow([subj, ti, lab, name, *probs])


def test_module_imports():
    assert callable(pf.to_portal_rows)
    assert callable(pf.write_portal_csv)
    assert pf.LABEL_TO_PORTAL == {
        "NP": "no_pain", "AP": "arm_pain", "HP": "hand_pain"}


def test_to_portal_rows_reindexes_sample_per_subject(tmp_path):
    """Global trial_index 0..N -> per-subject Sample resetting at 0."""
    src = tmp_path / "test_predictions.csv"
    # subject 3: global trials 0,1,2 ; subject 31: global trials 3,4
    _write_pred_csv(src, [(3, 0, "HP"), (3, 1, "AP"), (3, 2, "NP"),
                          (31, 3, "AP"), (31, 4, "NP")])
    rows = pf.to_portal_rows(src)
    assert [r["Subject_ID"] for r in rows] == [3, 3, 3, 31, 31]
    assert [r["Sample"] for r in rows] == [0, 1, 2, 0, 1]


def test_to_portal_rows_maps_label_strings(tmp_path):
    src = tmp_path / "test_predictions.csv"
    _write_pred_csv(src, [(3, 0, "HP"), (3, 1, "AP"), (3, 2, "NP")])
    rows = pf.to_portal_rows(src)
    assert [r["Predicted_label"] for r in rows] == [
        "hand_pain", "arm_pain", "no_pain"]


def test_to_portal_rows_sorts_by_subject_then_trial(tmp_path):
    """Out-of-order input rows are sorted subject-ascending, trial-ascending."""
    src = tmp_path / "test_predictions.csv"
    _write_pred_csv(src, [(31, 4, "NP"), (3, 2, "NP"), (31, 3, "AP"),
                          (3, 0, "HP"), (3, 1, "AP")])
    rows = pf.to_portal_rows(src)
    assert [(r["Subject_ID"], r["Sample"]) for r in rows] == [
        (3, 0), (3, 1), (3, 2), (31, 0), (31, 1)]


def test_write_portal_csv_header_and_rows(tmp_path):
    src = tmp_path / "test_predictions.csv"
    _write_pred_csv(src, [(3, 0, "HP"), (3, 1, "NP")])
    out = tmp_path / "EVOLVE_v1.csv"
    pf.write_portal_csv(src, out)
    assert out.exists()
    with open(out) as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header == ["Subject_ID", "Sample", "Predicted_label"]
    assert body == [["3", "0", "hand_pain"], ["3", "1", "no_pain"]]


def test_write_portal_csv_returns_row_count(tmp_path):
    src = tmp_path / "test_predictions.csv"
    _write_pred_csv(src, [(3, 0, "HP"), (3, 1, "NP"), (3, 2, "AP")])
    n = pf.write_portal_csv(src, tmp_path / "out.csv")
    assert n == 3


def test_to_portal_rows_rejects_unknown_label(tmp_path):
    src = tmp_path / "bad.csv"
    src.write_text("subject,trial_index,pred_label,pred_name,p_NP,p_AP,p_HP\n"
                   "3,0,0,XX,1,0,0\n")
    with pytest.raises(ValueError):
        pf.to_portal_rows(src)
