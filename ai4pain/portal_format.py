"""AI4Pain 2026 portal CSV formatter.

Test submissions are emailed to the organisers as `TeamName_Version.csv`
with columns `Subject_ID, Sample, Predicted_label` and lowercase string
labels. Our `test_predictions.csv` uses a *global* trial_index (0..431)
and NP/AP/HP names; this module re-indexes Sample per subject and maps
the labels. Pure stdlib -- runs anywhere, no torch.

Portal label strings (from the organisers' email example):
    no_pain / arm_pain / hand_pain
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABEL_TO_PORTAL = {"NP": "no_pain", "AP": "arm_pain", "HP": "hand_pain"}


def to_portal_rows(predictions_csv: Path) -> list[dict]:
    """Read a `test_predictions.csv` -> portal rows.

    Each row: {Subject_ID: int, Sample: int, Predicted_label: str}. `Sample`
    is re-indexed per subject (0..n-1) because the portal expects a
    per-subject sample counter, not our global trial_index. Output is
    sorted subject-ascending then Sample-ascending (matches the organisers'
    example ordering).
    """
    predictions_csv = Path(predictions_csv)
    raw: list[tuple[int, int, str]] = []
    with open(predictions_csv) as f:
        for row in csv.DictReader(f):
            name = row["pred_name"]
            if name not in LABEL_TO_PORTAL:
                raise ValueError(
                    f"unknown pred_name {name!r}; expected one of "
                    f"{sorted(LABEL_TO_PORTAL)}")
            raw.append((int(row["subject"]), int(row["trial_index"]), name))

    raw.sort(key=lambda t: (t[0], t[1]))  # subject, then global trial_index
    out: list[dict] = []
    sample_by_subject: dict[int, int] = {}
    for subject, _trial, name in raw:
        sample = sample_by_subject.get(subject, 0)
        sample_by_subject[subject] = sample + 1
        out.append({
            "Subject_ID": subject,
            "Sample": sample,
            "Predicted_label": LABEL_TO_PORTAL[name],
        })
    return out


def write_portal_csv(predictions_csv: Path, out_path: Path) -> int:
    """Write the portal CSV (`Subject_ID,Sample,Predicted_label`).
    Returns the number of prediction rows written."""
    rows = to_portal_rows(predictions_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Subject_ID", "Sample", "Predicted_label"])
        for r in rows:
            w.writerow([r["Subject_ID"], r["Sample"], r["Predicted_label"]])
    print(f"[portal] {predictions_csv} -> {out_path} ({len(rows)} rows)",
          flush=True)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format a test_predictions.csv for the AI4Pain portal.")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="path to a submission's test_predictions.csv")
    parser.add_argument("--out", required=True, type=Path,
                        help="output path, e.g. TeamName_Version.csv")
    args = parser.parse_args()
    write_portal_csv(args.predictions, args.out)
