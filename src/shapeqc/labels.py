"""Load human QA labels (tidy or wide ENIGMA QA_Status CSV).

Wide example: SubjID, ROI10, ROI11, ... with codes 1=fail, 2=warn, 3=pass.
Binary fail used in this package: y_fail = 1 iff code == 1.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROI_RE = re.compile(r"^(?:ROI|R)_?(\d+)$", re.IGNORECASE)
SUBJ_CANDS = ("subject_id", "SubjID", "SUBJID", "SubjectID", "ID")


def load_labels(path: str | Path, *, fail_code: int = 1) -> pd.DataFrame:
    """Return columns subject_id, roi, y_fail (and label_code if from a wide QA file)."""
    df = pd.read_csv(path)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    if {"subject_id", "roi", "y_fail"}.issubset(set(cols)):
        out = df[["subject_id", "roi", "y_fail"]].copy()
        out["roi"] = out["roi"].astype(int)
        out["y_fail"] = out["y_fail"].astype(int)
        return out
    roi_map = {}
    for column in cols:
        match = ROI_RE.match(column)
        if match:
            roi_map[column] = int(match.group(1))
    if not roi_map:
        raise SystemExit(
            "labels file needs either subject_id,roi,y_fail or wide ROI10,ROI11,... columns: {0}".format(
                path
            )
        )
    low = {c.lower(): c for c in cols}
    subj_col = None
    for cand in SUBJ_CANDS:
        hit = low.get(cand.lower())
        if hit is not None:
            subj_col = hit
            break
    if subj_col is None:
        for column in cols:
            if "subj" in column.lower():
                subj_col = column
                break
    if subj_col is None:
        raise SystemExit("wide QA CSV has ROI columns but no subject id column: {0}".format(path))
    rows = []
    for _, rec in df.iterrows():
        sid = str(rec[subj_col]).strip()
        if not sid or sid.lower() == "nan":
            continue
        for column, roi in roi_map.items():
            raw = rec[column]
            try:
                code = int(float(raw))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "subject_id": sid,
                    "roi": roi,
                    "label_code": code,
                    "y_fail": int(code == int(fail_code)),
                }
            )
    if not rows:
        raise SystemExit("no labeled ROI rows in {0}".format(path))
    return pd.DataFrame(rows)
