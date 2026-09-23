"""Labeled test metrics for frozen scores (F-share, F-recall, F1).

F-share = flagged / n  (review-queue size on the labeled test split)
F-recall = TP / n_fail  (fail recall)
F1 = harmonic mean of precision and F-recall

Uses the seed-42 ``preds_*.csv`` next to the frozen joblibs (already scored
test rows: y_true, p_fail). Does not read QA CSVs at qc-report time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shapeqc.interpolate import load_interpolator_table, roi_token, tau_from_table
from shapeqc.io import DEFAULT_ROIS, ROI_NAMES
from shapeqc.models_frozen import apply_tau_override, load_thresholds

_PKG = Path(__file__).resolve().parents[2]


def find_test_preds(model_dir: str | Path, roi, *, seed: int = 42) -> Path:
    token = roi_token(roi)
    needle = "_roi{0}_seed{1}_kpca_logistic.csv".format(token, int(seed))
    root = Path(model_dir)
    hits = sorted(
        p for p in root.glob("preds_*" + needle) if p.name.startswith("preds_")
    )
    if not hits:
        raise FileNotFoundError(
            "No test preds CSV for roi={0} seed={1} in {2}. "
            "Point MODEL_3CH at the training folder that holds preds_*.csv "
            "(same place as the joblibs).".format(token, seed, root)
        )
    if len(hits) > 1:
        raise SystemExit("Multiple test preds for roi={0}: {1}".format(token, hits))
    return hits[0]


def fail_metrics(y_true, p_fail, tau: float) -> dict:
    y = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(p_fail).ravel().astype(np.float64)
    pred = (p >= float(tau)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = int(y.size)
    n_fail = int((y == 1).sum())
    flagged = tp + fp
    f_share = flagged / n if n else 0.0
    f_recall = tp / n_fail if n_fail else float("nan")
    precision = tp / flagged if flagged else 0.0
    if n_fail and (precision + f_recall):
        f1 = 2.0 * precision * f_recall / (precision + f_recall)
    else:
        f1 = 0.0
    return {
        "n": n,
        "n_fail": n_fail,
        "tau": float(tau),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "F_share": f_share,
        "F_recall": f_recall,
        "precision": precision,
        "F1": f1,
    }


def labeled_test_table(
    model_dir: str | Path,
    *,
    rois=None,
    seed: int = 42,
    tau_override=None,
    interpolators: str | Path | None = None,
    desired_recall: float | None = None,
    channel_set: str = "thick+logjac+curv",
) -> pd.DataFrame:
    roi_list = list(rois) if rois else list(DEFAULT_ROIS)
    taus = apply_tau_override(load_thresholds(), tau_override)
    interp_table = None
    if interpolators:
        p = Path(interpolators)
        if not p.is_file():
            alt = _PKG / p
            if alt.is_file():
                p = alt
        interp_table = load_interpolator_table(p)
    elif desired_recall is not None:
        interp_table = load_interpolator_table(_PKG / "configs" / "interpolator.csv")

    rows = []
    for roi in roi_list:
        csv_path = find_test_preds(model_dir, roi, seed=seed)
        df = pd.read_csv(csv_path)
        if desired_recall is not None and interp_table is not None:
            tau = tau_from_table(
                interp_table,
                channel_set=channel_set,
                roi=roi,
                seed=seed,
                desired_recall=float(desired_recall),
            )
        else:
            key = (channel_set, int(roi), int(seed))
            if key not in taus:
                raise SystemExit("No tau for {0} roi={1}".format(channel_set, roi))
            tau = float(taus[key])
        rec = fail_metrics(df["y_true"], df["p_fail"], tau)
        rec.update(
            {
                "roi": int(roi),
                "roi_name": ROI_NAMES.get(int(roi), ""),
                "channel_set": channel_set,
                "preds_csv": csv_path.name,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)
