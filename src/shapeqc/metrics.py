"""Paper operating-point metrics. Fail class is y = 1."""

from __future__ import annotations

from typing import Any

import numpy as np


def f_score_mod(f_recall: float, f_share: float) -> float:
    """2 * recall * (1 - flagged) / (recall + 1 - flagged)."""
    a = float(f_recall)
    b = 1.0 - float(f_share)
    if a + b <= 0:
        return float("nan")
    return 2.0 * a * b / (a + b)


def paper_counts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray(y_true).ravel().astype(np.int64)
    s = np.asarray(y_score).ravel()
    pred = (s >= float(threshold)).astype(np.int64)
    tf = int(np.sum((y == 1) & (pred == 1)))
    ff = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    n = tf + ff + tn + fn
    f_recall = tf / (tf + fn) if (tf + fn) else float("nan")
    f_share = (tf + ff) / n if n else float("nan")
    precision = tf / (tf + ff) if (tf + ff) else float("nan")
    return {
        "TF": tf,
        "FF": ff,
        "TN": tn,
        "FN": fn,
        "n": n,
        "F_recall": f_recall,
        "F_share": f_share,
        "precision": precision,
        "F_score_mod": f_score_mod(f_recall, f_share),
        "threshold": float(threshold),
    }


def recall_floor_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target: float = 0.8,
) -> float:
    """Smallest score threshold on this set that keeps fail recall >= target."""
    y = np.asarray(y_true).ravel().astype(np.int64)
    s = np.asarray(y_score).ravel()
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return 0.5
    order = np.argsort(s)
    best = float(s.min()) if s.size else 0.5
    for tau in np.unique(s[order]):
        rec = float(np.sum((y == 1) & (s >= tau)) / n_pos)
        if rec >= float(target):
            best = float(tau)
        else:
            break
    return best
