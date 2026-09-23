"""Map a user-supplied fail-recall target to a predict_proba threshold.

Fit on labeled validation scores. ``tau_for(desired_recall)`` is the
recall-floor tau (largest threshold whose fail recall is still >= target).
``method="interp"`` linearly interpolates tau between stored operating points.

Keys are (channel_set, roi, seed) with roi ``17`` (single) or ``17_53`` (pair).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_RECALL_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def _recall_floor_threshold(
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


DEFAULT_PAIRS: tuple[tuple[int, int], ...] = (
    (10, 49),
    (11, 50),
    (12, 51),
    (13, 52),
    (17, 53),
    (18, 54),
    (26, 58),
)


def roi_token(roi) -> str:
    """Normalize 17, '17', '17_53', '17,53', or (17, 53) to a filename token."""
    if isinstance(roi, (tuple, list)):
        return "{0}_{1}".format(int(roi[0]), int(roi[1]))
    text = str(roi).strip().replace(",", "_").replace(" ", "")
    if not text:
        raise ValueError("empty roi")
    return text


def parse_roi_key(val):
    """int for singles, str ``13_52`` for pairs (thresholds.csv / interpolator.csv)."""
    token = roi_token(val)
    if "_" in token:
        return token
    return int(token)


class RecallThresholdInterpolator:
    """Stored (desired_recall, tau) curve for one modeling situation."""

    def __init__(self, desired_recall: Sequence[float], tau: Sequence[float]):
        rec = np.asarray(desired_recall, dtype=np.float64).ravel()
        th = np.asarray(tau, dtype=np.float64).ravel()
        if rec.size == 0 or rec.size != th.size:
            raise ValueError("desired_recall and tau must be same non-empty length")
        order = np.argsort(rec)
        self.desired_recall = rec[order]
        self.tau = th[order]

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        p_fail: np.ndarray,
        targets: Sequence[float] | None = None,
    ) -> "RecallThresholdInterpolator":
        grid = list(targets) if targets is not None else list(DEFAULT_RECALL_GRID)
        taus = [
            _recall_floor_threshold(y_true, p_fail, target=float(t)) for t in grid
        ]
        return cls(grid, taus)

    def tau_for(self, desired_recall: float, method: str = "floor") -> float:
        """Threshold for a fail-recall target.

        ``floor``: largest stored target >= desired_recall, else the max stored
        target (guarantees recall on the fit set at that grid point).
        ``interp``: ``np.interp`` on the stored curve (may undershoot recall).
        """
        r = float(desired_recall)
        if method == "interp":
            return float(np.interp(r, self.desired_recall, self.tau))
        if method != "floor":
            raise ValueError("method must be 'floor' or 'interp', got {0!r}".format(method))
        ge = np.where(self.desired_recall >= r)[0]
        if ge.size:
            return float(self.tau[ge[0]])
        return float(self.tau[-1])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"desired_recall": self.desired_recall, "tau": self.tau})


def load_interpolator_table(path: str | Path) -> dict[tuple[str, object, int], RecallThresholdInterpolator]:
    """CSV columns: channel_set, roi, seed, desired_recall, tau."""
    df = pd.read_csv(path)
    need = {"channel_set", "roi", "seed", "desired_recall", "tau"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit("interpolator CSV missing columns: {0}".format(", ".join(sorted(missing))))
    out: dict[tuple[str, object, int], RecallThresholdInterpolator] = {}
    for (ch, roi_raw, seed), g in df.groupby(["channel_set", "roi", "seed"], sort=False):
        key = (str(ch), parse_roi_key(roi_raw), int(seed))
        out[key] = RecallThresholdInterpolator(g["desired_recall"].to_numpy(), g["tau"].to_numpy())
    return out


def interpolator_rows(
    interp: RecallThresholdInterpolator,
    *,
    channel_set: str,
    roi,
    seed: int = 42,
) -> list[dict]:
    token = roi_token(roi)
    rows = []
    for rec, tau in zip(interp.desired_recall, interp.tau):
        rows.append(
            {
                "channel_set": channel_set,
                "roi": token,
                "seed": int(seed),
                "desired_recall": float(rec),
                "tau": float(tau),
            }
        )
    return rows


def tau_from_table(
    table: dict[tuple[str, object, int], RecallThresholdInterpolator],
    *,
    channel_set: str,
    roi,
    seed: int = 42,
    desired_recall: float,
    method: str = "floor",
) -> float:
    key = (str(channel_set), parse_roi_key(roi), int(seed))
    interp = table.get(key)
    if interp is None:
        raise SystemExit(
            "No interpolator for {0} roi={1} seed={2}".format(channel_set, roi_token(roi), seed)
        )
    return interp.tau_for(desired_recall, method=method)


def write_interpolator_csv(rows: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path
