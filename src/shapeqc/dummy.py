"""32-vertex dummy logistics for the in-repo smoke track. Not paper KPCA weights."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PKG_ROOT / "examples" / "dummy_models"
ROIS = (13, 17)
SEED = 42
N_VERT = 32


def _pipe(n_feat: int, seed: int = 0) -> Pipeline:
    rng = np.random.RandomState(seed)
    X = rng.randn(24, n_feat).astype(np.float32)
    y = np.array([0] * 18 + [1] * 6)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=0)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def write_dummy_models(out: Path | None = None) -> Path:
    dest = Path(out) if out is not None else DEFAULT_OUT
    dir3 = dest / "3ch"
    dir2 = dest / "2ch"
    dir3.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)
    pipe3 = _pipe(3 * N_VERT)
    pipe2 = _pipe(2 * N_VERT, seed=1)
    rows = []
    for roi in ROIS:
        joblib.dump(pipe3, dir3 / "dummy_roi{0}_seed{1}_kpca_logistic.joblib".format(roi, SEED))
        joblib.dump(pipe2, dir2 / "dummy_roi{0}_seed{1}_kpca_logistic.joblib".format(roi, SEED))
        rows.append({"channel_set": "thick+logjac+curv", "roi": roi, "seed": SEED, "tau": 0.5})
        rows.append({"channel_set": "thick+logjac", "roi": roi, "seed": SEED, "tau": 0.5})
    pd.DataFrame(rows).to_csv(dest / "thresholds.csv", index=False)
    return dest
