"""Minimal train path for examples and CI.

The paper model is KernelPCA + logistic (see configs/paper_l3_thick_logjac_curv.yaml).
This module fits a StandardScaler + logistic on concatenated vertex channels so
the CLI and tests run without the cluster training script.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shapeqc.io import load_channels
from shapeqc.metrics import paper_counts, recall_floor_threshold


def fit_simple(
    manifest: pd.DataFrame,
    *,
    features: list[str],
    require: list[str],
    C: float = 1.0,
    seed: int = 42,
) -> tuple[Pipeline, float]:
    Xs = []
    ys = []
    for _, row in manifest.iterrows():
        x = load_channels(
            row["subject_dir"],
            int(row["roi"]),
            features=features,
            require=require,
        )
        Xs.append(x)
        ys.append(int(row["y_fail"]))
    X = np.stack(Xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    scores = pipe.predict_proba(X)[:, 1]
    tau = recall_floor_threshold(y, scores, target=0.8)
    return pipe, tau


def score_rows(pipe: Pipeline, manifest: pd.DataFrame, features: list[str], require: list[str], tau: float) -> pd.DataFrame:
    rows = []
    y_true = []
    y_score = []
    for _, row in manifest.iterrows():
        x = load_channels(row["subject_dir"], int(row["roi"]), features=features, require=require)
        p = float(pipe.predict_proba(x.reshape(1, -1))[0, 1])
        y_true.append(int(row["y_fail"]))
        y_score.append(p)
        rows.append(
            {
                "subject_id": row["subject_id"],
                "roi": int(row["roi"]),
                "y_fail": int(row["y_fail"]),
                "P_fail": p,
                "tau": float(tau),
                "flag": int(p >= tau),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["paper"] = paper_counts(np.array(y_true), np.array(y_score), threshold=tau)
    return out


def save_model(pipe: Pipeline, tau: float, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "tau": float(tau)}, path)
