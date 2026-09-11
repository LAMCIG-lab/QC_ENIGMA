"""CLI: qc-report only (unlabeled .raw drop)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shapeqc.cli import report

TINY = Path(__file__).resolve().parents[1] / "examples" / "tiny"


def _dummy_3ch(model_dir: Path, rois, n_vert=32, seed=42, tau=0.5):
    n_feat = 3 * n_vert
    rng = np.random.RandomState(0)
    X = rng.randn(24, n_feat)
    y = np.array([0] * 18 + [1] * 6)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=0)),
        ]
    )
    pipe.fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    for roi in rois:
        joblib.dump(pipe, model_dir / "dummy_roi{0}_seed{1}_kpca_logistic.joblib".format(roi, seed))
    thr = pd.DataFrame(
        [{"channel_set": "thick+logjac+curv", "roi": r, "seed": seed, "tau": tau} for r in rois]
    )
    path = model_dir / "thresholds.csv"
    thr.to_csv(path, index=False)
    return path


def test_qc_report_unlabeled_tiny(tmp_path):
    models = tmp_path / "models3"
    thr = _dummy_3ch(models, [13, 17])
    out = tmp_path / "qc"
    report(
        [
            "--shape-root",
            str(TINY),
            "--rois",
            "13,17",
            "--out-dir",
            str(out),
            "--models-3ch",
            str(models),
            "--thresholds",
            str(thr),
            "--skip-joblib-verify",
        ]
    )
    preds = pd.read_csv(out / "predictions.csv")
    assert len(preds) == 4
    assert set(preds["roi"].astype(int)) == {13, 17}
    assert (preds["channel_set"] == "thick+logjac+curv").all()
    assert "y_fail" not in preds.columns
    assert "subject_dir" not in preds.columns
    assert "files" in preds.columns
    assert "paper_deployable" in preds.columns
    assert preds.loc[preds["roi"] == 17, "paper_deployable"].astype(bool).all()
    assert preds["thick_file"].str.contains("thick_").all()
    md = (out / "qc_report.md").read_text()
    assert "checking strategy" in md.lower()
    assert "curv_full_recall80" in md
    assert "Review list" in md or "files to check" in md
    assert "F-recall" not in md
    assert "thick_17.raw" in md or "thick_13.raw" in md or "No samples flagged" in md


def test_qc_report_requires_local_paths():
    try:
        report([])
    except SystemExit as exc:
        assert "local_paths" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
