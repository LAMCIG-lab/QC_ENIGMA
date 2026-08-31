"""CLI smoke on examples/tiny."""

import json
from pathlib import Path

from shapeqc.cli import prep, score, train

TINY = Path(__file__).resolve().parents[1] / "examples" / "tiny"


def test_prep_train_score(tmp_path):
    man = tmp_path / "manifest.csv"
    prep(
        [
            "--shape-root",
            str(TINY),
            "--rois",
            "17",
            "--labels",
            str(TINY / "labels.csv"),
            "--out",
            str(man),
        ]
    )
    df_path = man
    assert df_path.is_file()
    run = tmp_path / "run"
    train(["--manifest", str(man), "--out-dir", str(run), "--features", "thick,logjac,curv"])
    assert (run / "model_simple.joblib").is_file()
    metrics = json.loads((run / "paper_metrics.json").read_text())
    assert "F_recall" in metrics
    out = tmp_path / "score"
    score(
        [
            "--shape-root",
            str(TINY),
            "--rois",
            "17",
            "--model",
            str(run / "model_simple.joblib"),
            "--out-dir",
            str(out),
        ]
    )
    assert (out / "review_list.csv").is_file()
    assert (out / "inventory.csv").is_file()
