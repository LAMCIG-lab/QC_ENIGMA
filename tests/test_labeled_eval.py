"""F-share / F-recall on labeled test preds CSVs."""

from pathlib import Path

import pandas as pd

from shapeqc.labeled_eval import fail_metrics, find_test_preds, labeled_test_table


def test_fail_metrics():
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.2, 0.8, 0.1, 0.05]
    rec = fail_metrics(y, p, tau=0.5)
    assert rec["n"] == 5
    assert rec["n_fail"] == 2
    assert rec["TP"] == 1
    assert rec["FP"] == 1
    assert rec["FN"] == 1
    assert rec["F_share"] == 0.4
    assert rec["F_recall"] == 0.5
    assert rec["precision"] == 0.5
    assert abs(rec["F1"] - 0.5) < 1e-12


def test_labeled_test_table_roi17(tmp_path):
    csv = tmp_path / (
        "preds_logistic_per_roi_thick-logjac-curv_dummy_roi17_seed42_kpca_logistic.csv"
    )
    pd.DataFrame(
        {
            "y_true": [1, 1, 0, 0, 0, 0],
            "p_fail": [0.9, 0.05, 0.8, 0.2, 0.1, 0.01],
            "y_pred": [1, 0, 1, 0, 0, 0],
        }
    ).to_csv(csv, index=False)
    assert find_test_preds(tmp_path, 17).name == csv.name
    table = labeled_test_table(tmp_path, rois=[17], seed=42, tau_override=0.5)
    row = table.iloc[0]
    assert int(row["roi"]) == 17
    assert int(row["n"]) == 6
    assert int(row["n_fail"]) == 2
    assert abs(float(row["F_share"]) - (2 / 6)) < 1e-12
    assert abs(float(row["F_recall"]) - 0.5) < 1e-12
