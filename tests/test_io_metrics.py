"""Tests for shapeqc I/O and paper metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shapeqc.io import inventory_row, iter_subjects, load_channels, load_raw
from shapeqc.metrics import f_score_mod, paper_counts, recall_floor_threshold
from shapeqc.models_frozen import apply_tau_override

TINY = Path(__file__).resolve().parents[1] / "examples" / "tiny"


def test_tiny_raw_length():
    a = load_raw(TINY / "subjA" / "thick_17.raw")
    assert a.dtype == np.float32
    assert a.size == 32


def test_load_three_channels():
    x = load_channels(TINY / "subjA", 17, features=["thick", "logjac", "curv"], require=["thick", "logjac", "curv"])
    assert x.size == 96


def test_missing_curv_raises():
    with pytest.raises(FileNotFoundError):
        load_channels(TINY / "subjA", 10, features=["thick", "logjac", "curv"], require=["thick", "logjac", "curv"])


def test_paper_counts():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    m = paper_counts(y, s, threshold=0.5)
    assert m["TF"] == 1
    assert m["FN"] == 1
    assert m["FF"] == 1
    assert abs(m["F_recall"] - 0.5) < 1e-9
    assert abs(f_score_mod(1.0, 0.0) - 1.0) < 1e-9
    assert f_score_mod(0.8, 0.3) > 0.7


def test_recall_floor():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.6, 0.4, 0.1])
    tau = recall_floor_threshold(y, s, target=0.8)
    rec = float(np.mean((s[y == 1] >= tau)))
    assert rec >= 0.8


def test_annotate_errors():
    from shapeqc.report import annotate_errors

    df = pd.DataFrame({"y_fail": [1, 1, 0, 0], "P_fail": [0.9, 0.1, 0.8, 0.05]})
    out = annotate_errors(df, tau=0.5)
    assert list(out["error"]) == ["TP", "FN", "FP", "TN"]
    assert list(out["flag"]) == [1, 0, 1, 0]


def test_no_mesh_no_curv_stays_2channel(tmp_path):
    from shapeqc.report import ensure_curvature

    subj = tmp_path / "s1"
    subj.mkdir()
    np.zeros(8, dtype=np.float32).tofile(subj / "thick_17.raw")
    np.zeros(8, dtype=np.float32).tofile(subj / "LogJacs_17.raw")
    skipped = ensure_curvature(tmp_path, [17])
    assert skipped == ["s1 roi=17"]
    rec = inventory_row(subj, 17)
    assert rec["thick"] and rec["logjac"]
    assert not rec["curv"]


def test_iter_subjects_rejects_unknown_dirs(tmp_path):
    (tmp_path / "stats_bilateral").mkdir()
    subj = tmp_path / "top15_0001"
    subj.mkdir()
    np.zeros(4, dtype=np.float32).tofile(subj / "thick_17.raw")
    with pytest.raises(SystemExit, match="Unknown directories"):
        list(iter_subjects(tmp_path))


def test_iter_subjects_site_and_single(tmp_path):
    subj = tmp_path / "top15_0001"
    subj.mkdir()
    np.zeros(4, dtype=np.float32).tofile(subj / "thick_17.raw")
    names = [p.name for p in iter_subjects(tmp_path)]
    assert names == ["top15_0001"]
    lone = tmp_path / "one"
    lone.mkdir()
    np.zeros(4, dtype=np.float32).tofile(lone / "thick_17.raw")
    np.zeros(4, dtype=np.float32).tofile(lone / "LogJacs_17.raw")
    got = list(iter_subjects(lone))
    assert got == [lone]


def test_site_root_csv_ignored(tmp_path):
    subj = tmp_path / "s1"
    subj.mkdir()
    np.zeros(4, dtype=np.float32).tofile(subj / "thick_17.raw")
    (tmp_path / "QA_status_unix.csv").write_text("SubjID,ROI17\ns1,1\n")
    names = [p.name for p in iter_subjects(tmp_path)]
    assert names == ["s1"]


def test_apply_tau_override():
    taus = {
        ("thick+logjac+curv", 11, 42): 0.3,
        ("thick+logjac+curv", 17, 42): 0.5,
        ("thick+logjac", 11, 42): 0.4,
    }
    all_one = apply_tau_override(taus, 0.9)
    assert all(v == 0.9 for v in all_one.values())
    per = apply_tau_override(taus, {11: 0.7})
    assert per[("thick+logjac+curv", 11, 42)] == 0.7
    assert per[("thick+logjac", 11, 42)] == 0.7
    assert per[("thick+logjac+curv", 17, 42)] == 0.5


def test_paper_meta_scarce_rois():
    from shapeqc.paper_meta import load_roi_paper_meta

    meta = load_roi_paper_meta()
    assert meta[17]["paper_deployable"]
    assert meta[13]["paper_deployable"]
    assert meta[53]["paper_deployable"]
    assert not meta[11]["paper_deployable"]
    assert meta[11]["n_fail_paper"] == 9


def test_find_joblib_requires_unique(tmp_path):
    from shapeqc.models_frozen import find_joblib

    (tmp_path / "a_roi17_seed42_kpca_logistic.joblib").write_bytes(b"a")
    (tmp_path / "b_roi17_seed42_kpca_logistic.joblib").write_bytes(b"b")
    with pytest.raises(SystemExit, match="Multiple"):
        find_joblib(17, 42, tmp_path)
