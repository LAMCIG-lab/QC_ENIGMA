"""Tests for shapeqc I/O and paper metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shapeqc.io import load_channels, load_raw
from shapeqc.metrics import f_score_mod, paper_counts, recall_floor_threshold

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
