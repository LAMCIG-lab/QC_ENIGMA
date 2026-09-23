"""CLI: unlabeled .raw drop, dummy 32-vertex track, and doctor."""

from pathlib import Path

import pandas as pd

from shapeqc.cli import report
from shapeqc.discover import discover_subjects
from shapeqc.dummy import write_dummy_models

TINY = Path(__file__).resolve().parents[1] / "examples" / "tiny"
DUMMY = Path(__file__).resolve().parents[1] / "examples" / "dummy_models"


def _ensure_dummy():
    if not (DUMMY / "3ch").is_dir():
        write_dummy_models(DUMMY)


def test_qc_report_example(tmp_path):
    _ensure_dummy()
    out = tmp_path / "qc"
    report(["--example", "--out-dir", str(out)])
    preds = pd.read_csv(out / "predictions.csv")
    assert len(preds) == 4
    assert set(preds["roi"].astype(int)) == {13, 17}
    assert (preds["channel_set"] == "thick+logjac+curv").all()
    assert "y_fail" not in preds.columns
    assert "paper_deployable" in preds.columns
    assert preds.loc[preds["roi"] == 17, "paper_deployable"].astype(str).str.lower().eq("yes").all()
    md = (out / "qc_report.md").read_text()
    assert "checking strategy" in md.lower()


def test_doctor_example():
    _ensure_dummy()
    try:
        report(["--example", "--doctor"])
    except SystemExit as exc:
        assert exc.code in (0, None)


def test_discover_nested_and_dedupe(tmp_path):
    site = tmp_path / "mysite"
    keep = site / "subjA"
    dup = site / "bilat_shape" / "subjA"
    stats = site / "stats_bilateral"
    keep.mkdir(parents=True)
    dup.mkdir(parents=True)
    stats.mkdir()
    (keep / "thick_17.raw").write_bytes(b"\x00" * 8)
    (dup / "thick_17.raw").write_bytes(b"\x00" * 8)
    (stats / "notes.txt").write_text("no")
    found = discover_subjects(site)
    names = [p.name for p in found.subjects]
    assert names == ["subjA"]
    assert found.subjects[0].resolve() == keep.resolve()
    assert found.duplicates_dropped or any("bilat_shape" in p for p in found.skipped_dirs)


def test_discover_enigma_sz_allowlist(tmp_path):
    root = tmp_path / "ENIGMA_SZ"
    narsad = root / "Brazil" / "NARSAD" / "sN"
    wellcome = root / "Brazil" / "Wellcome" / "sW"
    top = root / "TOP" / "TOP_SZ" / "sT"
    for folder in (narsad, wellcome, top):
        folder.mkdir(parents=True)
        (folder / "thick_17.raw").write_bytes(b"\x00" * 8)
    # enough labeled children to trigger enigma_sz mode
    for extra in ("FBIRN", "UCIS", "Oxford"):
        (root / extra).mkdir()
    found = discover_subjects(root)
    names = sorted(p.name for p in found.subjects)
    assert "sN" in names
    assert "sT" in names
    assert "sW" not in names
    assert found.mode == "enigma_sz"
