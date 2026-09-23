"""Curvature from mesh: generate when ccbbm exists, else 2ch fallback."""

from pathlib import Path

from shapeqc.curvature import ensure_curvature


def test_mesh_without_ccbbm_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("CCBBM", raising=False)
    subj = tmp_path / "subjA"
    subj.mkdir()
    (subj / "thick_17.raw").write_bytes(b"\x00" * 16)
    (subj / "LogJacs_17.raw").write_bytes(b"\x00" * 16)
    (subj / "resliced_mesh_17.m").write_text("mesh")
    info = ensure_curvature(tmp_path, [17], write_curvature=True, ccbbm="")
    assert info["skipped_no_ccbbm"]
    assert not (subj / "curvature_17.raw").is_file()


def test_existing_curvature_not_rewritten(tmp_path):
    subj = tmp_path / "subjA"
    subj.mkdir()
    (subj / "thick_17.raw").write_bytes(b"\x00" * 16)
    (subj / "curvature_17.raw").write_bytes(b"\x01" * 16)
    info = ensure_curvature(tmp_path, [17], write_curvature=True)
    assert info["exists"]
    assert (subj / "curvature_17.raw").read_bytes() == b"\x01" * 16
