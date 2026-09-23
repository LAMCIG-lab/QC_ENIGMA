"""Inventory a Shape tree without scoring (qc-report --doctor)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from shapeqc.curvature import find_ccbbm
from shapeqc.discover import discover_subjects
from shapeqc.io import DEFAULT_ROIS, inventory_row, load_raw, subject_feature_paths
from shapeqc.models_frozen import find_joblib, load_joblib_checksums, verify_joblib

_VERT_CSV = Path(__file__).resolve().parents[2] / "configs" / "vertices.csv"
EXPECTED_SKLEARN = "1.3.2"


def _sklearn_version() -> str:
    try:
        import sklearn

        return str(sklearn.__version__)
    except Exception as exc:
        return "missing ({0})".format(exc)


def _load_vertices() -> dict[int, int]:
    out: dict[int, int] = {}
    if not _VERT_CSV.is_file():
        return out
    import pandas as pd

    df = pd.read_csv(_VERT_CSV)
    for _, row in df.iterrows():
        out[int(row["roi"])] = int(row["n_vertices"])
    return out


def run_doctor(
    shape_root: str | Path,
    *,
    rois: list[int] | None = None,
    max_subjects: int | None = None,
    models_3ch: str | Path | None = None,
    models_2ch: str | Path | None = None,
    ccbbm: str | Path | None = None,
    verify_checksums: bool = True,
    dummy: bool = False,
) -> dict:
    roi_list = list(rois) if rois else list(DEFAULT_ROIS)
    discovered = discover_subjects(shape_root, max_subjects=max_subjects)
    vertices = _load_vertices()
    sk = _sklearn_version()
    bin_path = find_ccbbm(ccbbm)

    n_thick_logjac = 0
    n_3ch = 0
    n_mesh_only = 0
    n_incomplete = 0
    vertex_mismatch = []
    for subj in discovered.subjects:
        for roi in roi_list:
            rec = inventory_row(subj, roi)
            if rec["thick"] and rec["logjac"]:
                n_thick_logjac += 1
                if rec["curv"]:
                    n_3ch += 1
                elif rec["mesh"]:
                    n_mesh_only += 1
                if rec["thick"]:
                    n_vert = int(load_raw(subject_feature_paths(subj, roi)["thick"]).size)
                    expected = vertices.get(int(roi))
                    if expected and n_vert != expected and not dummy:
                        vertex_mismatch.append(
                            "{0} roi={1} got {2} want {3}".format(
                                rec["subject_id"], roi, n_vert, expected
                            )
                        )
            elif rec["thick"] or rec["logjac"]:
                n_incomplete += 1

    joblib_ok = []
    joblib_fail = []
    if not dummy and verify_checksums:
        checksums = load_joblib_checksums()
        for channel_set, model_dir in (
            ("thick+logjac+curv", Path(models_3ch) if models_3ch else None),
            ("thick+logjac", Path(models_2ch) if models_2ch else None),
        ):
            if model_dir is None or not model_dir.is_dir():
                continue
            for roi in roi_list:
                try:
                    path = find_joblib(roi, 42, model_dir)
                    verify_joblib(
                        path, channel_set=channel_set, roi=roi, seed=42, checksums=checksums
                    )
                    joblib_ok.append("{0} roi={1} {2}".format(channel_set, roi, path.name))
                except SystemExit as exc:
                    joblib_fail.append(str(exc))
                except Exception as exc:
                    joblib_fail.append("{0} roi={1}: {2}".format(channel_set, roi, exc))

    payload = {
        "mode": "doctor",
        "shape_root": str(Path(shape_root).resolve()),
        "discover_mode": discovered.mode,
        "search_roots": [str(p) for p in discovered.search_roots],
        "n_subjects": len(discovered.subjects),
        "n_duplicates_dropped": len(discovered.duplicates_dropped),
        "n_skipped_dirs": len(discovered.skipped_dirs),
        "n_usable_thick_logjac_rows": n_thick_logjac,
        "n_3ch_ready": n_3ch,
        "n_mesh_only_can_write_curv": n_mesh_only,
        "n_incomplete_pairs": n_incomplete,
        "n_2ch_fallback": n_thick_logjac - n_3ch,
        "sklearn": sk,
        "sklearn_ok": sk == EXPECTED_SKLEARN or dummy,
        "ccbbm": str(bin_path) if bin_path else "",
        "ccbbm_ok": bin_path is not None or n_mesh_only == 0,
        "vertex_mismatch": vertex_mismatch[:20],
        "n_vertex_mismatch": len(vertex_mismatch),
        "n_joblib_ok": len(joblib_ok),
        "n_joblib_fail": len(joblib_fail),
        "joblib_fail": joblib_fail[:8],
        "dummy_track": dummy,
        "ok": (
            len(discovered.subjects) > 0
            and n_thick_logjac > 0
            and (dummy or sk == EXPECTED_SKLEARN)
            and not joblib_fail
            and (dummy or not vertex_mismatch)
        ),
    }
    return payload


def print_doctor(payload: dict) -> None:
    print(json.dumps(payload, indent=2))
    if payload.get("ok"):
        print("doctor: ready to score")
    else:
        print("doctor: fix the issues above before scoring frozen weights", file=sys.stderr)
