"""Generate curvature_<roi>.raw from resliced_mesh_<roi>.m via CCBBM.

Not part of stock ENIGMA-Shape. Requires the `ccbbm` binary from
ENIGMA-Shape MedialDemonsShared/bin. Set CCBBM to an absolute path.
Missing curvature with a mesh present is generated next to the mesh.
If ccbbm is unavailable the row falls back to the 2-channel model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from shapeqc.io import DEFAULT_ROIS, iter_subjects, subject_feature_paths

SMOOTH_STEPS = 20


def find_ccbbm(explicit: str | Path | None = None) -> Path | None:
    """Return the ccbbm binary, or None if it is not configured."""
    raw = str(explicit or "").strip() or os.environ.get("CCBBM", "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        sys.stderr.write("shapeqc: CCBBM must be an absolute path, got {0!r}\n".format(raw))
        return None
    if p.is_file():
        return p
    sys.stderr.write("shapeqc: CCBBM is set but not a file: {0}\n".format(p))
    return None


def generate_one(mesh: Path, out_curv: Path, *, ccbbm: Path | None = None, steps: int = SMOOTH_STEPS) -> Path:
    """Write smoothed mean curvature next to the mesh. Does not zero-fill."""
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    bin_path = ccbbm or find_ccbbm()
    if bin_path is None:
        raise FileNotFoundError(
            "ccbbm not found. Set CCBBM to an absolute path to "
            "MedialDemonsShared/bin/ccbbm (do not rely on PATH)."
        )
    out_curv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shapeqc_curv_") as tmp:
        tmp_dir = Path(tmp)
        mean_raw = tmp_dir / "mean.raw"
        gauss_raw = tmp_dir / "gauss.raw"
        subprocess.run(
            [str(bin_path), "-store_att_curv", str(mesh), str(mean_raw), str(gauss_raw)],
            check=True,
        )
        subprocess.run(
            [
                str(bin_path),
                "-laplacian_smooth_attribute",
                str(mesh),
                str(mean_raw),
                str(out_curv),
                str(int(steps)),
            ],
            check=True,
        )
    if not out_curv.is_file():
        raise RuntimeError("ccbbm did not write {0}".format(out_curv))
    return out_curv


def ensure_curvature(
    shape_root: str | Path,
    rois: list[int],
    *,
    max_subjects: int | None = None,
    subjects: Iterable[Path] | None = None,
    write_curvature: bool = True,
    ccbbm: str | Path | None = None,
) -> dict[str, list[str]]:
    """Fill missing curvature_*.raw from resliced_mesh_*.m when possible.

    Writes next to the mesh. Missing ccbbm or missing mesh leaves the row for
    2-channel fallback; the site is not aborted.

    Returns buckets: skipped_no_mesh, skipped_no_ccbbm, wrote, exists.
    """
    buckets = {
        "skipped_no_mesh": [],
        "skipped_no_ccbbm": [],
        "wrote": [],
        "exists": [],
        "failed": [],
    }
    subj_iter = list(subjects) if subjects is not None else list(
        iter_subjects(shape_root, max_subjects=max_subjects)
    )
    bin_path = find_ccbbm(ccbbm) if write_curvature else None
    warned_ccbbm = False
    for subj in subj_iter:
        for roi in rois:
            paths = subject_feature_paths(subj, roi)
            token = "{0} roi={1}".format(subj.name, roi)
            if paths["curv"].is_file():
                buckets["exists"].append(token)
                continue
            if not paths["mesh"].is_file():
                buckets["skipped_no_mesh"].append(token)
                continue
            if not write_curvature:
                buckets["skipped_no_ccbbm"].append(token)
                continue
            if bin_path is None:
                buckets["skipped_no_ccbbm"].append(token)
                if not warned_ccbbm:
                    sys.stderr.write(
                        "shapeqc: mesh present but CCBBM unset; "
                        "those ROIs use the 2-channel model. "
                        "Set CCBBM to write curvature_<roi>.raw next to the mesh.\n"
                    )
                    warned_ccbbm = True
                continue
            try:
                generate_one(paths["mesh"], paths["curv"], ccbbm=bin_path)
                buckets["wrote"].append(token)
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                buckets["failed"].append("{0} ({1})".format(token, exc))
                sys.stderr.write("shapeqc: curvature failed for {0}: {1}\n".format(token, exc))
    return buckets


def generate_tree(
    shape_root: str | Path,
    *,
    rois: Sequence[int] | None = None,
    overwrite: bool = False,
    ccbbm: str | Path | None = None,
) -> list[dict]:
    roi_list = list(rois) if rois is not None else list(DEFAULT_ROIS)
    bin_path = find_ccbbm(ccbbm)
    if bin_path is None:
        raise FileNotFoundError(
            "ccbbm not found. Set CCBBM to an absolute path to "
            "MedialDemonsShared/bin/ccbbm."
        )
    rows = []
    for subj in iter_subjects(shape_root):
        for roi in roi_list:
            paths = subject_feature_paths(subj, roi)
            rec = {
                "subject_id": subj.name,
                "roi": int(roi),
                "status": "skip",
                "path": str(paths["curv"]),
            }
            if paths["curv"].is_file() and not overwrite:
                rec["status"] = "exists"
                rows.append(rec)
                continue
            if not paths["mesh"].is_file():
                rec["status"] = "no_mesh"
                rows.append(rec)
                continue
            generate_one(paths["mesh"], paths["curv"], ccbbm=bin_path)
            rec["status"] = "wrote"
            rows.append(rec)
    return rows
