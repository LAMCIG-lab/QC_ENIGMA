"""Generate curvature_<roi>.raw from resliced_mesh_<roi>.m via CCBBM.

Not part of stock ENIGMA-Shape. Requires the `ccbbm` binary from
ENIGMA-Shape MedialDemonsShared/bin. Set CCBBM=/path/to/ccbbm if it is not
on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from shapeqc.io import DEFAULT_ROIS, iter_subjects, subject_feature_paths

SMOOTH_STEPS = 20


def find_ccbbm() -> Path:
    env = os.environ.get("CCBBM", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError("CCBBM is set but not a file: {0}".format(env))
    which = shutil.which("ccbbm")
    if which:
        return Path(which)
    raise FileNotFoundError(
        "ccbbm not found. Install ENIGMA-Shape and set CCBBM to "
        "MedialDemonsShared/bin/ccbbm"
    )


def generate_one(mesh: Path, out_curv: Path, *, ccbbm: Path | None = None, steps: int = SMOOTH_STEPS) -> Path:
    """Write smoothed mean curvature next to the mesh. Does not zero-fill."""
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    bin_path = ccbbm or find_ccbbm()
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


def generate_tree(
    shape_root: str | Path,
    *,
    rois: Sequence[int] | None = None,
    overwrite: bool = False,
) -> list[dict]:
    roi_list = list(rois) if rois is not None else list(DEFAULT_ROIS)
    ccbbm = find_ccbbm()
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
            generate_one(paths["mesh"], paths["curv"], ccbbm=ccbbm)
            rec["status"] = "wrote"
            rows.append(rec)
    return rows
