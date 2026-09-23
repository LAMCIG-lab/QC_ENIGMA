"""Locate and load per-vertex ENIGMA-Shape channels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_ROIS: tuple[int, ...] = (10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)

FEATURES_3CH = ["thick", "logjac", "curv"]
FEATURES_2CH = ["thick", "logjac"]

ROI_NAMES = {
    10: "L-Thalamus",
    11: "L-Caudate",
    12: "L-Putamen",
    13: "L-Pallidum",
    17: "L-Hippocampus",
    18: "L-Amygdala",
    26: "L-Accumbens",
    49: "R-Thalamus",
    50: "R-Caudate",
    51: "R-Putamen",
    52: "R-Pallidum",
    53: "R-Hippocampus",
    54: "R-Amygdala",
    58: "R-Accumbens",
}

CHANNEL_FILES = {
    "thick": "thick_{roi}.raw",
    "logjac": "LogJacs_{roi}.raw",
    "curv": "curvature_{roi}.raw",
}
MESH_FILE = "resliced_mesh_{roi}.m"


def load_raw(path: str | Path) -> np.ndarray:
    """Load one float32 `.raw` vector (one value per atlas vertex)."""
    return np.fromfile(path, dtype=np.float32)


def subject_feature_paths(subject_dir: str | Path, roi: int) -> dict[str, Path]:
    root = Path(subject_dir)
    return {
        "thick": root / CHANNEL_FILES["thick"].format(roi=int(roi)),
        "logjac": root / CHANNEL_FILES["logjac"].format(roi=int(roi)),
        "curv": root / CHANNEL_FILES["curv"].format(roi=int(roi)),
        "mesh": root / MESH_FILE.format(roi=int(roi)),
    }


def is_subject_dir(path: str | Path) -> bool:
    """True if this folder contains at least one thick_*.raw."""
    p = Path(path)
    return p.is_dir() and any(p.glob("thick_*.raw"))


def _child_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def load_channels(
    subject_dir: str | Path,
    roi: int,
    features: Sequence[str] = ("thick", "logjac", "curv"),
    *,
    require: Sequence[str] | None = None,
) -> np.ndarray:
    """Concatenate selected channels in vertex order. Missing required files raise."""
    paths = subject_feature_paths(subject_dir, roi)
    need = list(require) if require is not None else list(features)
    arrays: list[np.ndarray] = []
    n = None
    for name in features:
        if name not in CHANNEL_FILES:
            raise ValueError("Unknown feature {0!r}".format(name))
        path = paths[name]
        if not path.is_file():
            if name in need:
                raise FileNotFoundError(path)
            if n is None:
                raise FileNotFoundError(
                    "Cannot zero-fill {0}: no other channel to copy length from".format(name)
                )
            arrays.append(np.zeros(n, dtype=np.float32))
            continue
        arr = load_raw(path)
        if n is None:
            n = int(arr.size)
        elif int(arr.size) != n:
            raise ValueError("Length mismatch in {0}: {1} vs {2}".format(path, arr.size, n))
        arrays.append(arr)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def iter_subjects(shape_root: str | Path, *, max_subjects: int | None = None) -> Iterable[Path]:
    """Yield every subject folder under ``shape_root`` that has at least one thick_*.raw.

    Walks nested ENIGMA site layouts (not only immediate children). Duplicate
    subject names at two depths keep the shortest path. ``max_subjects`` caps
    how many folders are yielded; None/0 means all of them.
    """
    from shapeqc.discover import iter_discovered

    yield from iter_discovered(shape_root, max_subjects=max_subjects)


def inventory_row(subject_dir: Path, roi: int) -> dict:
    paths = subject_feature_paths(subject_dir, roi)
    sid = subject_dir.name

    def _rel(path: Path) -> str:
        if not path.is_file():
            return ""
        return "{0}/{1}".format(sid, path.name)

    return {
        "subject_id": sid,
        "roi": int(roi),
        "thick": paths["thick"].is_file(),
        "logjac": paths["logjac"].is_file(),
        "curv": paths["curv"].is_file(),
        "mesh": paths["mesh"].is_file(),
        "subject_dir": str(subject_dir.resolve()),
        "thick_file": _rel(paths["thick"]),
        "logjac_file": _rel(paths["logjac"]),
        "curv_file": _rel(paths["curv"]),
        "mesh_file": _rel(paths["mesh"]),
    }
