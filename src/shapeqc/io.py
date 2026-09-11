"""Locate and load per-vertex ENIGMA-Shape channels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_ROIS: tuple[int, ...] = (10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)

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
    """Yield subject folders that contain at least one thick_*.raw.

    Accepts either a site root (subject subfolders) or a single subject folder.
    Directories without thick_*.raw are rejected, not skipped.
    Loose files at the site root (README, leftover CSVs) are ignored.
    """
    root = Path(shape_root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    if is_subject_dir(root):
        unknown = [p.name for p in _child_dirs(root)]
        if unknown:
            raise SystemExit(
                "Unknown directories inside subject folder {0}: {1}".format(
                    root, ", ".join(unknown)
                )
            )
        yield root
        return

    found: list[Path] = []
    unknown: list[str] = []
    for child in _child_dirs(root):
        if is_subject_dir(child):
            found.append(child)
        else:
            unknown.append(child.name)
    if unknown:
        raise SystemExit(
            "Unknown directories under {0} (no thick_*.raw): {1}".format(
                root, ", ".join(unknown)
            )
        )
    if not found:
        raise SystemExit(
            "no subject folders with thick_*.raw under {0}".format(root)
        )
    n = 0
    for child in found:
        yield child
        n += 1
        if max_subjects is not None and n >= int(max_subjects):
            return


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
