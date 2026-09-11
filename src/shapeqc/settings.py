"""Load gitignored local_paths.py (SHAPE_ROOT, OUT_DIR, TAU, models)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class LocalSettings:
    shape_root: str = ""
    out_dir: str = "outputs/qc"
    rois: Optional[Sequence[int]] = None
    tau: Any = None
    model_3ch: str = ""
    model_2ch: str = ""
    max_subjects: int = 0
    path: Optional[Path] = None


def _candidates() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        Path.cwd() / "local_paths.py",
        here.parents[2] / "local_paths.py",
    ]


def find_local_paths_file() -> Optional[Path]:
    seen = set()
    for path in _candidates():
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            return path
    return None


def load_local_settings() -> LocalSettings:
    path = find_local_paths_file()
    if path is None:
        return LocalSettings()
    spec = importlib.util.spec_from_file_location("shapeqc_local_paths", path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load {0}".format(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rois = getattr(mod, "ROIS", None)
    if rois is not None:
        rois = [int(x) for x in rois]
    return LocalSettings(
        shape_root=str(getattr(mod, "SHAPE_ROOT", "") or ""),
        out_dir=str(getattr(mod, "OUT_DIR", "outputs/qc") or "outputs/qc"),
        rois=rois,
        tau=getattr(mod, "TAU", None),
        model_3ch=str(getattr(mod, "MODEL_3CH", "") or ""),
        model_2ch=str(getattr(mod, "MODEL_2CH", "") or ""),
        max_subjects=int(getattr(mod, "MAX_SUBJECTS", 0) or 0),
        path=path,
    )
