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
    desired_recall: Any = None
    interpolator: str = ""
    pairs: Any = None
    max_subjects: int = 0
    ccbbm: str = ""
    write_curvature: bool = True
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


def _as_bool(val, default: bool = True) -> bool:
    if val is None or val == "":
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    text = str(val).strip().lower()
    if text in ("0", "false", "no", "off"):
        return False
    if text in ("1", "true", "yes", "on"):
        return True
    return default


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
        desired_recall=getattr(mod, "DESIRED_RECALL", None),
        interpolator=str(getattr(mod, "INTERPOLATOR", "") or ""),
        pairs=getattr(mod, "PAIRS", None),
        max_subjects=int(getattr(mod, "MAX_SUBJECTS", 0) or 0),
        ccbbm=str(getattr(mod, "CCBBM", "") or ""),
        write_curvature=_as_bool(getattr(mod, "WRITE_CURVATURE", True), True),
        path=path,
    )


def score_options_from_local(local: LocalSettings) -> dict:
    """run_score kwargs shared by qc-report and the notebook.

    Does not include out_dir: only qc-report writes qc_report.md / predictions.csv.
    """
    desired_recall = local.desired_recall
    if desired_recall is not None and desired_recall != "":
        desired_recall = float(desired_recall)
    else:
        desired_recall = None
    rois = [int(x) for x in local.rois] if local.rois else None
    return {
        "rois": rois,
        "models_3ch": local.model_3ch or None,
        "models_2ch": local.model_2ch or None,
        "tau_override": local.tau,
        "max_subjects": local.max_subjects or None,
        "interpolators": local.interpolator or None,
        "desired_recall": desired_recall,
        "pairs": local.pairs,
        "ccbbm": local.ccbbm or None,
        "write_curvature": bool(local.write_curvature),
    }
