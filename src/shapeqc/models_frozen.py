"""Frozen KPCA-logistic models from the paper runs.

3-channel: curv_full_recall80 (thick+logjac+curv).
2-channel: 24239 (thick+logjac), used when a subject has no curvature/mesh.
Tau is the validation recall-floor threshold (seed 42), not a tau refit on the new tree.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shapeqc.io import load_channels

_THRESHOLDS_CSV = Path(__file__).resolve().parents[2] / "configs" / "thresholds.csv"
_CHECKSUMS_CSV = Path(__file__).resolve().parents[2] / "configs" / "joblib_sha256.csv"


def _dir_from_env(name: str) -> Path | None:
    env = os.environ.get(name, "").strip()
    if not env:
        return None
    p = Path(env)
    if not p.is_dir():
        raise SystemExit("{0} is not a directory: {1}".format(name, env))
    return p


def models_3ch_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_dir():
            raise SystemExit("3-channel model directory is not a directory: {0}".format(p))
        return p
    env = _dir_from_env("SHAPEQC_MODEL_3CH")
    if env is not None:
        return env
    raise SystemExit(
        "No 3-channel model directory. Set SHAPEQC_MODEL_3CH to the "
        "curv_full_recall80 training folder (do not hardcode cluster paths)."
    )


def models_2ch_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_dir():
            raise SystemExit("2-channel model directory is not a directory: {0}".format(p))
        return p
    env = _dir_from_env("SHAPEQC_MODEL_2CH")
    if env is not None:
        return env
    raise SystemExit(
        "No 2-channel model directory. Set SHAPEQC_MODEL_2CH to the 24239 "
        "training folder (thick+logjac KPCA .joblib files)."
    )


def apply_tau_override(
    taus: dict[tuple[str, int, int], float],
    override,
) -> dict[tuple[str, int, int], float]:
    """Replace paper taus. override is None, a float, or {roi: tau} / nested dict."""
    if override is None or override == "":
        return taus
    if isinstance(override, (int, float)):
        return {k: float(override) for k in taus}
    if not isinstance(override, dict):
        raise SystemExit("TAU must be None, a number, or a dict of roi -> tau")
    out = dict(taus)
    for roi_key, val in override.items():
        roi = int(roi_key)
        if isinstance(val, dict):
            for ch, tau in val.items():
                for key in list(out):
                    if key[0] == str(ch) and key[1] == roi:
                        out[key] = float(tau)
        else:
            for key in list(out):
                if key[1] == roi:
                    out[key] = float(val)
    return out


def load_thresholds(path: str | Path | None = None) -> dict[tuple[str, int, int], float]:
    csv_path = Path(path) if path else _THRESHOLDS_CSV
    if not csv_path.is_file():
        raise SystemExit("Missing thresholds CSV: {0}".format(csv_path))
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        out[(str(r["channel_set"]), int(r["roi"]), int(r["seed"]))] = float(r["tau"])
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_joblib_checksums(path: str | Path | None = None) -> dict[tuple[str, int, int], str]:
    csv_path = Path(path) if path else _CHECKSUMS_CSV
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        out[(str(r["channel_set"]), int(r["roi"]), int(r["seed"]))] = str(r["sha256"]).lower()
    return out


def verify_joblib(path: Path, *, channel_set: str, roi: int, seed: int, checksums: dict) -> None:
    key = (str(channel_set), int(roi), int(seed))
    expected = checksums.get(key)
    if not expected:
        return
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(
            "SHA256 mismatch for {0} roi={1} seed={2} file {3}. "
            "Expected {4}, got {5}. joblib files are pickle: only load "
            "trusted paper weights.".format(channel_set, roi, seed, path.name, expected, got)
        )


def find_joblib(roi: int, seed: int, model_dir: Path) -> Path:
    needle = "_roi{0}_seed{1}_kpca_logistic.joblib".format(int(roi), int(seed))
    hits = sorted(p for p in Path(model_dir).glob("*" + needle) if p.name.endswith(needle))
    if not hits:
        raise SystemExit("No KPCA joblib for roi={0} seed={1} in {2}".format(roi, seed, model_dir))
    if len(hits) > 1:
        raise SystemExit(
            "Multiple KPCA joblibs for roi={0} seed={1} in {2}: {3}".format(
                roi, seed, model_dir, ", ".join(p.name for p in hits)
            )
        )
    return hits[0]


def find_2ch_joblib(roi: int, seed: int = 42, model_dir: Path | None = None) -> Path:
    return find_joblib(roi, seed, models_2ch_dir(model_dir))


def _unwrap_pipe(blob):
    if isinstance(blob, dict) and "pipeline" in blob:
        return blob["pipeline"]
    return blob


def score_frozen(
    manifest: pd.DataFrame,
    *,
    roi: int,
    features: list[str],
    model_dir: Path,
    seed: int = 42,
    pipe=None,
) -> np.ndarray:
    if pipe is None:
        pipe = _unwrap_pipe(joblib.load(find_joblib(roi, seed, model_dir)))
    n_in = int(getattr(pipe, "n_features_in_", 0))
    scores = []
    for _, row in manifest.iterrows():
        x = load_channels(
            row["subject_dir"],
            int(row["roi"]),
            features=features,
            require=features,
        )
        if n_in and int(x.size) != n_in:
            raise SystemExit(
                "Model for ROI {0} expects {1} features (atlas vertex count), "
                "got {2} for subject {3}. Frozen paper models only fit real ENIGMA "
                "meshes, not the tiny example.".format(roi, n_in, x.size, row["subject_id"])
            )
        scores.append(float(pipe.predict_proba(x.reshape(1, -1))[0, 1]))
    return np.asarray(scores, dtype=np.float64)


def score_frozen_2ch(
    manifest: pd.DataFrame,
    *,
    roi: int,
    seed: int = 42,
    model_dir: Path | None = None,
) -> np.ndarray:
    return score_frozen(
        manifest,
        roi=roi,
        features=["thick", "logjac"],
        model_dir=models_2ch_dir(model_dir),
        seed=seed,
    )


def score_frozen_3ch(
    manifest: pd.DataFrame,
    *,
    roi: int,
    seed: int = 42,
    model_dir: Path | None = None,
) -> np.ndarray:
    return score_frozen(
        manifest,
        roi=roi,
        features=["thick", "logjac", "curv"],
        model_dir=models_3ch_dir(model_dir),
        seed=seed,
    )
