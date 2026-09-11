"""Paper-run identity and per-ROI deployability (checking strategy, not a detector)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY = _ROOT / "paper" / "deployment.csv"
_AGG = _ROOT / "paper" / "results_agg.csv"


def _as_bool(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)

# Joblibs loaded by qc-report (seed 42).
RUN_3CH_ID = "curv_full_recall80"
RUN_3CH_PBS = "24264"
RUN_3CH_REQUIRE = (
    "thick+logjac+curv; two sites (FBIRN, UCIS; 416 subjects) had no mesh, "
    "so curvature was zero-filled in training — those meshes are not reconstructed"
)
RUN_2CH_ID = "24239"
RUN_2CH_RETUNE = "24239_complete_retune_r08"


def run_id_for_channel_set(channel_set: str) -> str:
    if channel_set == "thick+logjac+curv":
        return RUN_3CH_ID
    return RUN_2CH_ID


def frozen_run_blurb() -> str:
    return (
        "Frozen weights: 3-channel **{0}** (PBS {1}; {2}). "
        "New subject folders with no mesh and no curvature use 2-channel **{3}** "
        "(retune {4}), not zero-fill. Reconstructing FBIRN/UCIS meshes is not "
        "in scope for this checking tool."
    ).format(
        RUN_3CH_ID,
        RUN_3CH_PBS,
        RUN_3CH_REQUIRE,
        RUN_2CH_ID,
        RUN_2CH_RETUNE,
    )


def load_roi_paper_meta() -> dict[int, dict]:
    """roi -> paper_deployable, n_fail_paper, verdict_reason (singles only)."""
    out: dict[int, dict] = {}
    if _DEPLOY.is_file():
        dep = pd.read_csv(_DEPLOY)
        for _, r in dep.iterrows():
            raw = str(r["roi"]).strip()
            if "," in raw:
                continue
            try:
                roi = int(float(raw))
            except ValueError:
                continue
            out[roi] = {
                "paper_deployable": _as_bool(r["deployable"]),
                "verdict_reason": str(r.get("verdict_reason", "") or ""),
                "n_fail_paper": None,
            }
    if _AGG.is_file():
        agg = pd.read_csv(_AGG)
        for _, r in agg.iterrows():
            raw = str(r["roi"]).strip()
            if "," in raw:
                continue
            try:
                roi = int(float(raw))
            except ValueError:
                continue
            rec = out.setdefault(
                roi,
                {"paper_deployable": False, "verdict_reason": "", "n_fail_paper": None},
            )
            rec["n_fail_paper"] = int(round(float(r["test_fail_mean"])))
    return out
