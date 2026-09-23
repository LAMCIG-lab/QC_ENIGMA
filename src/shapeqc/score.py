"""Unlabeled Shape tree → frozen models → review list.

.raw files in, trained KPCA-logistic out.
3-channel (thick+logjac+curv) when curvature exists; 2-channel (24239)
when there is no mesh and no curvature.
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from shapeqc.curvature import ensure_curvature
from shapeqc.interpolate import (
    DEFAULT_PAIRS,
    load_interpolator_table,
    parse_roi_key,
    roi_token,
    tau_from_table,
)
from shapeqc.io import (
    DEFAULT_ROIS,
    FEATURES_2CH,
    FEATURES_3CH,
    ROI_NAMES,
    inventory_row,
    iter_subjects,
    load_channels,
)
from shapeqc.models_frozen import (
    _unwrap_pipe,
    apply_tau_override,
    find_joblib,
    load_joblib_checksums,
    load_thresholds,
    models_2ch_dir,
    models_3ch_dir,
    verify_joblib,
)

_DEPLOY_CSV = Path(__file__).resolve().parents[2] / "configs" / "deployable.csv"


def load_deployable(path: str | Path | None = None) -> dict[str, bool]:
    csv_path = Path(path) if path else _DEPLOY_CSV
    out: dict[str, bool] = {}
    if not csv_path.is_file():
        return out
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        flag = str(row.get("deployable", "")).strip().lower() in ("yes", "true", "1")
        out[str(row["roi"])] = flag
    return out


RUN_3CH = "curv_full_recall80"
RUN_2CH = "24239"

CSV_COLUMNS = (
    "subject_id",
    "roi",
    "roi_name",
    "P_fail",
    "tau",
    "flag",
    "channel_set",
    "model",
    "run_id",
    "files",
    "thick_file",
    "logjac_file",
    "curv_file",
    "pair",
    "desired_recall",
    "paper_deployable",
)


def _run_id(channel_set: str) -> str:
    return RUN_3CH if channel_set == "thick+logjac+curv" else RUN_2CH


def _files_used(row: pd.Series, channel_set: str) -> str:
    parts = [row.get("thick_file") or "", row.get("logjac_file") or ""]
    if channel_set == "thick+logjac+curv":
        parts.append(row.get("curv_file") or "")
    return ", ".join(p for p in parts if p)


def _inventory(
    shape_root: str | Path,
    rois: list[int],
    *,
    max_subjects: int | None = None,
    subjects: list[Path] | None = None,
) -> pd.DataFrame:
    rows = []
    incomplete = []
    subj_iter = subjects if subjects is not None else list(
        iter_subjects(shape_root, max_subjects=max_subjects)
    )
    for subj in subj_iter:
        for roi in rois:
            rec = inventory_row(subj, roi)
            if rec["thick"] and rec["logjac"]:
                rows.append(rec)
            elif rec["thick"] or rec["logjac"]:
                incomplete.append("{0} roi={1}".format(rec["subject_id"], rec["roi"]))
    if incomplete:
        shown = "; ".join(incomplete[:12])
        extra = " ..." if len(incomplete) > 12 else ""
        sys.stderr.write(
            "shapeqc: skipping {0} incomplete thick/LogJacs pair(s): {1}{2}\n".format(
                len(incomplete), shown, extra
            )
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no thick_*.raw + LogJacs_*.raw pairs under {0}".format(shape_root))
    df["channel_set"] = np.where(df["curv"], "thick+logjac+curv", "thick+logjac")
    return df


def _pipe_for(
    roi,
    seed: int,
    model_dir: Path,
    cache: dict,
    *,
    channel_set: str,
    checksums: dict,
    verify_checksums: bool,
):
    key = (str(model_dir), roi_token(roi), int(seed))
    if key not in cache:
        path = find_joblib(roi, seed, model_dir)
        if verify_checksums:
            verify_joblib(path, channel_set=channel_set, roi=roi, seed=seed, checksums=checksums)
        cache[key] = _unwrap_pipe(joblib.load(path))
    return cache[key]


def _lookup_tau(
    *,
    channel_set: str,
    roi,
    seed: int,
    taus: dict,
    interpolators,
    desired_recall,
    interp_method: str,
) -> float:
    if interpolators is not None and desired_recall is not None:
        return tau_from_table(
            interpolators,
            channel_set=channel_set,
            roi=roi,
            seed=seed,
            desired_recall=float(desired_recall),
            method=interp_method,
        )
    key = (channel_set, parse_roi_key(roi), seed)
    if key not in taus:
        raise SystemExit(
            "No validation tau for {0} roi={1} seed={2}".format(
                channel_set, roi_token(roi), seed
            )
        )
    return float(taus[key])


def _score_one(
    row: pd.Series,
    *,
    roi_model,
    channel_set: str,
    features: list[str],
    model_dir: Path,
    model: str,
    seed: int,
    cache: dict,
    checksums: dict,
    verify_checksums: bool,
    taus: dict,
    interpolators,
    desired_recall,
    interp_method: str,
    pair: str,
) -> dict:
    roi = int(row["roi"])
    tau = _lookup_tau(
        channel_set=channel_set,
        roi=roi_model,
        seed=seed,
        taus=taus,
        interpolators=interpolators,
        desired_recall=desired_recall,
        interp_method=interp_method,
    )
    pipe = _pipe_for(
        roi_model,
        seed,
        model_dir,
        cache,
        channel_set=channel_set,
        checksums=checksums,
        verify_checksums=verify_checksums,
    )
    n_in = int(getattr(pipe, "n_features_in_", 0))
    x = load_channels(row["subject_dir"], roi, features=features, require=features)
    if n_in and int(x.size) != n_in:
        raise SystemExit(
            "Model for ROI {0} ({1}) expects {2} features, got {3} for {4}. "
            "Frozen models need real ENIGMA vertex counts.".format(
                roi_token(roi_model), channel_set, n_in, x.size, row["subject_id"]
            )
        )
    pfail = float(pipe.predict_proba(x.reshape(1, -1))[0, 1])
    return {
        "subject_id": row["subject_id"],
        "roi": roi,
        "roi_name": ROI_NAMES.get(roi, ""),
        "P_fail": pfail,
        "tau": tau,
        "flag": int(pfail >= tau),
        "channel_set": channel_set,
        "model": model,
        "run_id": _run_id(channel_set),
        "files": _files_used(row, channel_set),
        "thick_file": row.get("thick_file", ""),
        "logjac_file": row.get("logjac_file", ""),
        "curv_file": row.get("curv_file", "") if channel_set == "thick+logjac+curv" else "",
        "pair": pair,
        "desired_recall": float(desired_recall) if desired_recall is not None else None,
    }


def score_rows_frozen(
    inv: pd.DataFrame,
    *,
    seed: int = 42,
    models_3ch: Path | None = None,
    models_2ch: Path | None = None,
    thresholds: dict | None = None,
    verify_checksums: bool = True,
    interpolators=None,
    desired_recall: float | None = None,
    interp_method: str = "floor",
    pairs: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    taus = thresholds if thresholds is not None else load_thresholds()
    cache = {}
    dir3 = None
    dir2 = None
    rows = []
    checksums = load_joblib_checksums() if verify_checksums else {}

    def _channel_ctx(row: pd.Series):
        nonlocal dir3, dir2
        if bool(row["curv"]):
            if dir3 is None:
                dir3 = models_3ch_dir(models_3ch)
            return "thick+logjac+curv", FEATURES_3CH, dir3, "kpca_logistic_curv_full"
        if dir2 is None:
            dir2 = models_2ch_dir(models_2ch)
        return "thick+logjac", FEATURES_2CH, dir2, "kpca_logistic_24239"

    for _, row in inv.iterrows():
        channel_set, features, model_dir, model = _channel_ctx(row)
        rows.append(
            _score_one(
                row,
                roi_model=int(row["roi"]),
                channel_set=channel_set,
                features=features,
                model_dir=model_dir,
                model=model,
                seed=seed,
                cache=cache,
                checksums=checksums,
                verify_checksums=verify_checksums,
                taus=taus,
                interpolators=interpolators,
                desired_recall=desired_recall,
                interp_method=interp_method,
                pair="",
            )
        )
    pair_iter = DEFAULT_PAIRS if pairs is True else (pairs or [])
    for left, right in pair_iter:
        pair_tok = roi_token((left, right))
        for _, row in inv.iterrows():
            hemi = int(row["roi"])
            if hemi not in (int(left), int(right)):
                continue
            channel_set, features, model_dir, _model = _channel_ctx(row)
            rows.append(
                _score_one(
                    row,
                    roi_model=pair_tok,
                    channel_set=channel_set,
                    features=features,
                    model_dir=model_dir,
                    model="kpca_logistic_pair_{0}".format(pair_tok),
                    seed=seed,
                    cache=cache,
                    checksums=checksums,
                    verify_checksums=verify_checksums,
                    taus=taus,
                    interpolators=interpolators,
                    desired_recall=desired_recall,
                    interp_method=interp_method,
                    pair=pair_tok,
                )
            )
    return pd.DataFrame(rows)


def run_score(
    shape_root: str | Path,
    *,
    rois: list[int] | None = None,
    out_dir: str | Path | None = None,
    seed: int = 42,
    models_3ch: str | Path | None = None,
    models_2ch: str | Path | None = None,
    thresholds: str | Path | None = None,
    tau_override=None,
    max_subjects: int | None = None,
    verify_checksums: bool = True,
    interpolators: str | Path | None = None,
    desired_recall: float | None = None,
    interp_method: str = "floor",
    pairs: list[tuple[int, int]] | None = None,
    write_curvature: bool = True,
    ccbbm: str | Path | None = None,
) -> dict:
    roi_list = list(rois) if rois else list(DEFAULT_ROIS)
    subjects = list(iter_subjects(shape_root, max_subjects=max_subjects))
    curv_info = ensure_curvature(
        shape_root,
        roi_list,
        max_subjects=max_subjects,
        subjects=subjects,
        write_curvature=write_curvature,
        ccbbm=ccbbm,
    )
    inv = _inventory(shape_root, roi_list, max_subjects=max_subjects, subjects=subjects)
    taus = apply_tau_override(
        load_thresholds(thresholds) if thresholds else load_thresholds(),
        tau_override,
    )
    interp_table = None
    interp_path = interpolators
    if interp_path:
        p = Path(interp_path)
        if not p.is_file():
            alt = Path(__file__).resolve().parents[2] / p
            if alt.is_file():
                p = alt
        interp_table = load_interpolator_table(p)
    elif desired_recall is not None:
        default_interp = Path(__file__).resolve().parents[2] / "configs" / "interpolator.csv"
        if not default_interp.is_file():
            raise SystemExit("desired_recall requires interpolator CSV ({0})".format(default_interp))
        interp_table = load_interpolator_table(default_interp)
    pair_list = None
    if pairs is True:
        pair_list = list(DEFAULT_PAIRS)
    elif pairs:
        pair_list = list(pairs)
    preds = score_rows_frozen(
        inv,
        seed=seed,
        models_3ch=Path(models_3ch) if models_3ch else None,
        models_2ch=Path(models_2ch) if models_2ch else None,
        thresholds=taus,
        verify_checksums=verify_checksums,
        interpolators=interp_table,
        desired_recall=desired_recall,
        interp_method=interp_method,
        pairs=pair_list,
    )
    deploy = load_deployable()

    def _deploy_flag(row) -> str:
        pair = str(row.get("pair") or "")
        key = pair if pair else str(int(row["roi"]))
        return "yes" if deploy.get(key, False) else "no"

    if len(preds):
        preds = preds.copy()
        preds["paper_deployable"] = [_deploy_flag(r) for _, r in preds.iterrows()]
    singles = preds.loc[preds["pair"].fillna("") == ""] if "pair" in preds.columns else preds
    review = preds.loc[preds["flag"] == 1].copy()
    n_subjects = int(singles["subject_id"].nunique()) if len(singles) else 0
    per = []
    for roi in roi_list:
        g = singles.loc[singles["roi"] == roi] if len(singles) else singles
        rec = {
            "roi": str(int(roi)),
            "roi_name": ROI_NAMES.get(int(roi), ""),
            "n": int(len(g)),
            "n_flagged": int(g["flag"].sum()) if len(g) else 0,
            "flagged_frac": float(g["flag"].mean()) if len(g) else float("nan"),
            "n_3ch": int((g["channel_set"] == "thick+logjac+curv").sum()) if len(g) else 0,
            "n_2ch": int((g["channel_set"] == "thick+logjac").sum()) if len(g) else 0,
            "tau": float(g["tau"].iloc[0]) if len(g) else float("nan"),
            "paper_deployable": "yes" if deploy.get(str(int(roi)), False) else "no",
        }
        per.append(rec)
    macro = {
        "n_subjects": n_subjects,
        "n_rows": int(len(preds)),
        "n_flagged": int(len(review)),
        "flagged_frac": float(preds["flag"].mean()) if len(preds) else float("nan"),
        "n_3ch": int((preds["channel_set"] == "thick+logjac+curv").sum()) if len(preds) else 0,
        "n_2ch": int((preds["channel_set"] == "thick+logjac").sum()) if len(preds) else 0,
        "n_no_mesh_no_curv": len(curv_info.get("skipped_no_mesh", [])),
        "n_curv_wrote": len(curv_info.get("wrote", [])),
        "n_curv_no_ccbbm": len(curv_info.get("skipped_no_ccbbm", [])),
        "features": FEATURES_3CH,
        "run_3ch": RUN_3CH,
        "run_2ch": RUN_2CH,
        "threshold_source": (
            "interpolator desired_recall={0}".format(desired_recall)
            if desired_recall is not None
            else "training validation tau; override via local_paths.TAU"
        ),
        "note": (
            "Checking strategy, not a fail detector. flag=1 means look at the "
            "listed .raw files. Tau is the validation recall-floor unless TAU is set. "
            "Rows without curvature use the 24239 thick+logjac model."
        ),
    }
    payload = {
        "mode": "score",
        "macro": macro,
        "per_roi": per,
        "predictions": preds.to_dict(orient="records") if len(preds) else [],
        "review_list": review.to_dict(orient="records") if len(review) else [],
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "qc_report.md").write_text(render_score_markdown(payload), encoding="utf-8")
        export = preds[[c for c in CSV_COLUMNS if c in preds.columns]].copy()
        export.to_csv(out / "predictions.csv", index=False)
    return payload


def render_score_markdown(payload: dict) -> str:
    m = payload["macro"]
    rows = payload["per_roi"]
    review = payload.get("review_list") or []
    lines = [
        "# QC report (checking strategy)",
        "",
        "This list is a **checking strategy**: which subject × ROI files to look at. "
        "It is **not** an automatic fail detector. `flag=1` means open these files. "
        "`P_fail` is a ranking score at a recall-floor tau; it is not comparable "
        "across ROIs.",
        "",
        "Tau is the **training validation** recall-floor unless you set `TAU` in "
        "`local_paths.py` (example: `TAU = {11: 0.6}`). Default channels: "
        "**thick+logjac+curv** if `curvature_*.raw` exists (or was written from mesh). "
        "No mesh and no curvature -> **thick+logjac** (24239). Only ROIs 13, 17, 53 "
        "(and pair 13_52) met paper deployability floors; other flags are a weaker reference.",
        "",
        "| | |",
        "|--|--:|",
        "| subjects | {0} |".format(m.get("n_subjects", "")),
        "| rows scored | {0} |".format(m["n_rows"]),
        "| flagged for review | {0} |".format(m["n_flagged"]),
        "| flagged fraction | {0:.3f} |".format(m["flagged_frac"]),
        "| 3-channel (has curv) | {0} |".format(m["n_3ch"]),
        "| 2-channel fallback | {0} |".format(m["n_2ch"]),
        "| 3ch run | {0} |".format(m.get("run_3ch", RUN_3CH)),
        "| 2ch run | {0} |".format(m.get("run_2ch", RUN_2CH)),
        "",
        "## Per ROI",
        "",
        "| ROI | Name | flagged | n | fraction | tau | 3ch | 2ch | deployable |",
        "|-----|------|--------:|--:|---------:|----:|----:|----:|:----------:|",
    ]
    for r in rows:
        frac = "{0:.3f}".format(r["flagged_frac"]) if r["n"] else "-"
        tau = "{0:.4f}".format(r["tau"]) if r["n"] else "-"
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |".format(
                r["roi"],
                r["roi_name"],
                r["n_flagged"],
                r["n"],
                frac,
                tau,
                r["n_3ch"],
                r["n_2ch"],
                r.get("paper_deployable", ""),
            )
        )
    present = [r for r in rows if r["n"]]
    labels = [r["roi"] for r in present]
    fracs = [r["flagged_frac"] for r in present]
    counts = [r["n_flagged"] for r in present]
    ymax = max(counts + [1])
    if labels:
        lines.extend(
            [
                "",
                "```mermaid",
                "xychart-beta",
                '  title "Flagged fraction by ROI"',
                "  x-axis [{0}]".format(", ".join(labels)),
                '  y-axis "flagged fraction" 0 --> 1',
                "  bar [{0}]".format(", ".join("{0:.3f}".format(x) for x in fracs)),
                "```",
                "",
                "```mermaid",
                "xychart-beta",
                '  title "Review queue size by ROI"',
                "  x-axis [{0}]".format(", ".join(labels)),
                '  y-axis "flagged count" 0 --> {0}'.format(ymax),
                "  bar [{0}]".format(", ".join(str(x) for x in counts)),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Review list (files to check)",
            "",
            "`flag` means `P_fail ≥ tau`. Open the files in the last column.",
            "",
        ]
    )
    if not review:
        lines.append("No samples flagged.")
        lines.append("")
    else:
        lines.extend(
            [
                "| subject | ROI | name | P_fail | tau | files |",
                "|---------|-----|------|-------:|----:|-------|",
            ]
        )
        for rec in review:
            lines.append(
                "| {0} | {1} | {2} | {3:.4f} | {4:.4f} | {5} |".format(
                    rec["subject_id"],
                    rec["roi"],
                    rec.get("roi_name", ""),
                    float(rec["P_fail"]),
                    float(rec["tau"]),
                    rec.get("files", ""),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"
