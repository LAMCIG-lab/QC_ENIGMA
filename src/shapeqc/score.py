"""Unlabeled Shape tree → frozen models → review list.

Researcher path: .raw files in, trained KPCA-logistic out.
3-channel (thick+logjac+curv) when curvature exists; 2-channel (24239)
when there is no mesh and no curvature.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shapeqc.io import DEFAULT_ROIS, inventory_row, iter_subjects, load_channels
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
from shapeqc.paper_meta import frozen_run_blurb, load_roi_paper_meta, run_id_for_channel_set
from shapeqc.report import FEATURES_2CH, FEATURES_3CH, ROI_NAMES, ensure_curvature

CSV_COLUMNS = (
    "subject_id",
    "roi",
    "roi_name",
    "P_fail",
    "tau",
    "flag",
    "paper_deployable",
    "n_fail_paper",
    "channel_set",
    "model",
    "run_id",
    "files",
    "thick_file",
    "logjac_file",
    "curv_file",
)


def _files_used(row: pd.Series, channel_set: str) -> str:
    parts = [row.get("thick_file") or "", row.get("logjac_file") or ""]
    if channel_set == "thick+logjac+curv":
        parts.append(row.get("curv_file") or "")
    return ", ".join(p for p in parts if p)


def _inventory(
    shape_root: str | Path, rois: list[int], *, max_subjects: int | None = None
) -> pd.DataFrame:
    rows = []
    incomplete = []
    for subj in iter_subjects(shape_root, max_subjects=max_subjects):
        for roi in rois:
            rec = inventory_row(subj, roi)
            if rec["thick"] and rec["logjac"]:
                rows.append(rec)
            elif rec["thick"] or rec["logjac"]:
                incomplete.append("{0} roi={1}".format(rec["subject_id"], rec["roi"]))
    if incomplete:
        raise SystemExit(
            "incomplete thick/LogJacs pair (need both): {0}".format("; ".join(incomplete))
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no thick_*.raw + LogJacs_*.raw pairs under {0}".format(shape_root))
    df["channel_set"] = np.where(df["curv"], "thick+logjac+curv", "thick+logjac")
    return df


def _pipe_for(
    roi: int,
    seed: int,
    model_dir: Path,
    cache: dict,
    *,
    channel_set: str,
    checksums: dict,
    verify_checksums: bool,
):
    key = (str(model_dir), int(roi), int(seed))
    if key not in cache:
        path = find_joblib(roi, seed, model_dir)
        if verify_checksums:
            verify_joblib(path, channel_set=channel_set, roi=roi, seed=seed, checksums=checksums)
        cache[key] = _unwrap_pipe(joblib.load(path))
    return cache[key]


def score_rows_frozen(
    inv: pd.DataFrame,
    *,
    seed: int = 42,
    models_3ch: Path | None = None,
    models_2ch: Path | None = None,
    thresholds: dict | None = None,
    verify_checksums: bool = True,
) -> pd.DataFrame:
    taus = thresholds if thresholds is not None else load_thresholds()
    cache = {}
    dir3 = None
    dir2 = None
    rows = []
    meta = load_roi_paper_meta()
    checksums = load_joblib_checksums() if verify_checksums else {}
    for _, row in inv.iterrows():
        roi = int(row["roi"])
        if bool(row["curv"]):
            channel_set = "thick+logjac+curv"
            features = FEATURES_3CH
            if dir3 is None:
                dir3 = models_3ch_dir(models_3ch)
            model_dir = dir3
            model = "kpca_logistic_curv_full"
        else:
            channel_set = "thick+logjac"
            features = FEATURES_2CH
            if dir2 is None:
                dir2 = models_2ch_dir(models_2ch)
            model_dir = dir2
            model = "kpca_logistic_24239"
        key = (channel_set, roi, seed)
        if key not in taus:
            raise SystemExit("No validation tau for {0} roi={1} seed={2}".format(channel_set, roi, seed))
        tau = float(taus[key])
        pipe = _pipe_for(
            roi,
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
                "Frozen paper models need real ENIGMA vertex counts.".format(
                    roi, channel_set, n_in, x.size, row["subject_id"]
                )
            )
        pfail = float(pipe.predict_proba(x.reshape(1, -1))[0, 1])
        info = meta.get(roi, {})
        rec = {
            "subject_id": row["subject_id"],
            "roi": roi,
            "roi_name": ROI_NAMES.get(roi, ""),
            "P_fail": pfail,
            "tau": tau,
            "flag": int(pfail >= tau),
            "paper_deployable": bool(info.get("paper_deployable", False)),
            "n_fail_paper": info.get("n_fail_paper"),
            "channel_set": channel_set,
            "model": model,
            "run_id": run_id_for_channel_set(channel_set),
            "files": _files_used(row, channel_set),
            "thick_file": row.get("thick_file", ""),
            "logjac_file": row.get("logjac_file", ""),
            "curv_file": row.get("curv_file", "") if channel_set == "thick+logjac+curv" else "",
        }
        rows.append(rec)
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
) -> dict:
    roi_list = list(rois) if rois else list(DEFAULT_ROIS)
    skipped = ensure_curvature(shape_root, roi_list, max_subjects=max_subjects)
    inv = _inventory(shape_root, roi_list, max_subjects=max_subjects)
    taus = apply_tau_override(
        load_thresholds(thresholds) if thresholds else load_thresholds(),
        tau_override,
    )
    preds = score_rows_frozen(
        inv,
        seed=seed,
        models_3ch=Path(models_3ch) if models_3ch else None,
        models_2ch=Path(models_2ch) if models_2ch else None,
        thresholds=taus,
        verify_checksums=verify_checksums,
    )
    review = preds.loc[preds["flag"] == 1].copy()
    meta = load_roi_paper_meta()
    per = []
    for roi in roi_list:
        g = preds.loc[preds["roi"] == roi] if len(preds) else preds
        info = meta.get(int(roi), {})
        rec = {
            "roi": str(int(roi)),
            "roi_name": ROI_NAMES.get(int(roi), ""),
            "n": int(len(g)),
            "n_flagged": int(g["flag"].sum()) if len(g) else 0,
            "flagged_frac": float(g["flag"].mean()) if len(g) else float("nan"),
            "n_3ch": int((g["channel_set"] == "thick+logjac+curv").sum()) if len(g) else 0,
            "n_2ch": int((g["channel_set"] == "thick+logjac").sum()) if len(g) else 0,
            "tau": float(g["tau"].iloc[0]) if len(g) else float("nan"),
            "paper_deployable": bool(info.get("paper_deployable", False)),
            "n_fail_paper": info.get("n_fail_paper"),
            "verdict_reason": info.get("verdict_reason", ""),
        }
        per.append(rec)
    n_scarce_flagged = int((~review["paper_deployable"]).sum()) if len(review) else 0
    macro = {
        "n_rows": int(len(preds)),
        "n_flagged": int(len(review)),
        "n_scarce_flagged": n_scarce_flagged,
        "flagged_frac": float(preds["flag"].mean()) if len(preds) else float("nan"),
        "n_3ch": int((preds["channel_set"] == "thick+logjac+curv").sum()) if len(preds) else 0,
        "n_2ch": int((preds["channel_set"] == "thick+logjac").sum()) if len(preds) else 0,
        "n_no_mesh_no_curv": len(skipped),
        "features": FEATURES_3CH,
        "run_3ch": "curv_full_recall80",
        "run_2ch": "24239",
        "threshold_source": "training validation tau (not refit); override via local_paths.TAU",
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
        (out / "qc_report.md").write_text(render_score_markdown(payload))
        export = preds[[c for c in CSV_COLUMNS if c in preds.columns]].copy()
        export.to_csv(out / "predictions.csv", index=False)
    return payload


def render_score_markdown(payload: dict) -> str:
    m = payload["macro"]
    rows = payload["per_roi"]
    review = payload.get("review_list") or []
    scarce = [r for r in rows if not r.get("paper_deployable")]
    lines = [
        "# QC report (checking strategy)",
        "",
        "This list is a **checking strategy** for researchers: which subject × ROI "
        "files to look at. It is **not** an automatic fail detector and **not** a "
        "medical device. `flag=1` means open these files, not “this scan failed QC”. "
        "`P_fail` is a ranking score at a recall-floor tau; it is not comparable "
        "across ROIs.",
        "",
        frozen_run_blurb(),
        "",
        "Tau is the **training validation** recall-floor unless you set `TAU` in "
        "`local_paths.py` (example: `TAU = {11: 0.6}` to shrink the caudate queue). "
        "Default channels: **thick+logjac+curv**. No mesh and no curvature → "
        "**thick+logjac** (24239).",
        "",
        "## Scarce-label ROIs",
        "",
        "Only ROI **13, 17, 53** (pallidum / hippocampus) met the paper "
        "deployability floors. Other ROIs still get a score, but training had few "
        "fail labels (ROI 11 L-caudate: 9 fails in ~5409 rows, test precision ~2.5%). "
        "Do not read a caudate flag like a hippocampus flag. Raise `TAU` for those "
        "ROIs if the queue is too large.",
        "",
        "| | |",
        "|--|--:|",
        "| rows scored | {0} |".format(m["n_rows"]),
        "| flagged for review | {0} |".format(m["n_flagged"]),
        "| flagged on scarce ROIs | {0} |".format(m.get("n_scarce_flagged", 0)),
        "| flagged fraction | {0:.3f} |".format(m["flagged_frac"]),
        "| 3-channel (has curv) | {0} |".format(m["n_3ch"]),
        "| 2-channel fallback | {0} |".format(m["n_2ch"]),
        "| 3ch run | {0} |".format(m.get("run_3ch", "curv_full_recall80")),
        "| 2ch run | {0} |".format(m.get("run_2ch", "24239")),
        "",
        "## Per ROI",
        "",
        "| ROI | Name | flagged | n | fraction | tau | deployable | n_fail (paper) | 3ch | 2ch |",
        "|-----|------|--------:|--:|---------:|----:|:----------:|---------------:|----:|----:|",
    ]
    for r in rows:
        frac = "{0:.3f}".format(r["flagged_frac"]) if r["n"] else "—"
        tau = "{0:.4f}".format(r["tau"]) if r["n"] else "—"
        dep = "yes" if r.get("paper_deployable") else "no"
        nfail = r.get("n_fail_paper")
        nfail_s = str(nfail) if nfail is not None else "—"
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |".format(
                r["roi"],
                r["roi_name"],
                r["n_flagged"],
                r["n"],
                frac,
                tau,
                dep,
                nfail_s,
                r["n_3ch"],
                r["n_2ch"],
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
    if scarce:
        lines.extend(
            [
                "",
                "Scarce-label ROIs in this table: "
                + ", ".join("{0} {1}".format(r["roi"], r["roi_name"]).strip() for r in scarce),
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Review list (files to check)",
            "",
            "`flag` means `P_fail ≥ tau`. Open the files in the last column. "
            "`deployable=no` rows are a weaker reference.",
            "",
        ]
    )
    if not review:
        lines.append("No samples flagged.")
        lines.append("")
    else:
        lines.extend(
            [
                "| subject | ROI | name | P_fail | tau | deployable | files |",
                "|---------|-----|------|-------:|----:|:----------:|-------|",
            ]
        )
        for rec in review:
            dep = "yes" if rec.get("paper_deployable") else "no"
            lines.append(
                "| {0} | {1} | {2} | {3:.4f} | {4:.4f} | {5} | {6} |".format(
                    rec["subject_id"],
                    rec["roi"],
                    rec.get("roi_name", ""),
                    float(rec["P_fail"]),
                    float(rec["tau"]),
                    dep,
                    rec.get("files", ""),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"
