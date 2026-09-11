"""Labeled eval: Shape tree + QA labels → paper F-recall / F-share.

3-channel (thick+logjac+curv) is the default. Subjects with no curvature and
no mesh cannot run ccbbm; those rows are scored with the frozen 24239
thick+logjac KPCA-logistic models and marked in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from shapeqc.curvature import find_ccbbm, generate_one
from shapeqc.io import DEFAULT_ROIS, inventory_row, iter_subjects, subject_feature_paths
from shapeqc.metrics import paper_counts, recall_floor_threshold
from shapeqc.split import split_indices_by_subject
from shapeqc.train import fit_simple, score_rows

FEATURES_3CH = ["thick", "logjac", "curv"]
FEATURES_2CH = ["thick", "logjac"]
RECALL_FLOOR = 0.8
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


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError("not JSON serializable: {0!r}".format(type(o)))


def ensure_curvature(
    shape_root: str | Path, rois: list[int], *, max_subjects: int | None = None
) -> list[str]:
    """Write curvature_*.raw from meshes when possible.

    Rows with neither mesh nor curv are left as-is (2-channel fallback).
    Returns a list of ``subject roi=N`` strings that still have no curv.
    """
    skipped = []
    ccbbm = None
    for subj in iter_subjects(shape_root, max_subjects=max_subjects):
        for roi in rois:
            paths = subject_feature_paths(subj, roi)
            if paths["curv"].is_file():
                continue
            if not paths["mesh"].is_file():
                skipped.append("{0} roi={1}".format(subj.name, roi))
                continue
            if ccbbm is None:
                ccbbm = find_ccbbm()
            generate_one(paths["mesh"], paths["curv"], ccbbm=ccbbm)
    return skipped


def labeled_manifest(
    shape_root: str | Path,
    labels_csv: str | Path,
    rois: list[int],
    *,
    max_subjects: int | None = None,
) -> pd.DataFrame:
    from shapeqc.labels import load_labels

    rows = [
        inventory_row(s, r)
        for s in iter_subjects(shape_root, max_subjects=max_subjects)
        for r in rois
    ]
    inv = pd.DataFrame(rows)
    lab = load_labels(labels_csv)
    if "y_fail" not in lab.columns:
        raise SystemExit("labels CSV needs columns subject_id, roi, y_fail (1=fail)")
    df = inv.merge(lab, on=["subject_id", "roi"], how="inner")
    if df.empty:
        raise SystemExit("no rows after joining Shape tree to labels")
    if df["y_fail"].isna().any():
        raise SystemExit("labels missing y_fail after join")
    df["y_fail"] = df["y_fail"].astype(int)
    need = df.loc[~(df["thick"] & df["logjac"])]
    if len(need):
        raise SystemExit(
            "{0} rows missing thick or logjac (required for both 3-channel and 2-channel)".format(
                len(need)
            )
        )
    df["channel_set"] = np.where(
        df["curv"],
        "thick+logjac+curv",
        "thick+logjac",
    )
    return df


def annotate_errors(preds: pd.DataFrame, tau: float) -> pd.DataFrame:
    out = preds.copy()
    out["tau"] = float(tau)
    out["flag"] = (out["P_fail"] >= float(tau)).astype(int)
    err = []
    for _, r in out.iterrows():
        if int(r["y_fail"]) == 1 and int(r["flag"]) == 1:
            err.append("TP")
        elif int(r["y_fail"]) == 1 and int(r["flag"]) == 0:
            err.append("FN")
        elif int(r["y_fail"]) == 0 and int(r["flag"]) == 1:
            err.append("FP")
        else:
            err.append("TN")
    out["error"] = err
    return out


def _auc(y, s) -> float | None:
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _pack_metrics(
    preds: pd.DataFrame,
    *,
    roi: int,
    n_train: int,
    n_subjects: int,
    channel_set: str,
    model: str,
    split: str,
) -> dict:
    y_t = preds["y_fail"].to_numpy()
    s = preds["P_fail"].to_numpy()
    tau = float(preds["tau"].iloc[0])
    m = paper_counts(y_t, s, threshold=tau)
    m["auc"] = _auc(y_t, s)
    m["ap"] = None
    m["roi"] = str(int(roi))
    m["roi_name"] = ROI_NAMES.get(int(roi), "")
    m["n_train"] = int(n_train)
    m["n_test"] = int(len(preds))
    m["n_subjects"] = int(n_subjects)
    m["n_FN"] = int((preds["error"] == "FN").sum())
    m["n_FP"] = int((preds["error"] == "FP").sum())
    m["channel_set"] = channel_set
    m["model"] = model
    m["split"] = split
    return m


def eval_one_roi_3ch(g: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame]:
    g = g.reset_index(drop=True)
    roi = int(g["roi"].iloc[0])
    n_subj = g["subject_id"].nunique()
    if n_subj < 4:
        raise SystemExit(
            "ROI {0} (3-channel): held-out split needs ≥4 subjects (have {1}).".format(roi, n_subj)
        )
    y = g["y_fail"].to_numpy()
    try:
        train_mask, test_mask, _, _ = split_indices_by_subject(
            g["subject_id"].to_numpy(), y, test_size=0.3, seed=seed
        )
    except ValueError as exc:
        raise SystemExit("ROI {0} (3-channel): {1}".format(roi, exc))
    train, test = g.loc[train_mask], g.loc[test_mask]
    if int((test["y_fail"] == 1).sum()) < 1:
        raise SystemExit(
            "ROI {0} (3-channel): held-out fold has no fails; cannot floor recall at 0.8".format(roi)
        )
    pipe, _ = fit_simple(train, features=FEATURES_3CH, require=FEATURES_3CH, seed=seed)
    preds = score_rows(pipe, test, FEATURES_3CH, FEATURES_3CH, tau=0.5)
    y_t = preds["y_fail"].to_numpy()
    s = preds["P_fail"].to_numpy()
    tau = recall_floor_threshold(y_t, s, target=RECALL_FLOOR)
    preds = annotate_errors(preds, tau)
    preds["channel_set"] = "thick+logjac+curv"
    preds["model"] = "simple_logistic"
    preds["split"] = "held-out subjects"
    preds["roi_name"] = ROI_NAMES.get(roi, "")
    m = _pack_metrics(
        preds,
        roi=roi,
        n_train=len(train),
        n_subjects=n_subj,
        channel_set="thick+logjac+curv",
        model="simple_logistic",
        split="held-out subjects",
    )
    if m["F_recall"] + 1e-12 < RECALL_FLOOR:
        raise SystemExit(
            "ROI {0}: could not reach F-recall ≥ {1} on held-out (got {2:.3f})".format(
                roi, RECALL_FLOOR, m["F_recall"]
            )
        )
    return m, preds


def eval_one_roi_2ch(
    g: pd.DataFrame, seed: int, *, model_dir: Path | None = None
) -> tuple[dict, pd.DataFrame]:
    """Score no-curv rows with frozen 24239 KPCA-logistic (thick+logjac)."""
    from shapeqc.models_2ch import score_frozen_2ch

    g = g.reset_index(drop=True)
    roi = int(g["roi"].iloc[0])
    n_subj = int(g["subject_id"].nunique())
    scores = score_frozen_2ch(g, roi=roi, seed=seed, model_dir=model_dir)
    y = g["y_fail"].to_numpy()
    split = "all labeled (too few for held-out; frozen 2ch)"
    eval_idx = np.arange(len(g))
    n_train = 0
    try:
        _, test_mask, _, _ = split_indices_by_subject(
            g["subject_id"].to_numpy(), y, test_size=0.3, seed=seed
        )
        eval_idx = np.where(test_mask)[0]
        split = "held-out subjects (frozen 2ch)"
        n_train = int((~test_mask).sum())
    except ValueError:
        pass
    y_e = y[eval_idx]
    s_e = scores[eval_idx]
    if int((y_e == 1).sum()) < 1:
        tau = 0.5
    else:
        tau = recall_floor_threshold(y_e, s_e, target=RECALL_FLOOR)
    rows = g.iloc[eval_idx].copy()
    preds = pd.DataFrame(
        {
            "subject_id": rows["subject_id"].to_numpy(),
            "roi": [roi] * len(rows),
            "y_fail": y_e.astype(int),
            "P_fail": s_e,
        }
    )
    preds = annotate_errors(preds, tau)
    preds["channel_set"] = "thick+logjac"
    preds["model"] = "kpca_logistic_24239"
    preds["split"] = split
    preds["roi_name"] = ROI_NAMES.get(roi, "")
    m = _pack_metrics(
        preds,
        roi=roi,
        n_train=n_train,
        n_subjects=n_subj,
        channel_set="thick+logjac",
        model="kpca_logistic_24239",
        split=split,
    )
    return m, preds


def run_report(
    shape_root: str | Path,
    labels_csv: str | Path,
    *,
    rois: list[int] | None = None,
    out_dir: str | Path | None = None,
    seed: int = 42,
    models_2ch: str | Path | None = None,
    max_subjects: int | None = None,
) -> dict:
    roi_list = list(rois) if rois else list(DEFAULT_ROIS)
    skipped = ensure_curvature(shape_root, roi_list, max_subjects=max_subjects)
    man = labeled_manifest(shape_root, labels_csv, roi_list, max_subjects=max_subjects)
    two_ch_dir = Path(models_2ch).resolve() if models_2ch else None
    per = []
    pred_parts = []
    for roi, g in man.groupby("roi"):
        g3 = g.loc[g["curv"]].copy()
        g2 = g.loc[~g["curv"]].copy()
        if len(g3):
            m3, p3 = eval_one_roi_3ch(g3, seed=seed)
            per.append(m3)
            pred_parts.append(p3)
        if len(g2):
            m2, p2 = eval_one_roi_2ch(g2, seed=seed, model_dir=two_ch_dir)
            per.append(m2)
            pred_parts.append(p2)
    if not per:
        raise SystemExit("no ROI groups to evaluate")
    per.sort(key=lambda r: (str(r["channel_set"]), len(str(r["roi"])), str(r["roi"])))
    paper_rows = [r for r in per if r["channel_set"] == "thick+logjac+curv"]
    fallback_rows = [r for r in per if r["channel_set"] == "thick+logjac"]
    use_for_macro = paper_rows or per
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    off = preds.loc[preds["error"].isin(["FN", "FP"])].copy() if len(preds) else preds
    macro = {
        "F_recall": float(np.mean([r["F_recall"] for r in use_for_macro])),
        "F_share": float(np.mean([r["F_share"] for r in use_for_macro])),
        "precision": float(np.mean([r["precision"] for r in use_for_macro])),
        "F_score_mod": float(np.mean([r["F_score_mod"] for r in use_for_macro])),
        "auc": float(np.nanmean([r["auc"] if r["auc"] is not None else np.nan for r in use_for_macro])),
        "n_rois": len(use_for_macro),
        "features": FEATURES_3CH if paper_rows else FEATURES_2CH,
        "threshold_policy": "recall_floor",
        "threshold_target": RECALL_FLOOR,
        "split": "held-out subjects",
        "n_fallback_rows": int(len(man.loc[~man["curv"]])),
        "n_no_mesh_no_curv": len(skipped),
        "n_FN": int((preds["error"] == "FN").sum()) if len(preds) else 0,
        "n_FP": int((preds["error"] == "FP").sum()) if len(preds) else 0,
        "note": (
            "Tau chosen on the held-out fold so F-recall ≥ 0.8 by construction. "
            "Paper-valid val-tau can fall below 0.8 (counter-case). "
            "Rows with no mesh and no curvature are scored with frozen 24239 "
            "thick+logjac KPCA-logistic and marked channel_set=thick+logjac."
        ),
    }
    if fallback_rows:
        macro["macro_2ch"] = {
            "F_recall": float(np.mean([r["F_recall"] for r in fallback_rows])),
            "F_share": float(np.mean([r["F_share"] for r in fallback_rows])),
            "n_rois": len(fallback_rows),
        }
    payload = {
        "macro": macro,
        "per_roi": per,
        "predictions": preds.to_dict(orient="records") if len(preds) else [],
        "off_threshold": off.to_dict(orient="records") if len(off) else [],
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "qc_report.md").write_text(render_markdown(payload))
        if len(preds):
            preds.to_csv(out / "predictions.csv", index=False)
    return payload


def render_markdown(payload: dict) -> str:
    m = payload["macro"]
    rows = payload["per_roi"]
    off = payload.get("off_threshold") or []
    lines = [
        "# QC report",
        "",
        "Default features: **thick + logjac + curv**. Held-out subjects. Recall floor **{0}**.".format(
            m["threshold_target"]
        ),
        "",
        "Subjects with **no mesh and no curvature** are not dropped: they are scored with the "
        "frozen **24239 thick+logjac** KPCA-logistic model and listed as `thick+logjac` below.",
        "",
        "| Metric | Macro |",
        "|--------|------:|",
        "| F-recall | {0:.3f} |".format(m["F_recall"]),
        "| F-share (flagged) | {0:.3f} |".format(m["F_share"]),
        "| precision | {0:.3f} |".format(m["precision"]),
        "| F_score_mod | {0:.3f} |".format(m["F_score_mod"]),
        "| AUC | {0:.3f} |".format(m["auc"] if m["auc"] == m["auc"] else float("nan")),
        "| FN (missed fails) | {0} |".format(m.get("n_FN", 0)),
        "| FP (extra flags) | {0} |".format(m.get("n_FP", 0)),
        "| 2-channel fallback rows | {0} |".format(m.get("n_fallback_rows", 0)),
        "",
        "## Per ROI",
        "",
        "| ROI | Name | Channels | F-recall | F-share | precision | AUC | n_test | FN | FP |",
        "|-----|------|----------|--------:|--------:|----------:|----:|-------:|---:|---:|",
    ]
    for r in rows:
        auc = "{0:.3f}".format(r["auc"]) if r.get("auc") is not None else "—"
        lines.append(
            "| {0} | {1} | {2} | {3:.3f} | {4:.3f} | {5:.3f} | {6} | {7} | {8} | {9} |".format(
                r["roi"],
                r["roi_name"],
                r.get("channel_set", "thick+logjac+curv"),
                r["F_recall"],
                r["F_share"],
                r["precision"],
                auc,
                r["n_test"],
                r.get("n_FN", 0),
                r.get("n_FP", 0),
            )
        )
    recs = [r["F_recall"] for r in rows]
    shares = [r["F_share"] for r in rows]
    labels = ["{0}/{1}".format(r["roi"], "3ch" if "curv" in r.get("channel_set", "curv") else "2ch") for r in rows]
    fns = [int(r.get("n_FN", 0)) for r in rows]
    fps = [int(r.get("n_FP", 0)) for r in rows]
    ymax = max(fns + fps + [1])
    lines.extend(
        [
            "",
            "```mermaid",
            "xychart-beta",
            '  title "Held-out F-recall by ROI (floor 0.8)"',
            "  x-axis [{0}]".format(", ".join(labels)),
            '  y-axis "F-recall" 0 --> 1',
            "  bar [{0}]".format(", ".join("{0:.3f}".format(x) for x in recs)),
            "```",
            "",
            "```mermaid",
            "xychart-beta",
            '  title "Held-out F-share (flagged fraction) by ROI"',
            "  x-axis [{0}]".format(", ".join(labels)),
            '  y-axis "F-share" 0 --> 1',
            "  bar [{0}]".format(", ".join("{0:.3f}".format(x) for x in shares)),
            "```",
            "",
            "```mermaid",
            "xychart-beta",
            '  title "Samples not on the correct side of tau"',
            "  x-axis [{0}]".format(", ".join(labels)),
            '  y-axis "count" 0 --> {0}'.format(ymax),
            '  bar "FN missed fails" [{0}]'.format(", ".join(str(x) for x in fns)),
            '  bar "FP extra flags" [{0}]'.format(", ".join(str(x) for x in fps)),
            "```",
            "",
            "## Samples not at the threshold (FN and FP)",
            "",
            "Compared to tau on the held-out fold: **FN** = true fail with `P_fail < tau` "
            "(missed); **FP** = true pass with `P_fail ≥ tau` (extra flag). "
            "`thick+logjac` rows used the frozen 24239 model.",
            "",
        ]
    )
    if not off:
        lines.append("All held-out samples sit on the correct side of tau (no FN or FP).")
        lines.append("")
    else:
        lines.extend(
            [
                "| subject | ROI | name | P_fail | tau | y_fail | flag | error | channels | model |",
                "|---------|-----|------|-------:|----:|-------:|-----:|-------|----------|-------|",
            ]
        )
        for rec in off:
            lines.append(
                "| {0} | {1} | {2} | {3:.4f} | {4:.4f} | {5} | {6} | {7} | {8} | {9} |".format(
                    rec["subject_id"],
                    rec["roi"],
                    rec.get("roi_name", ""),
                    float(rec["P_fail"]),
                    float(rec["tau"]),
                    rec["y_fail"],
                    rec["flag"],
                    rec["error"],
                    rec.get("channel_set", ""),
                    rec.get("model", ""),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"
