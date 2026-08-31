"""Command-line entry points: qc-prep, qc-prep-curvature, qc-train, qc-score, qc-report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from shapeqc.io import DEFAULT_ROIS, inventory_row, iter_subjects
from shapeqc.metrics import paper_counts


def _parse_features(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _parse_rois(text: str | None) -> list[int]:
    if not text:
        return list(DEFAULT_ROIS)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def prep(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="qc-prep", description="Build a subject×ROI inventory from a Shape tree.")
    p.add_argument("--shape-root", required=True)
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--rois", default="")
    p.add_argument("--labels", default="", help="Optional labels CSV with subject_id,roi,y_fail")
    args = p.parse_args(argv)
    rois = _parse_rois(args.rois)
    rows = []
    for subj in iter_subjects(args.shape_root):
        for roi in rois:
            rows.append(inventory_row(subj, roi))
    df = pd.DataFrame(rows)
    if args.labels:
        lab = pd.read_csv(args.labels)
        df = df.merge(lab, on=["subject_id", "roi"], how="left")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("wrote {0} rows to {1}".format(len(df), out))


def prep_curvature(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="qc-prep-curvature",
        description="Write curvature_<roi>.raw from resliced_mesh_<roi>.m (requires ccbbm).",
    )
    p.add_argument("--shape-root", required=True)
    p.add_argument("--rois", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out", default="", help="Optional status CSV")
    args = p.parse_args(argv)
    from shapeqc.curvature import generate_tree

    rows = generate_tree(args.shape_root, rois=_parse_rois(args.rois) or None, overwrite=args.overwrite)
    df = pd.DataFrame(rows)
    print(df["status"].value_counts().to_string() if len(df) else "no subjects")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)


def train(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="qc-train", description="Fit a simple scaler+logistic (CI / tiny example).")
    p.add_argument("--manifest", required=True, help="CSV from qc-prep with subject_dir, roi, y_fail")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--features", default="thick,logjac,curv")
    p.add_argument("--require-features", default="thick,logjac,curv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    from shapeqc.train import fit_simple, save_model, score_rows

    man = pd.read_csv(args.manifest)
    if "y_fail" not in man.columns:
        raise SystemExit("manifest needs y_fail (1=fail)")
    features = _parse_features(args.features)
    require = _parse_features(args.require_features)
    pipe, tau = fit_simple(man, features=features, require=require, seed=args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model_simple.joblib"
    save_model(pipe, tau, model_path)
    preds = score_rows(pipe, man, features, require, tau)
    preds.to_csv(out / "predictions.csv", index=False)
    review = preds.loc[preds["flag"] == 1].copy()
    review.to_csv(out / "review_list.csv", index=False)
    paper = preds.attrs.get("paper", {})
    (out / "paper_metrics.json").write_text(json.dumps(paper, indent=2) + "\n")
    print("tau={0:.4f} F_recall={1:.3f} F_share={2:.3f}".format(
        tau, paper.get("F_recall", float("nan")), paper.get("F_share", float("nan"))
    ))
    print("wrote {0}".format(out))


def score(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="qc-score", description="Inventory a Shape tree; score if --model is given.")
    p.add_argument("--shape-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rois", default="")
    p.add_argument("--features", default="thick,logjac,curv")
    p.add_argument("--require-features", default="thick,logjac,curv")
    p.add_argument("--model", default="", help="joblib from qc-train (optional)")
    args = p.parse_args(argv)
    rois = _parse_rois(args.rois)
    rows = [inventory_row(s, r) for s in iter_subjects(args.shape_root) for r in rois]
    inv = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out / "inventory.csv", index=False)
    if not args.model:
        missing = (~inv["curv"]).sum()
        print("wrote inventory ({0} rows, {1} missing curvature)".format(len(inv), int(missing)))
        print("pass --model to write P_fail / flag")
        return
    import joblib
    from shapeqc.io import load_channels

    blob = joblib.load(args.model)
    pipe, tau = blob["pipeline"], float(blob["tau"])
    features = _parse_features(args.features)
    require = _parse_features(args.require_features)
    scored = []
    for _, row in inv.iterrows():
        try:
            x = load_channels(row["subject_dir"], int(row["roi"]), features=features, require=require)
        except FileNotFoundError:
            scored.append({**row, "P_fail": None, "tau": tau, "flag": None, "error": "missing_feature"})
            continue
        pfail = float(pipe.predict_proba(x.reshape(1, -1))[0, 1])
        scored.append({**row, "P_fail": pfail, "tau": tau, "flag": int(pfail >= tau), "error": ""})
    pred = pd.DataFrame(scored)
    pred.to_csv(out / "predictions.csv", index=False)
    pred.loc[pred["flag"] == 1].to_csv(out / "review_list.csv", index=False)
    print("wrote {0}".format(out))


def report(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="qc-report", description="Paper metrics from a predictions CSV.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    df = pd.read_csv(args.predictions)
    if "y_fail" not in df.columns:
        raise SystemExit("predictions CSV needs y_fail to compute paper metrics")
    tau = float(df["tau"].iloc[0]) if "tau" in df.columns else 0.5
    m = paper_counts(df["y_fail"].to_numpy(), df["P_fail"].to_numpy(), threshold=tau)
    text = json.dumps(m, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


