"""Public command is qc-report only (installed).

prep / train / score remain in this module for lab experiments but are
not console_scripts and are not called by qc-report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from shapeqc.io import DEFAULT_ROIS, inventory_row, iter_subjects
from shapeqc.settings import load_local_settings


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
    args = p.parse_args(argv)
    rois = _parse_rois(args.rois)
    rows = []
    for subj in iter_subjects(args.shape_root):
        for roi in rois:
            rec = inventory_row(subj, roi)
            if rec["thick"] and rec["logjac"]:
                rows.append(rec)
    df = pd.DataFrame(rows)
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
    p = argparse.ArgumentParser(
        prog="qc-train",
        description="Fit scaler+logistic on thick+logjac+curv (sklearn only; no imbalanced-learn).",
    )
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
    rows = []
    for s in iter_subjects(args.shape_root):
        for r in rois:
            rec = inventory_row(s, r)
            if rec["thick"] and rec["logjac"]:
                rows.append(rec)
    inv = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out / "inventory.csv", index=False)
    if not args.model:
        missing = (~inv["curv"]).sum() if len(inv) else 0
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
    print("wrote {0}".format(out))


def report(argv: list[str] | None = None) -> None:
    local = load_local_settings()
    p = argparse.ArgumentParser(
        prog="qc-report",
        description="Shape .raw files → qc_report.md + predictions.csv. Defaults come from local_paths.py.",
    )
    p.add_argument("--shape-root", default="", help=argparse.SUPPRESS)
    p.add_argument("--out-dir", default="", help=argparse.SUPPRESS)
    p.add_argument("--rois", default="", help=argparse.SUPPRESS)
    p.add_argument("--max-subjects", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    p.add_argument("--models-3ch", default="", help=argparse.SUPPRESS)
    p.add_argument("--models-2ch", default="", help=argparse.SUPPRESS)
    p.add_argument("--thresholds", default="", help=argparse.SUPPRESS)
    p.add_argument("--skip-joblib-verify", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    shape_root = args.shape_root or local.shape_root
    out_dir = args.out_dir or local.out_dir
    if not shape_root or shape_root == "/path/to/Shape_output":
        raise SystemExit(
            "qc-report: copy local_paths.example.py to local_paths.py and set SHAPE_ROOT "
            "(and OUT_DIR). Do not put cluster paths in committed files."
        )
    if not out_dir:
        raise SystemExit("qc-report: set OUT_DIR in local_paths.py")

    if args.rois:
        rois = _parse_rois(args.rois)
    elif local.rois:
        rois = [int(x) for x in local.rois]
    else:
        rois = None

    max_subjects = args.max_subjects or local.max_subjects or None
    models_3ch = args.models_3ch or local.model_3ch or None
    models_2ch = args.models_2ch or local.model_2ch or None

    from shapeqc.score import run_score

    payload = run_score(
        shape_root,
        rois=rois,
        out_dir=out_dir,
        seed=args.seed,
        models_3ch=models_3ch,
        models_2ch=models_2ch,
        thresholds=args.thresholds or None,
        tau_override=local.tau,
        max_subjects=max_subjects,
        verify_checksums=not args.skip_joblib_verify,
    )
    m = payload["macro"]
    out_json = {
        "mode": "score",
        "n_rows": m["n_rows"],
        "n_flagged": m["n_flagged"],
        "flagged_frac": round(m["flagged_frac"], 3) if m["n_rows"] else 0.0,
        "n_3ch": m["n_3ch"],
        "n_2ch": m["n_2ch"],
        "out_dir": str(out_dir),
    }
    print(json.dumps(out_json, indent=2))
    print("wrote {0}/qc_report.md and predictions.csv".format(out_dir))
    if m["n_flagged"]:
        print("{0} flagged rows — see the Review list for the .raw files to open".format(m["n_flagged"]))
