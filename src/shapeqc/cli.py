"""Public command is qc-report only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapeqc.io import DEFAULT_ROIS
from shapeqc.settings import load_local_settings, score_options_from_local

PKG_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PKG_ROOT / "examples" / "tiny"
DUMMY_3CH = PKG_ROOT / "examples" / "dummy_models" / "3ch"
DUMMY_2CH = PKG_ROOT / "examples" / "dummy_models" / "2ch"
DUMMY_THRESHOLDS = PKG_ROOT / "examples" / "dummy_models" / "thresholds.csv"


def _parse_pairs(text: str | None):
    if not text:
        return None
    t = text.strip().lower()
    if t in ("1", "true", "yes", "all"):
        return True
    out = []
    for token in text.split(","):
        tok = token.strip().replace("-", "_")
        if not tok:
            continue
        if "_" not in tok:
            raise SystemExit("pair token must look like 17_53, got {0!r}".format(token))
        a, b = tok.split("_", 1)
        out.append((int(a), int(b)))
    return out or None


def _parse_rois(text: str | None) -> list[int]:
    if not text:
        return list(DEFAULT_ROIS)
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def report(argv: list[str] | None = None) -> None:
    local = load_local_settings()
    opts = score_options_from_local(local)
    p = argparse.ArgumentParser(
        prog="qc-report",
        description=(
            "Score Shape .raw files. Writes qc_report.md and predictions.csv. "
            "Defaults come from local_paths.py."
        ),
    )
    p.add_argument(
        "--shape-root",
        default="",
        help="Site folder, ENIGMA_SZ root, or one subject folder. Default: SHAPE_ROOT",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Where to write qc_report.md and predictions.csv. Default: OUT_DIR",
    )
    p.add_argument(
        "--rois",
        default="",
        help="Comma-separated ROI ids, e.g. 17,53. Default: ROIS or all 14",
    )
    p.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Cap subject folders under SHAPE_ROOT. 0 = every subject. Default: MAX_SUBJECTS",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Frozen joblib seed (paper weights are seed 42)",
    )
    p.add_argument(
        "--models-3ch",
        default="",
        help="Folder of thick+logjac+curv joblibs. Default: MODEL_3CH",
    )
    p.add_argument(
        "--models-2ch",
        default="",
        help="Folder of thick+logjac joblibs. Default: MODEL_2CH",
    )
    p.add_argument(
        "--thresholds",
        default="",
        help="CSV of frozen taus. Default: configs/thresholds.csv",
    )
    p.add_argument(
        "--interpolator",
        default="",
        help="CSV of recall->tau curves. Default: INTERPOLATOR, or configs/interpolator.csv if DESIRED_RECALL is set",
    )
    p.add_argument(
        "--desired-recall",
        type=float,
        default=None,
        help="Fail-recall target; selects tau from the interpolator. Default: DESIRED_RECALL",
    )
    p.add_argument(
        "--pairs",
        default="",
        help="L/R pair models: all, or 17_53,13_52. Default: PAIRS (unset = singles only)",
    )
    p.add_argument(
        "--skip-joblib-verify",
        action="store_true",
        help="Skip SHA256 check of joblibs",
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Inventory the tree (subjects, channels, sklearn, joblib hashes). Do not score.",
    )
    p.add_argument(
        "--example",
        action="store_true",
        help="Score examples/tiny with in-repo 32-vertex dummy models (not paper weights).",
    )
    p.add_argument(
        "--ccbbm",
        default="",
        help="Absolute path to ccbbm. Default: CCBBM in local_paths.py or env CCBBM",
    )
    p.add_argument(
        "--no-write-curvature",
        action="store_true",
        help="Do not generate curvature_*.raw from meshes; missing curv uses 2-channel",
    )
    args = p.parse_args(argv)

    dummy = bool(args.example)
    if dummy:
        if not EXAMPLE_ROOT.is_dir():
            raise SystemExit("missing examples/tiny")
        shape_root = str(EXAMPLE_ROOT)
        out_dir = args.out_dir or local.out_dir or "outputs/qc_tiny"
        opts["models_3ch"] = str(DUMMY_3CH)
        opts["models_2ch"] = str(DUMMY_2CH)
        opts["rois"] = [13, 17]
        opts["write_curvature"] = False
        verify = False
        thresholds = str(DUMMY_THRESHOLDS)
    else:
        shape_root = args.shape_root or local.shape_root
        out_dir = args.out_dir or local.out_dir
        verify = not args.skip_joblib_verify
        thresholds = args.thresholds or None
        if not shape_root or shape_root == "/path/to/Shape_output":
            raise SystemExit(
                "qc-report: copy local_paths.example.py to local_paths.py and set SHAPE_ROOT "
                "(and OUT_DIR), or run qc-report --example"
            )

    if not args.doctor and not out_dir:
        raise SystemExit("qc-report: set OUT_DIR in local_paths.py")

    if args.rois:
        opts["rois"] = _parse_rois(args.rois)
    if args.max_subjects:
        opts["max_subjects"] = args.max_subjects
    if args.models_3ch:
        opts["models_3ch"] = args.models_3ch
    if args.models_2ch:
        opts["models_2ch"] = args.models_2ch
    if args.interpolator:
        opts["interpolators"] = args.interpolator
    if args.desired_recall is not None:
        opts["desired_recall"] = args.desired_recall
    if args.pairs:
        opts["pairs"] = _parse_pairs(args.pairs)
    if args.ccbbm:
        opts["ccbbm"] = args.ccbbm
    if args.no_write_curvature:
        opts["write_curvature"] = False

    if args.doctor:
        from shapeqc.doctor import print_doctor, run_doctor

        payload = run_doctor(
            shape_root,
            rois=opts.get("rois"),
            max_subjects=opts.get("max_subjects"),
            models_3ch=opts.get("models_3ch"),
            models_2ch=opts.get("models_2ch"),
            ccbbm=opts.get("ccbbm"),
            verify_checksums=verify,
            dummy=dummy,
        )
        print_doctor(payload)
        if not payload.get("ok"):
            raise SystemExit(1)
        return

    from shapeqc.score import run_score

    payload = run_score(
        shape_root,
        out_dir=out_dir,
        seed=args.seed,
        thresholds=thresholds,
        verify_checksums=verify,
        **opts,
    )
    m = payload["macro"]
    out_json = {
        "mode": "score",
        "dummy_track": dummy,
        "n_subjects": m.get("n_subjects", 0),
        "n_rows": m["n_rows"],
        "n_flagged": m["n_flagged"],
        "flagged_frac": round(m["flagged_frac"], 3) if m["n_rows"] else 0.0,
        "n_3ch": m["n_3ch"],
        "n_2ch": m["n_2ch"],
        "n_curv_wrote": m.get("n_curv_wrote", 0),
        "out_dir": str(out_dir),
    }
    print(json.dumps(out_json, indent=2))
    print("wrote {0}/qc_report.md and predictions.csv".format(out_dir))
    if dummy:
        print("dummy 32-vertex track: not paper KPCA weights")
    if m["n_flagged"]:
        print("{0} flagged rows - see the Review list for the .raw files to open".format(m["n_flagged"]))
