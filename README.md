# Shape QC scoring

Checking list for ENIGMA subcortical shape files: which subject x ROI to look at.
**Not** an automatic fail detector, **not** a medical device.

Default channels are **thickness + log-Jacobian + curvature**. Frozen joblibs are
sklearn **1.3.2** pickles (a newer sklearn can fail or silently mis-score).

## 30-minute path

```bash
cp local_paths.example.py local_paths.py
# set SHAPE_ROOT, OUT_DIR, MODEL_3CH, MODEL_2CH, CCBBM
chmod 600 local_paths.py
pip install -e ".[dev]" --only-binary=:all:
# if pandas/sklearn try to compile:
# pip install numpy pandas 'scikit-learn==1.3.2' joblib --only-binary=:all: && pip install -e . --no-deps
qc-report --doctor
qc-report
```

Laptop check **without paper weights** (32-vertex dummy logistics, not KPCA):

```bash
qc-report --example --doctor
qc-report --example
```

Plots: `notebooks/demo_predict_qc.ipynb` (same `local_paths.py` knobs; does not write `OUT_DIR`).
Install `pip install 'matplotlib<3.8' --only-binary=:all:` for figures.

`qc-report --help` lists every flag.

## Get the weights

Paper models are **not** in git (joblib is pickle). Copy the trusted seed-42 folders:

| Knob | Run | When used |
|------|-----|-----------|
| `MODEL_3CH` | `curv_full_recall80` training dir | `curvature_<roi>.raw` present (or written from mesh) |
| `MODEL_2CH` | `24239` training dir | no curvature and no mesh |

Files look like `*_roi17_seed42_kpca_logistic.joblib`. `qc-report` SHA256-checks
them against `configs/joblib_sha256.csv`. Train/load with **scikit-learn 1.3.2**.
Do not commit the joblibs. `SHAPEQC_MODEL_3CH` / `SHAPEQC_MODEL_2CH` also work.

## What SHAPE_ROOT can be

The tool **searches** for usable files (`thick_*.raw` + `LogJacs_<roi>.raw`).

```
SHAPE_ROOT you set
  one subject folder          -> score that folder
  ENIGMA_SZ                   -> 18 labeled sites in configs/site_layout.csv
  one site (TOP, Oxford, ...) -> nested shape folders under that site
```

Expected names inside a subject folder:

```
thick_17.raw
LogJacs_17.raw
curvature_17.raw          # optional; default 3-channel when present
resliced_mesh_17.m        # optional; used to write curvature if missing
```

ROI ids are FreeSurfer aseg labels:

| ROI | Name | Vertices | 3ch dim |
|-----|------|----------|---------|
| 10 / 49 | Thalamus L/R | 2502 | 7506 |
| 11 / 50 | Caudate L/R | 2502 | 7506 |
| 12 / 51 | Putamen L/R | 2502 | 7506 |
| 13 / 52 | Pallidum L/R | 1254 | 3762 |
| 17 / 53 | Hippocampus L/R | 2502 | 7506 |
| 18 / 54 | Amygdala L/R | 1368 | 4104 |
| 26 / 58 | Accumbens L/R | 930 | 2790 |

32-vertex `examples/tiny` is **only** for `--example`. Frozen KPCA expects the atlas
lengths above (`n_features_in_` must match).

Duplicate subject folders (for example `TOP_SZ/<id>` and `TOP_SZ/bilat_shape/<id>`)
keep the **shortest** path. `stats_*`, QA, and FreeSurfer dirs are skipped.
Incomplete thick/LogJacs pairs are skipped, not a site-wide abort.

The 18 labeled sites (when root is `ENIGMA_SZ`): Basel_SCORE, Brazil (NARSAD),
CAMH, Dublin, FBIRN, FIDMAG, Galway, Indiana, Kochunov, Olin, Oxford,
PAFIP_GE_SZ, PAFIP_Phillips_SZ, R_SCZ, StaLucia, TOP, UCIS, Upenn.

## Curvature default

1. `curvature_<roi>.raw` exists -> 3-channel model.
2. Else `resliced_mesh_<roi>.m` exists and `WRITE_CURVATURE` is true (default) and
   `CCBBM` is an absolute path -> write `curvature_<roi>.raw` **next to the mesh**,
   then 3-channel.
3. Else 2-channel (`24239`). The site is not failed.

## Outputs

`P_fail` is `predict_proba` (class 1). `flag` is `P_fail >= tau`. Tau is per ROI
and **not comparable across ROIs**. `flag=1` is a review queue.

Only ROIs **13, 17, 53** (and pair **13_52**) met paper deployability floors.
Other ROI flags are a weaker reference (scarce fail labels).

| File | Meaning |
|------|---------|
| `qc_report.md` | Review list |
| `predictions.csv` | Every scored row (`P_fail`, `tau`, `flag`, `paper_deployable`, files) |

## Settings (`local_paths.py` / flags)

| Knob | Flag | Default |
|------|------|---------|
| `SHAPE_ROOT` | `--shape-root` | required (or `--example`) |
| `OUT_DIR` | `--out-dir` | `qc-report` writes here |
| `ROIS` | `--rois` | all 14 |
| `MAX_SUBJECTS` | `--max-subjects` | every subject |
| `TAU` | (file only) | `configs/thresholds.csv` |
| `DESIRED_RECALL` | `--desired-recall` | else thresholds.csv |
| `INTERPOLATOR` | `--interpolator` | `configs/interpolator.csv` if recall is set |
| `PAIRS` | `--pairs` | singles only |
| `WRITE_CURVATURE` | `--no-write-curvature` | generate from mesh when possible |
| `CCBBM` | `--ccbbm` | env `CCBBM` also works |
| `MODEL_3CH` | `--models-3ch` | `SHAPEQC_MODEL_3CH` |
| `MODEL_2CH` | `--models-2ch` | `SHAPEQC_MODEL_2CH` |

Also: `--doctor`, `--example`, `--thresholds`, `--seed` (paper is 42), `--skip-joblib-verify`.

Human QA CSVs are not read at score time.
