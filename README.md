# enigma-shape-qc

Checking strategy for ENIGMA subcortical shape files (which subject × ROI to look at). **Not** an automatic fail detector and **not** a medical device.

```bash
cp local_paths.example.py local_paths.py   # then edit SHAPE_ROOT, OUT_DIR, MODEL_3CH, MODEL_2CH
chmod 600 local_paths.py
qc-report
```

`local_paths.py` is gitignored. Do not put cluster paths in committed source.

Default model: **thick + logjac + curv**. If `curvature_<roi>.raw` is missing but `resliced_mesh_<roi>.m` exists, curvature is generated (`CCBBM` must be an absolute path). If both are missing, that row uses the frozen 2-channel model. Tau is the training validation operating point unless you set `TAU` (example: `TAU = {11: 0.6}`).

Human QA CSVs are not read.

## What `qc-report` loads

The 3-channel model is **`curv_full_recall80`** (PBS 24264). Two sites (FBIRN and UCIS, 416 subjects) had no meshes; curvature was zero-filled **in that training run**. Those meshes are not reconstructed. That is the intended technician reference.

| Channel set | Run id | When it is used |
|-------------|--------|-----------------|
| thick+logjac+curv | `curv_full_recall80` (PBS 24264) | Default, when `curvature_*.raw` exists (or was generated from mesh) |
| thick+logjac | `24239` (retune `24239_complete_retune_r08`) | New folders with **no mesh and no curvature** (2-channel; not zero-fill) |

`paper/` numbers are from these same runs. `configs/joblib_sha256.csv` lists SHA256 of the seed-42 joblibs. `qc-report` checks that hash before `joblib.load` (pickle). Weights stay **out of git**; GitHub CI does **not** reproduce frozen `P_fail` (it only fits 32-vertex dummy logistics on `examples/tiny`).

Only ROI **13, 17, 53** met paper deployability floors. Other ROIs are still scored (`paper_deployable=no` in the report). Treat those flags as a weaker reference — e.g. ROI 11 had 9 fail labels in ~5409 rows.

## Outputs

| File | Meaning |
|------|---------|
| `qc_report.md` | Checking list of flagged **files**, with deployability banners |
| `predictions.csv` | Every scored row (`P_fail`, `tau`, `flag`, `paper_deployable`, `files`). No absolute `subject_dir`. `flag==1` is who to look at |

## Folder map

| Path | What |
|------|------|
| `src/shapeqc/` | Library. Only `qc-report` is installed as a command |
| `local_paths.example.py` | Template for the gitignored `local_paths.py` |
| `configs/` | Paper taus, joblib SHA256, training YAML |
| `paper/` | Frozen Table 1 numbers |
| `examples/tiny/` | Fake 32-vertex tree for CI (no labels) |
| `tests/` | pytest |
| `docs/` | Extra notes |
| `extras/` | Not this CLI |

Training code (`src/shapeqc/train.py`) stays in the package for us. It is **not** a `qc-train` command. Paper KPCA training is cluster-only (`QC_ML/scripts/logistic/`).

## Install

```bash
cd /path/to/QC_ENIGMA
pip install -e ".[dev]"
export CCBBM=/absolute/path/to/ccbbm
```

Set `MODEL_3CH` / `MODEL_2CH` in `local_paths.py`. Those directories must hold the trusted paper `.joblib` files. Do not commit them.

## Allowed inputs

Either a site of subject folders or one subject folder of `thick_*.raw` + `LogJacs_*.raw`. Directories without `thick_*.raw` are **rejected**.
