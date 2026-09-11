# Folder map

Public command: **`qc-report` only**. Training and prep stay as Python functions, not shell commands.

| Path | What it is |
|------|------------|
| `README.md` | Start here |
| `local_paths.example.py` | Template; copy to gitignored `local_paths.py` |
| `src/shapeqc/` | Installable library |
| `configs/` | Paper validation **tau** and YAML (cluster training recipes) |
| `paper/` | Frozen Table 1 numbers (no models) |
| `examples/tiny/` | 32-vertex fake tree for CI (no labels) |
| `tests/` | pytest |
| `docs/` | Extra notes |
| `extras/` | Pointers to cluster research that is **not** this CLI |

## `src/shapeqc/` (code)

| File | Role |
|------|------|
| `cli.py` | `qc-report` entry. `prep` / `train` / `score` functions exist but are **not** installed as commands |
| `settings.py` | Load gitignored `local_paths.py` |
| `score.py` | Main path: frozen KPCA, training tau, write md + csv |
| `train.py` | **Internal** simple logistic (scaler + logistic, curvature required). Not the paper KPCA. No `imbalanced-learn` |
| `report.py` | Held-out toy eval (lab only; not on `qc-report`) |
| `paper_meta.py` | Frozen run ids + paper_deployable / n_fail from `paper/` |
| `models_frozen.py` | Load KPCA joblibs (env / local_paths); SHA256 check; no cluster paths |
| `labels.py` | Lab-only tidy/wide QA reader (not used by `qc-report`) |
| `curvature.py` | `ccbbm` wrapper |
| `io.py` | `thick_*.raw`, `LogJacs_*.raw`, `curvature_*.raw` |
| `metrics.py` | F-recall, F-share, recall-floor tau |
| `split.py` | Subject-level split (toy eval only) |

Paper KPCA **training** lives outside this tree: `QC_ML/scripts/logistic/train_logistic_enigma.py` (cluster-only).
