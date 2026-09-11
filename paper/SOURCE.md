# Paper source run

`qc-report` loads these frozen joblibs (seed 42; SHA256 in `configs/joblib_sha256.csv`):

- 3-channel: `curv_full_recall80` (PBS 24264). Features `thick,logjac,curv`.
  Two sites (**FBIRN**, **UCIS**; 416 subjects) had no meshes, so curvature was
  **zero-filled in training**. Reconstructing those meshes is not in scope.
  This is the intended checking-tool weight.
- 2-channel fallback on **new** data with no mesh and no curvature: `24239`
  (retune `24239_complete_retune_r08`). That path does not zero-fill.

Threshold `recall_floor` @ 0.8. Seeds 42–46. 105/105 jobs ok.

`configs/paper_l3_thick_logjac_curv.yaml` is a lab recipe that would require
curvature for every training row. It is **not** what `qc-report` ships.

Lab copies of models and manifests stay on the cluster, not in this git tree.
This folder only has `deployment.csv`, `results_agg.csv`, `metrics_macro.json`.
