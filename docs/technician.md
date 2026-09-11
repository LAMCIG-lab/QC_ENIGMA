# Technician path

```bash
cp local_paths.example.py local_paths.py
# edit SHAPE_ROOT, OUT_DIR, MODEL_3CH, MODEL_2CH
chmod 600 local_paths.py
qc-report
```

Writes `qc_report.md` and `predictions.csv`. `flag==1` is a **checking** list of files to open, not a fail label.

QA CSVs are ignored. Scarce-label ROIs (not `paper_deployable`) are weaker references — raise `TAU` (e.g. `{11: 0.6}`) if caudate floods the queue.

`CCBBM` must be an absolute path. The 3-channel model is `curv_full_recall80` (FBIRN/UCIS no-mesh rows were zero-filled in training; those meshes are not reconstructed). New folders with no mesh and no curvature use 2-channel `24239`. GitHub CI does not load the frozen KPCA weights.
