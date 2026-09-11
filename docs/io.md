# Inputs and outputs

## Per-subject Shape directory (input)

`qc-report` accepts a site folder of subject directories, or one subject folder.

```
<subject>/
  resliced_mesh_17.m
  thick_17.raw
  LogJacs_17.raw
  curvature_17.raw      # generated from mesh when missing; optional (2ch fallback)
  curve_17.ucf          # unused
```

Vertex `i` is the same atlas location in all three `.raw` files.

If curvature and mesh are both missing, `qc-report` scores that row with the frozen 24239 thick+logjac model instead of dropping it.

Unknown sibling directories (no `thick_*.raw`) abort the run.

## `qc-report` outputs

| File | Contents |
|------|----------|
| `qc_report.md` | Checking-strategy banners, per-ROI table (includes `paper_deployable`), review list of flagged files |
| `predictions.csv` | subject_id, ROI, P_fail, tau, flag, paper_deployable, n_fail_paper, files (no absolute subject_dir) |

`flag` means `P_fail >= tau`. That is a list of files to check, not a ground-truth fail. Raise `TAU` in `local_paths.py` to shrink a scarce-ROI queue.
