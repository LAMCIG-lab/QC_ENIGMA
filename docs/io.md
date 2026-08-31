# Inputs and outputs

## Per-subject Shape directory (input)

```
<subject>/
  resliced_mesh_17.m
  thick_17.raw
  LogJacs_17.raw
  curvature_17.raw      # generated; required for default model
  curve_17.ucf          # unused
```

Vertex `i` is the same atlas location in all three `.raw` files.

## Train outputs (`qc-train`)

| File | Public? | Contents |
|------|---------|----------|
| `models/*.joblib` | internal | Fitted pipeline per ROI × seed |
| `results_long.csv` | internal | Per job metrics |
| `results_agg.csv` | paper | Seed-mean metrics |
| `deployment.csv` | paper | Weak-ROI verdict |
| `paper_metrics.json` | paper | Macro F-recall, F-share, AUC |
| `preds_*.csv` | restricted | Subject-level scores |

## Score outputs (`qc-score`)

| File | Public? | Contents |
|------|---------|----------|
| `review_list.csv` | restricted | Subject, ROI, P_fail, flag |
| `summary_by_roi.csv` | paper | Flagged % only, no IDs |

`flag` means `P_fail >= tau` with `tau` from validation at recall floor 0.8.
That is a review queue, not an automatic ground-truth label.
