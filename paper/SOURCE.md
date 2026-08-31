# Paper source run

Run id: `curv_full_recall80` (PBS 24264). Features `thick,logjac,curv`.
**Require was `thick,logjac` only** (zero-fill on 416 FBIRN/UCIS subjects).
Threshold `recall_floor` @ 0.8. Seeds 42–46. 105/105 jobs ok.

Camera-ready Table 1: retrain with `configs/paper_l3_thick_logjac_curv.yaml`
(`require_features: [thick, logjac, curv]`).

Ablation: run id `24239_complete_retune_r08` (thick+logjac, same L3 policy).

Lab copies of models and manifests stay on the cluster, not in this git tree.
This folder only has `deployment.csv`, `results_agg.csv`, `metrics_macro.json`.
