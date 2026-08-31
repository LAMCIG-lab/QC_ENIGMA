# Contributing

This tree is the public software drop for LAMCIG (`enigma-shape-qc` on GitHub).

- Do not commit ENIGMA QA CSVs, ADNI paths, `.joblib` models, or FreeSurfer trees.
- Default features are `thick,logjac,curv`. Missing curvature must error, not zero-fill.
- Paper Table 1 uses `configs/paper_l3_thick_logjac_curv.yaml` after a retrain that **requires** curvature files.
- Keep `examples/tiny/` synthetic and tiny so CI can run without `ccbbm`.
