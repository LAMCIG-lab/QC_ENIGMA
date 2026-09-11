# Contributing

This tree is the public software drop for LAMCIG (`QC_ENIGMA` on GitHub).

- Do not commit ENIGMA QA CSVs, lab filesystem paths, `.joblib` models, `.raw` imaging (except `examples/tiny/`), or FreeSurfer trees.
- Put Shape root / output / model dirs in gitignored `local_paths.py` (copy `local_paths.example.py`, then `chmod 600`).
- Public CLI is `qc-report` only. It is a checking strategy, not a fail detector.
- GitHub CI fits dummy 32-vertex logistics. It does **not** reproduce frozen `P_fail`.
- Default features are `thick,logjac,curv`. Shipped 3-channel weights are `curv_full_recall80` (FBIRN/UCIS no-mesh rows were zero-filled in training; those meshes are not reconstructed). New data with no mesh and no curvature uses the 2-channel `24239` model, not zero-fill.
- Keep `examples/tiny/` synthetic and tiny so CI can run without `ccbbm`. No labels CSV in that tree.
