# Technician path

1. Finish ENIGMA-Shape so each subject folder has `resliced_mesh_*.m`, `thick_*.raw`, `LogJacs_*.raw`.
2. Generate curvature (`ccbbm` from ENIGMA-Shape `MedialDemonsShared/bin`):

   ```bash
   export CCBBM=/path/to/MedialDemonsShared/bin/ccbbm
   qc-prep-curvature --shape-root /path/to/Shape_output
   ```

3. Inventory, then score if you have a model file:

   ```bash
   qc-score --shape-root /path/to/Shape_output --out-dir ./outputs/score
   qc-score --shape-root /path/to/Shape_output --out-dir ./outputs/score --model /path/to/model.joblib
   ```

4. Give raters `review_list.csv` (`flag=1`). Weak ROIs (thalamus, amygdala, putamen) flag a large share of scans at the L3 recall floor.

If a site has no meshes, use `--features thick,logjac` or re-run Shape. Do not zero-fill curvature.
