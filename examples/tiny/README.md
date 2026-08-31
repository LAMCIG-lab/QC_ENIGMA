# Tiny synthetic Shape tree (ROI 17 only, 32 vertices)

Not real ENIGMA or ADNI data. Used for install checks and CI.

```
examples/tiny/
  labels.csv                 # subjA pass (0), subjB fail (1)
  subjA/thick_17.raw
  subjA/LogJacs_17.raw
  subjA/curvature_17.raw
  subjB/...
```

No meshes: `qc-prep-curvature` will report `no_mesh`. Scoring uses the three `.raw` files already here.
