# Tiny synthetic Shape tree (ROI 13 and 17, 32 vertices)

Not real ENIGMA or ADNI data. Used for install checks and CI. No QA labels.

```
examples/tiny/
  subjA/thick_17.raw
  subjA/LogJacs_17.raw
  subjA/curvature_17.raw
  subjA/... same for ROI 13
  subjB/...
```

No meshes: missing curvature would fall back to 2-channel. These folders already have `.raw` files.
