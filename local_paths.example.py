"""Copy this file to local_paths.py and edit. local_paths.py is gitignored.
Then: chmod 600 local_paths.py && qc-report
"""

# Site folder of subject subfolders, or one subject folder of .raw files.
SHAPE_ROOT = "/path/to/Shape_output"

# Where qc_report.md and predictions.csv are written.
OUT_DIR = "./outputs/qc"

# None = all 14 ENIGMA ROIs. Or a list, e.g. [17, 53].
ROIS = None

# None = paper taus in configs/thresholds.csv (recall-floor 0.8).
# float = same tau for every ROI.
# dict = per-ROI override, e.g. {11: 0.6} (other ROIs keep the paper tau).
# Nested: {11: {"thick+logjac+curv": 0.6, "thick+logjac": 0.5}}
TAU = None

# Frozen KPCA folders (or set SHAPEQC_MODEL_3CH / SHAPEQC_MODEL_2CH in the environment).
# Trusted paper weights only — joblib is pickle. qc-report SHA256-checks seed-42
# files against configs/joblib_sha256.csv. Do not commit real cluster paths.
MODEL_3CH = ""
MODEL_2CH = ""
