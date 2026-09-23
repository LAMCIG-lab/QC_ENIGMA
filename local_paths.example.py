"""Copy this file to local_paths.py and edit. local_paths.py is gitignored.
Then: chmod 600 local_paths.py && qc-report --doctor && qc-report

qc-report and notebooks/demo_predict_qc.ipynb read the same knobs.
Only qc-report writes OUT_DIR (qc_report.md + predictions.csv).
Laptop smoke test (no paper weights): qc-report --example
"""

# Site folder, ENIGMA_SZ root, or one subject folder of .raw files.
# The scorer searches for folders that contain thick_*.raw (nested ENIGMA
# layouts included). Duplicate names keep the shortest path.
SHAPE_ROOT = "/path/to/Shape_output"

OUT_DIR = "./outputs/qc"

# None = all 14 ENIGMA ROIs. Or a list, e.g. [17, 53].
ROIS = None

# 0 or None = every subject under SHAPE_ROOT. Positive int = cap.
MAX_SUBJECTS = None

# None = taus in configs/thresholds.csv (recall-floor 0.8).
TAU = None

# Optional fail-recall -> interpolator tau. None = thresholds.csv.
DESIRED_RECALL = None
INTERPOLATOR = "configs/interpolator.csv"

# None = singles only. True = all L/R homolog pairs. Or [(17, 53), (13, 52)].
PAIRS = None

# If curvature_<roi>.raw is missing but resliced_mesh_<roi>.m exists, write
# curvature next to the mesh (needs CCBBM). True = generate. If ccbbm is
# missing, those rows use the 2-channel model instead of failing the site.
WRITE_CURVATURE = True
CCBBM = ""  # absolute path to MedialDemonsShared/bin/ccbbm

# Frozen KPCA folders (sklearn 1.3.2 pickles). SHA256-checked against
# configs/joblib_sha256.csv. Or set SHAPEQC_MODEL_3CH / SHAPEQC_MODEL_2CH.
MODEL_3CH = "/path/to/curv_full_recall80/training"
MODEL_2CH = "/path/to/24239/training"
