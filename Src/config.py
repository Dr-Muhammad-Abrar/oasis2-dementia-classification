# =============================================================
# config.py  —  Central configuration for all paths and settings
# =============================================================

import os

# ── Dataset Paths ─────────────────────────────────────────────
BASE_DIR  = r"C:\Users\Muhammad Abrar\PycharmProjects\AlzhiemerDiseaseProgression\Datasets for ODe"
PART1_DIR = os.path.join(BASE_DIR, "OAS2_RAW_PART1")
PART2_DIR = os.path.join(BASE_DIR, "OAS2_RAW_PART2")
CSV_PATH  = os.path.join(BASE_DIR, "oasis_longitudinal_demographics.xlsx")

# ── Output Paths ──────────────────────────────────────────────
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
SLICES_DIR      = os.path.join(OUTPUT_DIR, "mri_slices")      # saved 2D PNG slices
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")           # saved model weights
RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")          # plots, metrics, XAI
ODE_DIR         = os.path.join(OUTPUT_DIR, "ode_results")      # ODE fitting outputs

# ── MRI Settings ──────────────────────────────────────────────
IMG_SIZE        = 128          # resize each 2D slice to 128×128
SLICE_AXIS      = 1            # axis to slice along (1 = coronal, good for hippocampus)
SLICE_FRACTION  = 0.45         # take slice at 45% along the axis (hippocampal region)
MRI_FILENAME    = "mpr-1.nifti.hdr"   # always use mpr-1 as representative scan

# ── Deep Learning Settings ────────────────────────────────────
BATCH_SIZE      = 8            # small batch for CPU
NUM_EPOCHS      = 15           # realistic for CPU training
LEARNING_RATE   = 0.0001
NUM_CLASSES     = 3            # Nondemented, Converted, Demented
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# ── ODE Settings ──────────────────────────────────────────────
ODE_STATE_VAR   = "CDR"        # primary state variable for ODE
ODE_TIME_VAR    = "MR Delay"   # time axis in days
MIN_VISITS      = 2            # minimum visits per patient for ODE fitting

# ── Label Mapping ─────────────────────────────────────────────
LABEL_MAP = {
    "Nondemented": 0,
    "Converted":   1,
    "Demented":    2
}
LABEL_NAMES = ["Nondemented", "Converted", "Demented"]
