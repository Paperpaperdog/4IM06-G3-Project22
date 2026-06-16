from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RAISE_RAW_DIR = DATA_DIR / "raise_raw"
RAISE_PNG_DIR = DATA_DIR / "raise_png"
GENERATED_DIR = DATA_DIR / "generated"
PILOT_RESULTS_DIR = DATA_DIR / "pilot_results"

MANIFEST_CSV = DATA_DIR / "manifest.csv"

TARGET_SIZE = 384
REFERENCE_SIZES = (64, 85, 96)
K_VALUES = (-1, 0, 1)
K_OFFSETS = (0, 1, 2)

INTERP_ORDER = 3
TV_WEIGHT = 1.0
DETECTION_RADIUS = 10
LOG10_NFA_THRESHOLD = -5.0

# Distances tied to JPEG block grid (for width=384: n/8=48, n/16=24, …)
JPEG_PROBE_DIVISORS = (8, 16)

# Extra distances recorded in Idea 1 reports
IDEA1_PROBE_DISTANCES = (8, 16, 24, 48, 64, 96, 128)
