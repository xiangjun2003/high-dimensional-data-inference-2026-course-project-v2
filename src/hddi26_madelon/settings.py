"""Project paths and runtime constants."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"
TMP_DIR = OUTPUT_DIR / "tmp"
CACHE_DIR = PROJECT_ROOT / ".cache"

RANDOM_STATE = 26
CLASS_NAMES = {-1: "negative", 1: "positive"}
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/171/madelon.zip"
UCI_FILES = {
    "train_data": "madelon_train.data",
    "train_labels": "madelon_train.labels",
    "valid_data": "madelon_valid.data",
    "valid_labels": "madelon_valid.labels",
}


def ensure_runtime_dirs() -> None:
    """Create runtime directories and configure local caches."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        FIGURE_DIR,
        RESULT_DIR,
        TMP_DIR,
        CACHE_DIR / "matplotlib",
        CACHE_DIR / "numba",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
