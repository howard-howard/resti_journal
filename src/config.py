"""Centralized Configuration and Path Definitions for RESTI Journal Project.

Defines spatial boundaries, database paths, feature vectors, and monsoon
season categorizations across WPPNRI 711 (Natuna Sea / Karimata Strait).
"""

from pathlib import Path

# ==============================================================================
# DIRECTORY STRUCTURE
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_OUTPUT_DIR = DATA_DIR / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure runtime directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Key file paths
DB_PATH = DATA_RAW_DIR / "wpp711_lakehouse.duckdb"
ZONES_CSV_PATH = DATA_OUTPUT_DIR / "hdbscan_spatio_behavioral_zones.csv"

# ==============================================================================
# GEOGRAPHIC BOUNDARIES (WPPNRI 711: Natuna Sea & Karimata Strait)
# ==============================================================================
MIN_LAT = -3.226
MAX_LAT = 4.800
MIN_LON = 102.681
MAX_LON = 110.304

# ==============================================================================
# MONSOON SEASONS (Climatological Calendar in Western Indonesia)
# ==============================================================================
MONSOON_SEASONS = {
    "West_Monsoon": [12, 1, 2],    # Northwest Monsoon (Boreal Winter)
    "Transition_1": [3, 4, 5],     # First Inter-monsoon (Spring)
    "East_Monsoon": [6, 7, 8],     # Southeast Monsoon (Austral Winter)
    "Transition_2": [9, 10, 11]    # Second Inter-monsoon (Autumn)
}

# ==============================================================================
# HDBSCAN HYPERPARAMETERS & 6D FEATURE SPACE
# ==============================================================================
HDBSCAN_PARAMS = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "metric": "euclidean",
    "gen_min_span_tree": True
}

FEATURE_COLUMNS = [
    "centroid_lat",
    "centroid_lon",
    "fishing_hours_IDN_Fleet",
    "fishing_hours_Non_IDN_Fleet",
    "loitering_ratio_IDN_Fleet",
    "loitering_ratio_Non_IDN_Fleet"
]

# Benchmark configuration
BENCHMARK_TRIALS = 30

