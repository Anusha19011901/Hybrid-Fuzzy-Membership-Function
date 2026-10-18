"""Project-wide constants: paths, random seed, column groups, risk mapping.

Everything that a later phase might need to agree on lives here so that the
notebooks, scripts and tests cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "heart_disease_uci.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.2  # 920 rows -> 184 test rows, matching the patent's DL baseline
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Columns of the UCI Heart Disease dataset (Kaggle `heart_disease_uci.csv`)
# --------------------------------------------------------------------------- #
ID_COL = "id"
SITE_COL = "dataset"  # source hospital: Cleveland, Hungary, Switzerland, VA
RAW_TARGET = "num"    # 0 = <50% narrowing, 1-4 = >50% narrowing, increasing severity
TARGET = "risk"       # derived 3-level label (see RISK_MAPPINGS)

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak"]
DISCRETE_COLS = ["ca"]  # integer count 0-3; never outlier-clipped
NUMERIC_COLS = CONTINUOUS_COLS + DISCRETE_COLS
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Human-readable names used in plots and the explainability output.
ATTRIBUTE_LABELS = {
    "age": "Age (years)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "thalch": "Max heart rate achieved",
    "oldpeak": "Exercise ST depression",
    "ca": "Major vessels coloured (0-3)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "exang": "Exercise-induced angina",
    "slope": "ST slope",
    "thal": "Thalassemia test",
}

# Physiologically impossible zeros that the source files use as a missing marker.
# A resting blood pressure or a serum cholesterol of 0 cannot occur in a living
# patient; 172 rows (almost all from Switzerland) carry chol == 0.
ZERO_MEANS_MISSING = ["trestbps", "chol"]

# --------------------------------------------------------------------------- #
# Mapping the 5-level `num` target onto Low / Moderate / High risk
# --------------------------------------------------------------------------- #
RISK_LEVELS = ["Low", "Moderate", "High"]

RISK_MAPPINGS: dict[str, dict[int, str]] = {
    # Recommended. num >= 1 means angiographically significant disease (>50%
    # narrowing); the integer is commonly read as the number of affected major
    # vessels. Single-vessel disease (1) is treated as Moderate, multi-vessel
    # disease (2-4) as High. Class sizes: 411 / 265 / 244.
    "default": {0: "Low", 1: "Moderate", 2: "High", 3: "High", 4: "High"},
    # Alternative suggested in the project brief. Keeps High for the most
    # severe cases only. Class sizes: 411 / 374 / 135.
    "severe_only": {0: "Low", 1: "Moderate", 2: "Moderate", 3: "High", 4: "High"},
}
DEFAULT_RISK_MAPPING = "default"

# --------------------------------------------------------------------------- #
# Fuzzy membership (phase 4)
# --------------------------------------------------------------------------- #
# Linguistic terms for continuous attributes, in increasing order of value.
LINGUISTIC_TERMS = ["Low", "Normal", "High"]
# Training-set quantiles that place the trapezoid break points:
# Low  = (-inf, -inf, q[0], q[1]),  Normal = (q[0], q[1], q[2], q[3]),  High = (q[2], q[3], inf, inf)
TRAPEZOID_QUANTILES = (0.2, 0.4, 0.6, 0.8)
# Clinical units for plots and explanations.
ATTRIBUTE_UNITS = {
    "age": "years", "trestbps": "mm Hg", "chol": "mg/dl", "thalch": "bpm", "oldpeak": "mm",
}
# Where the selected subset is persisted by scripts/phase3_bbso.py.
SELECTED_ATTRIBUTES_JSON = RESULTS_DIR / "03_runtime.json"
