
# Helper file for used paths

import os
from os.path import abspath

# Define the common paths
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = abspath(os.path.join(FILE_PATH, ".."))

DATA_PATH = abspath(os.path.join(BASE_PATH, "data"))
FIGURES_PATH = abspath(os.path.join(BASE_PATH, "figures"))

# Original cleaned ECG data
CLEANED_DATA_PATH = abspath(
    os.path.join(DATA_PATH, "cleaned")
)

# Cleaned ECG data without quality filtering
CLEANED_NO_QUALITY_FILTER_PATH = abspath(
    os.path.join(DATA_PATH, "cleaned_no_quality_filter")
)

# Feature data root directory
FEATURE_DATA_PATH = abspath(
    os.path.join(DATA_PATH, "features")
)

# Optional separate feature directory
FEATURE_NO_QUALITY_FILTER_PATH = abspath(
    os.path.join(DATA_PATH, "features_no_quality_filter")
)

# Raw ECG data
RAW_DATA_PATH = abspath(
    os.path.join(DATA_PATH, "raw", "Raw ECG project")
)

# Other project directories
MODELS_PATH = abspath(os.path.join(BASE_PATH, "models"))
CONFIG_PATH = abspath(os.path.join(BASE_PATH, "configs"))
RESULTS_PATH = abspath(os.path.join(BASE_PATH, "results"))
