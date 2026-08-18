# aux_ai_project/params.py
import numpy as np
np.random.seed(42)   # Reproducibility fix: ensures reported numbers match across runs

# ----------------------
# Dataset selection
# ----------------------
DATASET_BACKEND = "torchvision"
DATASET_NAME = "CITYSCAPES"    # Change this to switch datasets
DATASET_ROOT = "./data"

# ----------------------
# APSS METRIC BOUNDS (IEEE DFT 2025 ALIGNMENT)
# ----------------------
APSS_AREA_LIMIT = 0.58
APSS_POS_Y_LIMIT = 0.45
APSS_SYMMETRY_MAX = 0.20
APSS_TEMPORAL_THRESHOLD = 0.05

CITYSCAPES_ZIPS = {
    "images": "./data/leftImg8bit_trainvaltest.zip",
    "labels": "./data/gtFine_trainvaltest.zip"
}
CITYSCAPES_IGNORE_INDEX = 255

# ----------------------
# GLOBAL DEFAULTS
# ----------------------
DEFAULTS = {
    "INPUT_DIM": 526,
    "NUM_FRAMES": 1000,
    "NUM_UAVS": 5,
    "CLIENTS_PER_ROUND": 5,       # FIX: was 3, but never actually used to
                                  # subsample clients anywhere in
                                  # run_simulation.py -- all NUM_UAVS clients
                                  # participate every round (full-participation
                                  # FedAvg). Set equal to NUM_UAVS so the
                                  # declared config matches actual behavior
                                  # instead of implying partial-participation
                                  # sampling that was never implemented.
    "FL_ROUNDS": 10,              # FIX: was 5, but run_simulation.py silently
                                  # forced max(FL_ROUNDS, 10) everywhere it was
                                  # used, so the simulation always actually ran
                                  # 10 rounds regardless of this value. Declaring
                                  # 10 here makes the config match what actually
                                  # executes -- see the corresponding
                                  # simplification in run_simulation.py.
    "LOCAL_EPOCHS": 1,
    "LEARNING_RATE": 0.0001,     # FIX: was 0.001; paper states eta=1e-4
    "MAX_UPDATE_NORM": 5.0,
    "PHASE_RATIOS": {
        "novel": 0.3,
        "asd": 0.5,
        "fail": 0.75
    }
}

# ----------------------
# Dataset-specific configuration blocks
# ----------------------
DATASET_CONFIGS = {
    "CITYSCAPES":   {"NUM_FRAMES": 500},
    "CIFAR10":      {},
    "CIFAR100":     {},
    "MNIST":        {},
    "FashionMNIST": {},
    "KMNIST":       {},
    "SVHN":         {},
    "CustomSensor": {
        "NUM_FRAMES": 2000,
        "NUM_UAVS": 3,
        "CLIENTS_PER_ROUND": 2,
        "FL_ROUNDS": 10
    },
    "CustomTabular": {
        "NUM_FRAMES": 1500,
        "NUM_UAVS": 3,
        "CLIENTS_PER_ROUND": 2,
        "FL_ROUNDS": 8,
        "LEARNING_RATE": 0.0001,
        "MAX_UPDATE_NORM": 3.0
    },
    "SDKExample": {
        "CLIENTS_PER_ROUND": 3
    }
}

# ----------------------
# Common simulation parameters
# ----------------------
NOVELTY_PROB = 0.3
ASD_PROB = 0.05
SENSOR_FAIL_PROB = 0.05
PRINT_EVERY = 10

# Logic Thresholds
ANOMALY_THRESHOLD = 2.28
SENSOR_THRESHOLD = 0.60
STABILITY_THRESHOLD = 0.30

# Temporal / object tracking
STABILITY_WINDOW = 10
OBJECT_BUFFER_SIZE = 10

# Consensus
CONSENSUS_CLIENTS = 3
CONSENSUS_WINDOW = 3

# Fusion
USE_ATTENTION_FUSION = True

# ----------------------
# Flattening Logic
# ----------------------
if DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(f"Dataset '{DATASET_NAME}' is not defined.")

ACTIVE = {**DEFAULTS, **DATASET_CONFIGS[DATASET_NAME]}

INPUT_DIM         = ACTIVE["INPUT_DIM"]
NUM_FRAMES        = ACTIVE["NUM_FRAMES"]
NUM_UAVS          = ACTIVE["NUM_UAVS"]
CLIENTS_PER_ROUND = ACTIVE["CLIENTS_PER_ROUND"]
FL_ROUNDS         = ACTIVE["FL_ROUNDS"]
LOCAL_EPOCHS      = ACTIVE["LOCAL_EPOCHS"]
LEARNING_RATE     = ACTIVE["LEARNING_RATE"]
MAX_UPDATE_NORM   = ACTIVE["MAX_UPDATE_NORM"]
