from pathlib import Path

# Project directories
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"

PROCESSED_DIR = DATA_DIR / "processed"

DATASET_DIR = DATA_DIR / "raw" / "ptb-xl"

# PTB-XL metadata
METADATA_FILE = DATASET_DIR / "ptbxl_database.csv"
SCP_FILE = DATASET_DIR / "scp_statements.csv"

# ECG signal records
RECORDS_DIR = DATASET_DIR / "records100"

# Output directories
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# General settings
SEED = 42
SAMPLING_RATE = 100

BATCH_SIZE = 16
NUM_WORKERS = 0

TRAIN_STATS_FILE = METRICS_DIR / "train_signal_stats.json"

# Training configuration
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 1e-4

# Learning-rate scheduler
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5

# Validation
VALIDATION_THRESHOLD = 0.5
PRIMARY_METRIC = "macro_auprc"
