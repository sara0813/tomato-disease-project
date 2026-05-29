from pathlib import Path

# Dataset paths
RAW_DATA_DIR = Path("data/raw/plantvillage")
PROCESSED_DATA_DIR = Path("data/processed/plantvillage")

TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Model paths
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training settings
EPOCHS = 3
SEED = 42

# Class count
NUM_CLASSES = 10