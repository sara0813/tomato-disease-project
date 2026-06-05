from pathlib import Path

# ============================================================
# Project root
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Dataset paths
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw" / "plantvillage"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "plantvillage"

TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# External datasets
TAIWAN_EXTERNAL_DIR = DATA_DIR / "processed" / "taiwan_external_test"
BANGLADESH_BBOX_EXTERNAL_DIR = DATA_DIR / "processed" / "bangladesh_bbox_external_test"


# ============================================================
# Model paths
# ============================================================
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = {
    "baseline_cnn": MODEL_DIR / "baseline_cnn.keras",
    "densenet121": MODEL_DIR / "densenet121.keras",
    "efficientnetb0": MODEL_DIR / "efficientnetb0.keras",
    "mobilenetv2": MODEL_DIR / "mobilenetv2.keras",
    "efficientnetb0_classweight": MODEL_DIR / "imbalance" / "efficientnetb0_classweight.keras",
}


# ============================================================
# Result paths
# ============================================================
RESULT_DIR = PROJECT_ROOT / "results"

COMMON_RESULT_DIR = RESULT_DIR / "_common"
CLASS_NAMES_PATH = COMMON_RESULT_DIR / "class_names.json"

RESULT_PATHS = {
    "baseline_cnn": RESULT_DIR / "baseline_cnn",
    "densenet121": RESULT_DIR / "densenet121",
    "efficientnetb0": RESULT_DIR / "efficientnetb0",
    "mobilenetv2": RESULT_DIR / "mobilenetv2",
    "efficientnetb0_classweight": RESULT_DIR / "imbalance" / "efficientnetb0_classweight",
    "efficientnetb0_classweight_aug": RESULT_DIR / "imbalance" / "efficientnetb0_classweight_aug",
}

EXTERNAL_RESULT_PATHS = {
    "taiwan": RESULT_DIR / "external" / "taiwan",
    "bangladesh_bbox": RESULT_DIR / "external" / "bangladesh_bbox",
}

IMBALANCE_RESULT_DIR = RESULT_DIR / "imbalance"
CORRUPTION_RESULT_DIR = RESULT_DIR / "corruption"


# ============================================================
# Image settings
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# Training settings
# ============================================================
EPOCHS = 3
SEED = 42


# ============================================================
# Class count
# ============================================================
NUM_CLASSES = 10


# ============================================================
# Directory creation helper
# ============================================================
def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "imbalance").mkdir(parents=True, exist_ok=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    COMMON_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for path in RESULT_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

    for path in EXTERNAL_RESULT_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

    IMBALANCE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CORRUPTION_RESULT_DIR.mkdir(parents=True, exist_ok=True)