from pathlib import Path
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    TAIWAN_EXTERNAL_DIR,
    BANGLADESH_BBOX_EXTERNAL_DIR,
    MODEL_PATHS,
    EXTERNAL_RESULT_PATHS,
    IMG_SIZE,
    BATCH_SIZE,
    ensure_dirs,
)


PLANTVILLAGE_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


MODEL_CONFIGS = {
    "baseline": {
        "path": MODEL_PATHS["baseline_cnn"],
        "preprocess": "rescale",
    },
    "densenet121": {
        "path": MODEL_PATHS["densenet121"],
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "mobilenetv2": {
        "path": MODEL_PATHS["mobilenetv2"],
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "efficientnetb0": {
        "path": MODEL_PATHS["efficientnetb0"],
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "efficientnetb0_classweight": {
        "path": MODEL_PATHS["efficientnetb0_classweight"],
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
}


DATASET_CONFIGS = {
    "taiwan": {
        "name": "Taiwan External Test",
        "dataset_dir": TAIWAN_EXTERNAL_DIR,
        "result_dir": EXTERNAL_RESULT_PATHS["taiwan"],
        "file_prefix": "taiwan",
        "prepare_command": "python src\\data_prep\\prepare_taiwan_external_test.py",
    },
    "bangladesh_bbox": {
        "name": "Bangladesh BBox External Test",
        "dataset_dir": BANGLADESH_BBOX_EXTERNAL_DIR,
        "result_dir": EXTERNAL_RESULT_PATHS["bangladesh_bbox"],
        "file_prefix": "bangladesh_bbox",
        "prepare_command": "python src\\data_prep\\convert_bangladesh_bbox_crop.py",
    },
}


def shorten_class_name(name):
    name = name.replace("Tomato___", "")
    name = name.replace("_", " ")
    name = name.replace("Spider mites Two-spotted spider mite", "Spider mites")
    name = name.replace("Tomato Yellow Leaf Curl Virus", "Yellow Leaf Curl")
    name = name.replace("Tomato mosaic virus", "Mosaic virus")
    name = name.replace("Septoria leaf spot", "Septoria")
    name = name.replace("healthy", "Healthy")
    return name


def load_external_dataset(dataset_config, preprocess):
    dataset_dir = dataset_config["dataset_dir"]

    if not dataset_dir.exists():
        print(f"[ERROR] External dataset does not exist: {dataset_dir}")
        print("Run this first:")
        print(dataset_config["prepare_command"])
        sys.exit(1)

    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        class_names=PLANTVILLAGE_CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    if preprocess == "rescale":
        ds = ds.map(
            lambda image, label: (tf.cast(image, tf.float32) / 255.0, label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda image, label: (preprocess(tf.cast(image, tf.float32)), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_confusion_matrix(y_true, y_pred, model_name, dataset_config):
    result_dir = dataset_config["result_dir"]
    file_prefix = dataset_config["file_prefix"]
    dataset_name = dataset_config["name"]

    result_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(PLANTVILLAGE_CLASSES))),
    )

    short_names = [shorten_class_name(name) for name in PLANTVILLAGE_CLASSES]

    fig, ax = plt.subplots(figsize=(14, 12))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=short_names,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=True,
    )

    ax.set_title(f"{model_name} {dataset_name} Confusion Matrix", fontsize=15)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.setp(ax.get_xticklabels(), ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    fig.tight_layout()

    save_path = result_dir / f"{model_name}_{file_prefix}_confusion_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved: {save_path}")


def evaluate_model(dataset_name, model_name):
    if dataset_name not in DATASET_CONFIGS:
        print("[ERROR] Unknown dataset name.")
        print("Available datasets:")
        for name in DATASET_CONFIGS.keys():
            print(f"- {name}")
        sys.exit(1)

    if model_name not in MODEL_CONFIGS:
        print("[ERROR] Unknown model name.")
        print("Available models:")
        for name in MODEL_CONFIGS.keys():
            print(f"- {name}")
        sys.exit(1)

    dataset_config = DATASET_CONFIGS[dataset_name]
    model_config = MODEL_CONFIGS[model_name]

    model_path = model_config["path"]
    result_dir = dataset_config["result_dir"]
    file_prefix = dataset_config["file_prefix"]

    if not model_path.exists():
        print(f"[ERROR] Model file does not exist: {model_path}")
        print("\nExpected model files:")
        for name, config in MODEL_CONFIGS.items():
            print(f"- {name}: {config['path']}")
        sys.exit(1)

    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"{dataset_config['name']} - {model_name}")
    print("=" * 80)

    model = tf.keras.models.load_model(str(model_path))
    test_ds = load_external_dataset(dataset_config, model_config["preprocess"])

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(PLANTVILLAGE_CLASSES))),
        target_names=PLANTVILLAGE_CLASSES,
        digits=4,
        zero_division=0,
    )

    pred_counter = Counter(y_pred)

    print(f"\n{dataset_config['name']} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    print("\nPrediction Distribution:")
    for idx, class_name in enumerate(PLANTVILLAGE_CLASSES):
        print(f"{class_name}: {pred_counter[idx]}")

    report_path = result_dir / f"{model_name}_{file_prefix}_external_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_config['name']}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report)

        f.write("\n\nPrediction Distribution:\n")
        for idx, class_name in enumerate(PLANTVILLAGE_CLASSES):
            f.write(f"{class_name}: {pred_counter[idx]}\n")

    print(f"\nReport saved: {report_path}")

    save_confusion_matrix(y_true, y_pred, model_name, dataset_config)


def print_usage():
    print("Usage:")
    print("python src\\evaluate\\evaluate_external.py <dataset_name> <model_name>")
    print()
    print("Available datasets:")
    for dataset_name in DATASET_CONFIGS.keys():
        print(f"- {dataset_name}")
    print()
    print("Available models:")
    for model_name in MODEL_CONFIGS.keys():
        print(f"- {model_name}")
    print()
    print("Examples:")
    print("python src\\evaluate\\evaluate_external.py taiwan efficientnetb0")
    print("python src\\evaluate\\evaluate_external.py taiwan efficientnetb0_classweight")
    print("python src\\evaluate\\evaluate_external.py bangladesh_bbox efficientnetb0")
    print("python src\\evaluate\\evaluate_external.py bangladesh_bbox mobilenetv2")


def main():
    ensure_dirs()

    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    dataset_name = sys.argv[1].lower()
    model_name = sys.argv[2].lower()

    evaluate_model(dataset_name, model_name)


if __name__ == "__main__":
    main()