from pathlib import Path
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import MODEL_DIR, RESULT_DIR, IMG_SIZE, BATCH_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAIWAN_EXTERNAL_DIR = PROJECT_ROOT / "data" / "processed" / "taiwan_external_test"

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
        "path": MODEL_DIR / "baseline_cnn.keras",
        "preprocess": "rescale",
    },
    "densenet121": {
        "path": MODEL_DIR / "densenet121.keras",
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "mobilenetv2": {
        "path": MODEL_DIR / "mobilenetv2.keras",
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "efficientnetb0": {
        "path": MODEL_DIR / "efficientnetb0.keras",
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
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


def load_external_dataset(preprocess):
    if not TAIWAN_EXTERNAL_DIR.exists():
        print(f"[ERROR] Taiwan external dataset does not exist: {TAIWAN_EXTERNAL_DIR}")
        print("Run this first:")
        print("python src\\prepare_taiwan_external_test.py")
        sys.exit(1)

    ds = tf.keras.utils.image_dataset_from_directory(
        TAIWAN_EXTERNAL_DIR,
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
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        ds = ds.map(
            lambda image, label: (preprocess(tf.cast(image, tf.float32)), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(PLANTVILLAGE_CLASSES)))
    )

    short_names = [shorten_class_name(name) for name in PLANTVILLAGE_CLASSES]

    fig, ax = plt.subplots(figsize=(14, 12))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=short_names
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=True
    )

    ax.set_title(f"{model_name} Taiwan External Test Confusion Matrix", fontsize=15)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.setp(ax.get_xticklabels(), ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    fig.tight_layout()

    save_dir = RESULT_DIR / "external" / "taiwan"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{model_name}_taiwan_confusion_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved: {save_path}")


def evaluate_model(model_name):
    if model_name not in MODEL_CONFIGS:
        print("[ERROR] Unknown model name.")
        print("Available models:")
        for name in MODEL_CONFIGS.keys():
            print(f"- {name}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]

    if not model_path.exists():
        print(f"[ERROR] Model file does not exist: {model_path}")
        print("\nAvailable model files:")
        for file in MODEL_DIR.glob("*.keras"):
            print(f"- {file.name}")
        sys.exit(1)

    print("=" * 70)
    print(f"Taiwan External Test - {model_name}")
    print("=" * 70)

    model = tf.keras.models.load_model(model_path)
    test_ds = load_external_dataset(config["preprocess"])

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
        zero_division=0
    )

    pred_counter = Counter(y_pred)

    print(f"\nTaiwan External Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    print("\nPrediction Distribution:")
    for idx, class_name in enumerate(PLANTVILLAGE_CLASSES):
        print(f"{class_name}: {pred_counter[idx]}")

    save_dir = RESULT_DIR / "external" / "taiwan"
    save_dir.mkdir(parents=True, exist_ok=True)

    report_path = save_dir / f"{model_name}_taiwan_external_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Taiwan External Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nPrediction Distribution:\n")
        for idx, class_name in enumerate(PLANTVILLAGE_CLASSES):
            f.write(f"{class_name}: {pred_counter[idx]}\n")

    print(f"\nReport saved: {report_path}")
    save_confusion_matrix(y_true, y_pred, model_name)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python src\\evaluate_taiwan_external.py efficientnetb0")
        print("python src\\evaluate_taiwan_external.py mobilenetv2")
        print("python src\\evaluate_taiwan_external.py baseline")
        print("python src\\evaluate_taiwan_external.py densenet121")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    evaluate_model(model_name)


if __name__ == "__main__":
    main()