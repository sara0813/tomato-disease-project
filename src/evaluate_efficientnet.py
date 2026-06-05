import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

from config import (
    TEST_DIR,
    MODEL_PATHS,
    RESULT_PATHS,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    ensure_dirs,
)


# ============================================================
# Experiment Setting
# ============================================================

MODEL_NAME = "efficientnetb0_classweight"
MODEL_PATH = MODEL_PATHS[MODEL_NAME]
MODEL_RESULT_DIR = RESULT_PATHS[MODEL_NAME]


def shorten_class_name(name):
    name = name.replace("Tomato___", "")
    name = name.replace("_", " ")

    name = name.replace("Spider mites Two-spotted spider mite", "Spider mites")
    name = name.replace("Tomato Yellow Leaf Curl Virus", "Yellow Leaf Curl")
    name = name.replace("Tomato mosaic virus", "Mosaic virus")
    name = name.replace("Septoria leaf spot", "Septoria")
    name = name.replace("healthy", "Healthy")

    return name


def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = test_ds.class_names

    test_ds = test_ds.map(
        lambda image, label: (
            tf.keras.applications.efficientnet.preprocess_input(
                tf.cast(image, tf.float32)
            ),
            label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return test_ds, class_names


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    short_class_names = [shorten_class_name(name) for name in class_names]

    fig, ax = plt.subplots(figsize=(14, 12))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=short_class_names
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=True
    )

    ax.set_title("EfficientNetB0 + Class Weight Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    plt.setp(ax.get_xticklabels(), ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    fig.tight_layout()

    save_path = MODEL_RESULT_DIR / f"{MODEL_NAME}_confusion_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved to: {save_path}")


def main():
    ensure_dirs()

    if not TEST_DIR.exists():
        print("Test dataset folder does not exist.")
        print("Run this first:")
        print("python src\\split_dataset.py")
        return

    if not MODEL_PATH.exists():
        print("Model file does not exist.")
        print(f"Missing model path: {MODEL_PATH}")
        print("Run this first:")
        print("python src\\train_efficientnet.py")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    test_ds, class_names = load_test_dataset()

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    test_accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"Experiment: {MODEL_NAME}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("=" * 60)
    print(report)

    report_path = MODEL_RESULT_DIR / f"{MODEL_NAME}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {MODEL_NAME}\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)

    result_path = MODEL_RESULT_DIR / f"{MODEL_NAME}_test_result.txt"

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {MODEL_NAME}\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    with open(MODEL_RESULT_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    save_confusion_matrix(y_true, y_pred, class_names)

    print("\nEfficientNetB0 + Class Weight evaluation completed!")
    print(f"Report saved to: {report_path}")
    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()