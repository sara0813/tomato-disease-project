import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import TEST_DIR, MODEL_DIR, RESULT_DIR, IMG_SIZE, BATCH_SIZE


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
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, label)
    )

    return test_ds, class_names


def main():
    model_path = MODEL_DIR / "baseline_cnn.keras"

    if not model_path.exists():
        print("Model file does not exist.")
        print("Run this first:")
        print("python src\\train_baseline.py")
        return

    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "confusion_matrix").mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    test_ds, class_names = load_test_dataset()

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print(report)

    with open(RESULT_DIR / "reports" / "baseline_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(RESULT_DIR / "reports" / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    plt.figure(figsize=(12, 10))
    disp.plot(cmap="Blues", xticks_rotation=90)
    plt.title("Baseline CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix" / "baseline_confusion_matrix.png")
    plt.close()

    print("Evaluation completed!")


if __name__ == "__main__":
    main()