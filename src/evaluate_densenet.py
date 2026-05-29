import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import TEST_DIR, MODEL_DIR, RESULT_DIR, IMG_SIZE, BATCH_SIZE


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
            tf.keras.applications.densenet.preprocess_input(tf.cast(image, tf.float32)),
            label
        )
    )

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

    ax.set_title("DenseNet121 Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    plt.setp(ax.get_xticklabels(), ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    fig.tight_layout()

    save_path = RESULT_DIR / "confusion_matrix" / "densenet121_confusion_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    model_path = MODEL_DIR / "densenet121.keras"

    if not model_path.exists():
        print("Model file does not exist.")
        print("Run this first:")
        print("python src\\train_densenet.py")
        return

    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "confusion_matrix").mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    test_ds, class_names = load_test_dataset()

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print(report)

    with open(
        RESULT_DIR / "reports" / "densenet121_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    with open(
        RESULT_DIR / "reports" / "class_names.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    save_confusion_matrix(y_true, y_pred, class_names)

    print("DenseNet121 evaluation completed!")


if __name__ == "__main__":
    main()