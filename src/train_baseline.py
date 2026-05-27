from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODEL_DIR,
    RESULT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    SEED
)

from models import build_baseline_cnn


def prepare_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "graphs").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    with open(RESULT_DIR / "reports" / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    return train_ds, val_ds, test_ds, class_names


def preprocess_train(image, label):
    image = tf.cast(image, tf.float32) / 255.0

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    image = data_augmentation(image, training=True)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def preprocess_eval(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def optimize_dataset(train_ds, val_ds, test_ds):
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=autotune)
    val_ds = val_ds.map(preprocess_eval, num_parallel_calls=autotune)
    test_ds = test_ds.map(preprocess_eval, num_parallel_calls=autotune)

    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds


def plot_history(history):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(RESULT_DIR / "reports" / "baseline_history.csv", index=False)

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline CNN Accuracy")
    plt.legend()
    plt.savefig(RESULT_DIR / "graphs" / "baseline_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline CNN Loss")
    plt.legend()
    plt.savefig(RESULT_DIR / "graphs" / "baseline_loss.png")
    plt.close()


def main():
    prepare_dirs()

    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        print("Processed dataset folders do not exist.")
        print("Run this first:")
        print("python src\\split_dataset.py")
        return

    train_ds, val_ds, test_ds, class_names = load_datasets()
    train_ds, val_ds, test_ds = optimize_dataset(train_ds, val_ds, test_ds)

    num_classes = len(class_names)

    print("Class names:", class_names)
    print("Number of classes:", num_classes)

    model = build_baseline_cnn(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=num_classes
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "baseline_cnn.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(test_ds)

    print("-" * 50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("-" * 50)

    with open(RESULT_DIR / "reports" / "baseline_test_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")


if __name__ == "__main__":
    main()