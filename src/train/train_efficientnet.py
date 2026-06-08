import json
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

layers = tf.keras.layers

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODEL_PATHS,
    RESULT_PATHS,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    SEED,
    ensure_dirs,
)


# ============================================================
# Experiment Setting
# ============================================================

MODEL_NAME = "efficientnetb0"

MODEL_PATH = MODEL_PATHS[MODEL_NAME]
MODEL_RESULT_DIR = RESULT_PATHS[MODEL_NAME]


# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
    layers.RandomContrast(0.1, seed=SEED),
], name="data_augmentation")


# ============================================================
# Load Dataset
# ============================================================

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

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    return train_ds, val_ds, test_ds, class_names


# ============================================================
# Preprocessing
# ============================================================

def preprocess_train(image, label):
    image = tf.cast(image, tf.float32)
    image = data_augmentation(image, training=True)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


def preprocess_eval(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


def optimize_dataset(train_ds, val_ds, test_ds):
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=autotune)
    val_ds = val_ds.map(preprocess_eval, num_parallel_calls=autotune)
    test_ds = test_ds.map(preprocess_eval, num_parallel_calls=autotune)

    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds


# ============================================================
# Build Model
# ============================================================

def build_efficientnetb0(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_Tomato")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Plot History
# ============================================================

def plot_history(history):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODEL_RESULT_DIR / f"{MODEL_NAME}_history.csv", index=False)

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EfficientNetB0 Accuracy")
    plt.legend()
    plt.savefig(MODEL_RESULT_DIR / f"{MODEL_NAME}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EfficientNetB0 Loss")
    plt.legend()
    plt.savefig(MODEL_RESULT_DIR / f"{MODEL_NAME}_loss.png")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ensure_dirs()

    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        print("Processed dataset folders do not exist.")
        print("Run this first:")
        print("python src\\data_prep\\split_dataset.py")
        return

    train_ds, val_ds, test_ds, class_names = load_datasets()
    train_ds, val_ds, test_ds = optimize_dataset(train_ds, val_ds, test_ds)

    num_classes = len(class_names)

    print("Class names:", class_names)
    print("Number of classes:", num_classes)
    print("Experiment:", MODEL_NAME)

    model = build_efficientnetb0(num_classes=num_classes)

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
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

    # EarlyStopping으로 복원된 best weight를 다시 저장
    model.save(str(MODEL_PATH))

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(test_ds)

    print("-" * 50)
    print(f"Experiment: {MODEL_NAME}")
    print(f"EfficientNetB0 Test Loss: {test_loss:.4f}")
    print(f"EfficientNetB0 Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print("-" * 50)

    with open(MODEL_RESULT_DIR / f"{MODEL_NAME}_test_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Experiment: {MODEL_NAME}\n")
        f.write(f"EfficientNetB0 Test Loss: {test_loss:.4f}\n")
        f.write(f"EfficientNetB0 Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Model saved to: {MODEL_PATH}\n")


if __name__ == "__main__":
    main()