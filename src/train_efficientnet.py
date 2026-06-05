import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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


# ============================================================
# Experiment Setting
# ============================================================

EXPERIMENT_NAME = "efficientnetb0_classweight"
USE_CLASS_WEIGHT = True

MODEL_SAVE_DIR = MODEL_DIR / "imbalance"
EXPERIMENT_RESULT_DIR = RESULT_DIR / "imbalance" / EXPERIMENT_NAME
GRAPH_DIR = EXPERIMENT_RESULT_DIR / "graphs"
REPORT_DIR = EXPERIMENT_RESULT_DIR / "reports"


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
# Prepare Directories
# ============================================================

def prepare_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    (RESULT_DIR / "graphs").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

    # 기존 평가 코드와의 호환성을 위해 기존 위치에도 저장
    with open(RESULT_DIR / "reports" / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    # 이번 실험 결과 폴더에도 따로 저장
    with open(REPORT_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    return train_ds, val_ds, test_ds, class_names


# ============================================================
# Class Weight
# ============================================================

def make_class_weight(class_names):
    """
    TRAIN_DIR 안의 클래스별 이미지 개수를 기준으로 class weight 계산.
    데이터가 적은 클래스일수록 더 큰 weight가 부여됨.
    """

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    labels = []
    class_counts = {}

    for class_idx, class_name in enumerate(class_names):
        class_dir = TRAIN_DIR / class_name

        image_count = 0

        if class_dir.exists():
            for image_path in class_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                    image_count += 1

        class_counts[class_name] = image_count
        labels.extend([class_idx] * image_count)

    labels = np.array(labels)

    if len(labels) == 0:
        raise ValueError("No training images found. Please check TRAIN_DIR.")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels
    )

    class_weight_dict = {
        i: float(weight)
        for i, weight in enumerate(class_weights)
    }

    print("\n" + "=" * 60)
    print("Class Weight Applied")
    print("=" * 60)

    rows = []

    for class_idx, class_name in enumerate(class_names):
        image_count = class_counts[class_name]
        weight = class_weight_dict[class_idx]

        print(f"{class_idx} - {class_name}: count={image_count}, weight={weight:.4f}")

        rows.append({
            "class_index": class_idx,
            "class_name": class_name,
            "image_count": image_count,
            "class_weight": weight
        })

    class_weight_df = pd.DataFrame(rows)
    class_weight_df.to_csv(REPORT_DIR / "class_weight_info.csv", index=False)

    with open(REPORT_DIR / "class_weight_dict.json", "w", encoding="utf-8") as f:
        json.dump(class_weight_dict, f, ensure_ascii=False, indent=4)

    print("=" * 60 + "\n")

    return class_weight_dict


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

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_Tomato_ClassWeight")

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
    history_df.to_csv(REPORT_DIR / f"{EXPERIMENT_NAME}_history.csv", index=False)

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EfficientNetB0 + Class Weight Accuracy")
    plt.legend()
    plt.savefig(GRAPH_DIR / f"{EXPERIMENT_NAME}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EfficientNetB0 + Class Weight Loss")
    plt.legend()
    plt.savefig(GRAPH_DIR / f"{EXPERIMENT_NAME}_loss.png")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    prepare_dirs()

    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        print("Processed dataset folders do not exist.")
        print("Run this first:")
        print("python src\\split_dataset.py")
        return

    train_ds, val_ds, test_ds, class_names = load_datasets()

    class_weight_dict = None

    if USE_CLASS_WEIGHT:
        class_weight_dict = make_class_weight(class_names)

    train_ds, val_ds, test_ds = optimize_dataset(train_ds, val_ds, test_ds)

    num_classes = len(class_names)

    print("Class names:", class_names)
    print("Number of classes:", num_classes)
    print("Experiment:", EXPERIMENT_NAME)

    model = build_efficientnetb0(num_classes=num_classes)

    model.summary()

    model_save_path = MODEL_SAVE_DIR / f"{EXPERIMENT_NAME}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
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
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # EarlyStopping으로 복원된 best weight를 다시 저장
    model.save(model_save_path)

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(test_ds)

    print("-" * 50)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"EfficientNetB0 + Class Weight Test Loss: {test_loss:.4f}")
    print(f"EfficientNetB0 + Class Weight Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_save_path}")
    print("-" * 50)

    with open(REPORT_DIR / f"{EXPERIMENT_NAME}_test_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Experiment: {EXPERIMENT_NAME}\n")
        f.write(f"EfficientNetB0 + Class Weight Test Loss: {test_loss:.4f}\n")
        f.write(f"EfficientNetB0 + Class Weight Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Model saved to: {model_save_path}\n")


if __name__ == "__main__":
    main()