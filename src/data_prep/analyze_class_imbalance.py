from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import TRAIN_DIR, IMBALANCE_RESULT_DIR, ensure_dirs


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_images_by_class(data_dir):
    class_counts = {}

    if not data_dir.exists():
        print(f"[ERROR] Dataset folder does not exist: {data_dir}")
        return class_counts

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        image_count = len([
            image_path
            for image_path in class_dir.rglob("*")
            if image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ])

        class_counts[class_dir.name] = image_count

    return class_counts


def main():
    ensure_dirs()

    class_counts = count_images_by_class(TRAIN_DIR)

    if not class_counts:
        print("[ERROR] No class images found.")
        return

    df = pd.DataFrame({
        "class_name": list(class_counts.keys()),
        "image_count": list(class_counts.values())
    })

    df = df.sort_values("image_count", ascending=False)

    print("\nClass Distribution - Train Dataset")
    print(df)

    max_count = df["image_count"].max()
    min_count = df["image_count"].min()

    print("\nImbalance Ratio")

    if min_count == 0:
        print(f"Max / Min = {max_count} / {min_count}")
        print("[WARNING] Minimum class count is 0, so imbalance ratio cannot be calculated.")
        imbalance_ratio = None
    else:
        imbalance_ratio = max_count / min_count
        print(f"Max / Min = {max_count} / {min_count} = {imbalance_ratio:.2f}")

    csv_path = IMBALANCE_RESULT_DIR / "class_distribution_train.csv"
    png_path = IMBALANCE_RESULT_DIR / "class_distribution_train.png"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 6))
    plt.bar(df["class_name"], df["image_count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Class Distribution - Train Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("\nSaved files:")
    print(f"- {csv_path}")
    print(f"- {png_path}")


if __name__ == "__main__":
    main()