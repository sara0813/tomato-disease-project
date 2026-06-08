from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import IMBALANCE_RESULT_DIR, ensure_dirs


MODEL_NAME = "efficientnetb0"

data = {
    "class_name": [
        "Bacterial spot",
        "Early blight",
        "Late blight",
        "Leaf Mold",
        "Septoria leaf spot",
        "Spider mites",
        "Target Spot",
        "Yellow Leaf Curl Virus",
        "Mosaic virus",
        "Healthy"
    ],
    "f1_score": [
        0.8772,
        0.5209,
        0.9043,
        0.7654,
        0.7797,
        0.8363,
        0.7240,
        0.9712,
        0.8491,
        0.9031
    ]
}


def main():
    ensure_dirs()

    df = pd.DataFrame(data)
    df = df.sort_values("f1_score", ascending=True)

    csv_path = IMBALANCE_RESULT_DIR / f"{MODEL_NAME}_classwise_f1.csv"
    png_path = IMBALANCE_RESULT_DIR / f"{MODEL_NAME}_classwise_f1.png"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 6))
    plt.bar(df["class_name"], df["f1_score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Class-wise F1-score of EfficientNetB0")
    plt.xlabel("Class")
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(df)
    print("\nSaved files:")
    print(f"- {csv_path}")
    print(f"- {png_path}")


if __name__ == "__main__":
    main()