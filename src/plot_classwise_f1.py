import pandas as pd
import matplotlib.pyplot as plt
import os

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

df = pd.DataFrame(data)
df = df.sort_values("f1_score", ascending=True)

os.makedirs("results/imbalance", exist_ok=True)

plt.figure(figsize=(12, 6))
plt.bar(df["class_name"], df["f1_score"])
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.title("Class-wise F1-score of EfficientNetB0")
plt.xlabel("Class")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig("results/imbalance/efficientnetb0_classwise_f1.png", dpi=300)
plt.close()

df.to_csv("results/imbalance/efficientnetb0_classwise_f1.csv", index=False)

print(df)