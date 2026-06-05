import os
import pandas as pd
import matplotlib.pyplot as plt

# 본인 경로에 맞게 수정
DATA_DIR = "data/processed/plantvillage/test"

class_counts = {}

for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)

    if os.path.isdir(class_path):
        image_count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        class_counts[class_name] = image_count

df = pd.DataFrame({
    "class_name": list(class_counts.keys()),
    "image_count": list(class_counts.values())
})

df = df.sort_values("image_count", ascending=False)

print(df)

max_count = df["image_count"].max()
min_count = df["image_count"].min()
imbalance_ratio = max_count / min_count

print("\nImbalance Ratio")
print(f"Max / Min = {max_count} / {min_count} = {imbalance_ratio:.2f}")

os.makedirs("results/imbalance", exist_ok=True)
df.to_csv("results/imbalance/class_distribution.csv", index=False)

plt.figure(figsize=(12, 6))
plt.bar(df["class_name"], df["image_count"])
plt.xticks(rotation=45, ha="right")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig("results/imbalance/class_distribution.png", dpi=300)
plt.close()