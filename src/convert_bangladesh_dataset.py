from pathlib import Path
import shutil
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "bangladesh"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "bangladesh"

CLASS_NAMES = {
    0: "Early_Blight",
    1: "Black_Spot",
    2: "Late_Blight",
    3: "Leaf_Mold",
    4: "Bacterial_Spot",
    5: "Target_Spot",
    6: "Healthy",
}

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_folder(base_dir, candidates):
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    return None


def find_image(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def read_class_id(label_path):
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        return None, "empty_label"

    class_ids = []

    for line in lines:
        parts = line.split()
        if len(parts) < 1:
            continue

        try:
            class_ids.append(int(parts[0]))
        except ValueError:
            return None, "invalid_class_id"

    if not class_ids:
        return None, "no_class_id"

    unique_class_ids = set(class_ids)

    if len(unique_class_ids) > 1:
        return None, "multiple_classes"

    class_id = class_ids[0]

    if class_id not in CLASS_NAMES:
        return None, "unknown_class_id"

    return class_id, None


def prepare_output_dirs():
    for output_split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES.values():
            (OUTPUT_DIR / output_split / class_name).mkdir(parents=True, exist_ok=True)


def convert_split(source_split, output_split):
    split_dir = SOURCE_DIR / source_split

    if not split_dir.exists():
        print(f"[SKIP] {source_split} folder does not exist.")
        return Counter()

    images_dir = find_folder(split_dir, ["images", "image", "Images", "Image"])
    labels_dir = find_folder(split_dir, ["labels", "label", "Labels", "Label"])

    if images_dir is None or labels_dir is None:
        print(f"[SKIP] {source_split}: images or labels folder not found.")
        return Counter()

    stats = Counter()

    for label_path in labels_dir.glob("*.txt"):
        class_id, error = read_class_id(label_path)

        if error is not None:
            stats[f"skipped_{error}"] += 1
            continue

        image_path = find_image(images_dir, label_path.stem)

        if image_path is None:
            stats["skipped_image_not_found"] += 1
            continue

        class_name = CLASS_NAMES[class_id]
        output_path = OUTPUT_DIR / output_split / class_name / image_path.name

        shutil.copy2(image_path, output_path)
        stats[class_name] += 1
        stats["total_copied"] += 1

    return stats


def main():
    prepare_output_dirs()

    total_stats = {}

    for source_split, output_split in SPLIT_MAP.items():
        stats = convert_split(source_split, output_split)

        if stats:
            total_stats[output_split] = stats

    print("=" * 60)
    print("Bangladesh dataset conversion completed!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    for split_name, stats in total_stats.items():
        print(f"\n[{split_name.upper()}]")
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()