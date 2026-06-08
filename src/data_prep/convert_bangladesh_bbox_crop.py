from pathlib import Path
import sys
import shutil
from collections import defaultdict

from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR, BANGLADESH_BBOX_EXTERNAL_DIR


RAW_DIR = DATA_DIR / "raw" / "bangladesh"
OUT_DIR = BANGLADESH_BBOX_EXTERNAL_DIR


PLANTVILLAGE_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Bangladesh YOLO class id -> PlantVillage class
# 0 Early Blight
# 1 Black Spot
# 2 Late Blight
# 3 Leaf Mold
# 4 Bacterial Spot
# 5 Target Spot
# 6 Healthy
CLASS_ID_MAPPING = {
    0: "Tomato___Early_blight",
    1: None,  # Black Spot은 PlantVillage 10개 클래스와 정확히 매칭 안 되므로 제외
    2: "Tomato___Late_blight",
    3: "Tomato___Leaf_Mold",
    4: "Tomato___Bacterial_spot",
    5: "Tomato___Target_Spot",
    6: "Tomato___healthy",
}

SOURCE_SPLITS = ["train", "valid", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def reset_output_dir():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in PLANTVILLAGE_CLASSES:
        (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)


def yolo_to_pixel_bbox(x_center, y_center, width, height, img_w, img_h):
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    return x1, y1, x2, y2


def convert_bbox_crop():
    if not RAW_DIR.exists():
        print(f"[ERROR] Raw Bangladesh folder does not exist: {RAW_DIR}")
        return

    counts = defaultdict(int)

    total_images = 0
    total_labels = 0
    total_crops = 0

    skipped_no_label = 0
    skipped_invalid_line = 0
    skipped_black_spot = 0
    skipped_small_crop = 0

    for split in SOURCE_SPLITS:
        split_dir = RAW_DIR / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not split_dir.exists():
            continue

        if not images_dir.exists() or not labels_dir.exists():
            print(f"[SKIP] images/labels folder missing in: {split_dir}")
            continue

        print(f"\nProcessing split: {split}")

        image_files = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_files:
            total_images += 1

            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                skipped_no_label += 1
                continue

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size

                lines = label_path.read_text(encoding="utf-8").strip().splitlines()

                for line_idx, line in enumerate(lines):
                    if not line.strip():
                        continue

                    parts = line.strip().split()

                    if len(parts) < 5:
                        skipped_invalid_line += 1
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        skipped_invalid_line += 1
                        continue

                    total_labels += 1

                    target_class = CLASS_ID_MAPPING.get(class_id)

                    if target_class is None:
                        skipped_black_spot += 1
                        continue

                    x1, y1, x2, y2 = yolo_to_pixel_bbox(
                        x_center, y_center, width, height, img_w, img_h
                    )

                    if x2 <= x1 or y2 <= y1:
                        skipped_invalid_line += 1
                        continue

                    crop_w = x2 - x1
                    crop_h = y2 - y1

                    if crop_w < 5 or crop_h < 5:
                        skipped_small_crop += 1
                        continue

                    cropped = img.crop((x1, y1, x2, y2))

                    save_dir = OUT_DIR / target_class
                    save_name = (
                        f"bangladesh_{split}_{image_path.stem}_"
                        f"bbox{line_idx:03d}_class{class_id}.jpg"
                    )
                    save_path = save_dir / save_name

                    cropped.save(save_path, quality=95)

                    counts[target_class] += 1
                    total_crops += 1

    print("\n" + "=" * 70)
    print("Bangladesh bbox crop external dataset created!")
    print(f"Output folder: {OUT_DIR}")
    print("=" * 70)

    print("\nCrop counts:")
    for class_name in PLANTVILLAGE_CLASSES:
        print(f"{class_name}: {counts[class_name]}")

    print("\nSummary:")
    print(f"Total images checked: {total_images}")
    print(f"Total YOLO labels read: {total_labels}")
    print(f"Total crops saved: {total_crops}")
    print(f"Skipped no label: {skipped_no_label}")
    print(f"Skipped invalid line: {skipped_invalid_line}")
    print(f"Skipped Black Spot: {skipped_black_spot}")
    print(f"Skipped small crop: {skipped_small_crop}")


def main():
    reset_output_dir()
    convert_bbox_crop()


if __name__ == "__main__":
    main()