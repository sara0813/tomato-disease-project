from pathlib import Path
import sys
import random
import shutil

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SEED


RAW_DIR = RAW_DATA_DIR
OUTPUT_DIR = PROCESSED_DATA_DIR

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split_name in ["train", "val", "test"]:
        make_dir(OUTPUT_DIR / split_name)


def copy_images(image_list, class_name, split_name):
    target_dir = OUTPUT_DIR / split_name / class_name
    make_dir(target_dir)

    for image_path in image_list:
        shutil.copy2(image_path, target_dir / image_path.name)


def main():
    if not RAW_DIR.exists():
        print(f"Raw dataset folder does not exist: {RAW_DIR}")
        return

    if OUTPUT_DIR.name != "plantvillage":
        print(f"[ERROR] OUTPUT_DIR must be the PlantVillage processed folder, but got: {OUTPUT_DIR}")
        print("Check PROCESSED_DATA_DIR in config.py")
        return

    class_folders = [folder for folder in RAW_DIR.iterdir() if folder.is_dir()]
    
    if len(class_folders) == 0:
        print("No class folders found.")
        return

    reset_output_dir()

    random.seed(SEED)

    print(f"Found {len(class_folders)} classes.")
    print(f"Raw dataset: {RAW_DIR}")
    print(f"Output dataset: {OUTPUT_DIR}")
    print("-" * 60)

    for class_folder in sorted(class_folders):
        class_name = class_folder.name

        images = sorted([
            file for file in class_folder.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ])

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        copy_images(train_images, class_name, "train")
        copy_images(val_images, class_name, "val")
        copy_images(test_images, class_name, "test")

        print(f"{class_name}")
        print(f"  Total: {total}")
        print(f"  Train: {len(train_images)}")
        print(f"  Val:   {len(val_images)}")
        print(f"  Test:  {len(test_images)}")
        print("-" * 60)

    print("Dataset split completed!")


if __name__ == "__main__":
    main()