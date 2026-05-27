from pathlib import Path
import random
import shutil

RAW_DIR = Path("data/raw/plantvillage")
OUTPUT_DIR = Path("data/processed/plantvillage")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

random.seed(SEED)


def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def copy_images(image_list, class_name, split_name):
    target_dir = OUTPUT_DIR / split_name / class_name
    make_dir(target_dir)

    for image_path in image_list:
        shutil.copy2(image_path, target_dir / image_path.name)


def main():
    if not RAW_DIR.exists():
        print(f"Raw dataset folder does not exist: {RAW_DIR}")
        return

    class_folders = [folder for folder in RAW_DIR.iterdir() if folder.is_dir()]

    if len(class_folders) == 0:
        print("No class folders found.")
        return

    print(f"Found {len(class_folders)} classes.")
    print("-" * 60)

    for class_folder in sorted(class_folders):
        class_name = class_folder.name

        images = [
            file for file in class_folder.rglob("*")
            if file.suffix in IMAGE_EXTENSIONS
        ]

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