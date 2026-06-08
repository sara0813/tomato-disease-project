from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import RAW_DATA_DIR


DATA_DIR = RAW_DATA_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("Dataset path:", DATA_DIR.resolve())
    print("-" * 60)

    if not DATA_DIR.exists():
        print("Dataset folder does not exist.")
        return

    class_folders = [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]

    print(f"Number of classes: {len(class_folders)}")
    print("-" * 60)

    total_images = 0

    for class_folder in sorted(class_folders):
        image_count = sum(
            1 for file in class_folder.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_images += image_count
        print(f"{class_folder.name}: {image_count}")

    print("-" * 60)
    print(f"Total images: {total_images}")


if __name__ == "__main__":
    main()