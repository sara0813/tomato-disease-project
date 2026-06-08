from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROCESSED_DATA_DIR


DATA_DIR = PROCESSED_DATA_DIR
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = DATA_DIR / split

        print("=" * 60)
        print(f"{split.upper()} SET")
        print("=" * 60)

        if not split_dir.exists():
            print(f"{split_dir} does not exist.")
            continue

        total = 0

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            count = sum(
                1
                for file in class_dir.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )

            total += count
            print(f"{class_dir.name}: {count}")

        print("-" * 60)
        print(f"Total {split}: {total}")
        print()


if __name__ == "__main__":
    main()