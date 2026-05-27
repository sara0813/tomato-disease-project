from pathlib import Path

DATA_DIR = Path("data/processed/plantvillage")
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


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
            if class_dir.is_dir():
                count = sum(
                    1 for file in class_dir.rglob("*")
                    if file.suffix in IMAGE_EXTENSIONS
                )
                total += count
                print(f"{class_dir.name}: {count}")

        print("-" * 60)
        print(f"Total {split}: {total}")


if __name__ == "__main__":
    main()