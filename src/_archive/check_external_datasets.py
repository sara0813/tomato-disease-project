from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Taiwan Raw": PROJECT_ROOT / "data" / "raw" / "taiwan",
    "Bangladesh Raw": PROJECT_ROOT / "data" / "raw" / "bangladesh",
    "Bangladesh Processed": PROJECT_ROOT / "data" / "processed" / "bangladesh",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder_path):
    return sum(
        1
        for file_path in folder_path.rglob("*")
        if file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_class_counts(base_path):
    class_folders = [p for p in base_path.iterdir() if p.is_dir()]

    total_images = 0

    for folder in sorted(class_folders):
        image_count = count_images(folder)
        total_images += image_count
        print(f"  {folder.name}: {image_count}")

    print(f"  Total folders: {len(class_folders)}")
    print(f"  Total images: {total_images}")


def check_dataset(name, dataset_path):
    print("=" * 70)
    print(name)
    print("=" * 70)

    if not dataset_path.exists():
        print(f"{dataset_path} does not exist.")
        print()
        return

    split_names = ["train", "val", "valid", "test"]
    has_split = any((dataset_path / split).exists() for split in split_names)

    if has_split:
        for split in split_names:
            split_path = dataset_path / split

            if split_path.exists():
                print(f"\n[{split.upper()}]")
                print_class_counts(split_path)
    else:
        print("\n[NO SPLIT / DIRECT FOLDERS]")
        print_class_counts(dataset_path)

    print()


def main():
    for name, path in DATASETS.items():
        check_dataset(name, path)


if __name__ == "__main__":
    main()