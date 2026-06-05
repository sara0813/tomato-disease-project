from pathlib import Path

DATA_DIR = Path("data/raw/plantvillage")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

print("Dataset path:", DATA_DIR.resolve())
print("-" * 60)

if not DATA_DIR.exists():
    print("Dataset folder does not exist.")
else:
    class_folders = [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]

    print(f"Number of classes: {len(class_folders)}")
    print("-" * 60)

    total_images = 0

    for class_folder in sorted(class_folders):
        image_count = sum(
            1 for file in class_folder.rglob("*")
            if file.suffix in IMAGE_EXTENSIONS
        )

        total_images += image_count
        print(f"{class_folder.name}: {image_count}")

    print("-" * 60)
    print(f"Total images: {total_images}")