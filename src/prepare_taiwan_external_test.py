from pathlib import Path
import shutil
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_TAIWAN_DIR = PROJECT_ROOT / "data" / "raw" / "taiwan"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "taiwan_external_test"

# PlantVillage 모델이 학습한 10개 클래스 이름
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

# Taiwan 데이터셋 클래스명 → PlantVillage 클래스명
CLASS_MAPPING = {
    "Bacterial spot": "Tomato___Bacterial_spot",
    "Late blight": "Tomato___Late_blight",
    "health": "Tomato___healthy",
    "healthy": "Tomato___healthy",
    "Healthy": "Tomato___healthy",
}

# Taiwan은 train/test가 있지만, 우리는 Taiwan으로 학습하지 않으니까
# train + test 전체를 외부 테스트 데이터로 사용
SOURCE_SPLITS = ["train", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def reset_output_dir():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 나중에 모델의 10개 클래스 순서와 맞추기 위해 10개 폴더를 모두 생성
    for class_name in PLANTVILLAGE_CLASSES:
        (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)


def copy_taiwan_images():
    if not RAW_TAIWAN_DIR.exists():
        print(f"[ERROR] Taiwan raw dataset folder does not exist: {RAW_TAIWAN_DIR}")
        print("Check your folder path.")
        return

    counts = defaultdict(int)
    skipped_folders = set()
    total_copied = 0

    for split in SOURCE_SPLITS:
        split_dir = RAW_TAIWAN_DIR / split

        if not split_dir.exists():
            print(f"[SKIP] Split folder does not exist: {split_dir}")
            continue

        print(f"\nProcessing split: {split}")

        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir():
                continue

            taiwan_class_name = class_folder.name
            target_class_name = CLASS_MAPPING.get(taiwan_class_name)

            if target_class_name is None:
                skipped_folders.add(taiwan_class_name)
                continue

            target_dir = OUT_DIR / target_class_name

            image_files = [
                p for p in class_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]

            for idx, src_path in enumerate(image_files):
                safe_class_name = taiwan_class_name.replace(" ", "_")
                dst_name = f"taiwan_{split}_{safe_class_name}_{idx:05d}{src_path.suffix.lower()}"
                dst_path = target_dir / dst_name

                shutil.copy2(src_path, dst_path)

                counts[target_class_name] += 1
                total_copied += 1

    print("\n" + "=" * 60)
    print("Taiwan external test dataset created!")
    print(f"Output folder: {OUT_DIR}")
    print("=" * 60)

    print("\nCopied image counts:")
    for class_name in PLANTVILLAGE_CLASSES:
        print(f"{class_name}: {counts[class_name]}")

    print(f"\nTotal copied images: {total_copied}")

    if skipped_folders:
        print("\nSkipped Taiwan folders because they do not match PlantVillage classes:")
        for folder_name in sorted(skipped_folders):
            print(f"- {folder_name}")


def main():
    reset_output_dir()
    copy_taiwan_images()


if __name__ == "__main__":
    main()