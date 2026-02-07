import cv2
from pathlib import Path

INPUT_ROOT = Path("dataset")
OUTPUT_ROOT = Path("dataset_preprocessed")

IMG_SIZE = 128

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in INPUT_ROOT.joinpath(split).iterdir():
            if cls.is_dir():
                (OUTPUT_ROOT / split / cls.name).mkdir(parents=True, exist_ok=True)

def preprocess_image(src, dst):
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    cv2.imwrite(str(dst), img)

def process_split(split):
    for cls_dir in (INPUT_ROOT / split).iterdir():
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            out_path = OUTPUT_ROOT / split / cls_dir.name / img_path.name
            preprocess_image(img_path, out_path)

def main():
    ensure_dirs()
    for split in ["train", "val", "test"]:
        process_split(split)

if __name__ == "__main__":
    main()