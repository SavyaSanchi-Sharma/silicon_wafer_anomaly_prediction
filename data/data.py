import json
import shutil
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path(".")
OUTPUT_ROOT = Path("dataset")

FINAL_CLASSES = [
    "clean",
    "scratch",
    "particle",
    "bridge",
    "contamination",
    "coating_defect",
    "etch_defect",
    "other",
]

D1_CLASS_MAP = {
    "SCRATCH": "scratch",
    "PARTICLE": "particle",
    "PIQ PARTICLE": "particle",
    "BLOCK ETCH": "etch_defect",
    "COATING BAD": "coating_defect",
    "PO CONTAMINATION": "contamination",
    "SEZ BURNT": "other",
}

D2_CLASS_MAP = {
    "bridge": "bridge",
    "particle": "particle",
    "scratch": "scratch",
}


def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in FINAL_CLASSES:
            (OUTPUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)


def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


def build_index(coco):
    img_id_to_name = {i["id"]: i["file_name"] for i in coco["images"]}
    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann["category_id"])
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    return img_id_to_name, img_id_to_anns, cat_id_to_name


def assign_label(cat_ids, cat_id_to_name, class_map):
    mapped = set()
    for cid in cat_ids:
        name = cat_id_to_name[cid]
        if name in class_map:
            mapped.add(class_map[name])
    if not mapped:
        return None
    priority = [
        "bridge",
        "scratch",
        "particle",
        "etch_defect",
        "coating_defect",
        "contamination",
        "other",
    ]
    for p in priority:
        if p in mapped:
            return p
    return mapped.pop()


def process_defect_dataset(name, class_map):
    for split in ["train", "valid", "test"]:
        coco_path = DATA_ROOT / name / split / "_annotations.coco.json"
        img_dir = DATA_ROOT / name / split
        if not coco_path.exists():
            continue
        coco = load_coco(coco_path)
        img_id_to_name, img_id_to_anns, cat_id_to_name = build_index(coco)
        out_split = split.replace("valid", "val")
        for img_id, fname in img_id_to_name.items():
            anns = img_id_to_anns.get(img_id, [])
            label = assign_label(anns, cat_id_to_name, class_map)
            if label is None:
                continue
            src = img_dir / fname
            dst = OUTPUT_ROOT / out_split / label / fname
            if src.exists():
                shutil.copy2(src, dst)


def process_clean():
    for split in ["clean_train", "clean_valid", "clean_test"]:
        out_split = split.replace("clean_", "").replace("valid", "val")
        src_dir = DATA_ROOT / "clean" / split
        if not src_dir.exists():
            continue
        for img in src_dir.iterdir():
            if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dst = OUTPUT_ROOT / out_split / "clean" / img.name
                shutil.copy2(img, dst)


def main():
    ensure_dirs()
    process_clean()
    process_defect_dataset("d1", D1_CLASS_MAP)
    process_defect_dataset("d2", D2_CLASS_MAP)


if __name__ == "__main__":
    main()
