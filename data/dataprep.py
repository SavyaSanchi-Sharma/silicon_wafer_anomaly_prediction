import json
import os
import cv2
import random
import uuid
from collections import defaultdict

def extract_clean_patches(
    coco,
    image_dir,
    output_dir,
    patch_size,
    margin,
    max_patches_per_image,
    max_defect_ratio=0.25
):
    os.makedirs(output_dir, exist_ok=True)

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann["bbox"])

    saved = 0
    processed = 0

    for img in coco["images"]:
        img_id = img["id"]
        path = os.path.join(image_dir, img["file_name"])

        if not os.path.exists(path):
            continue

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape
        image_area = h * w

        bboxes = anns_by_image.get(img_id, [])
        defect_area = sum(bw * bh for (_, _, bw, bh) in bboxes)

        if defect_area / image_area > max_defect_ratio:
            continue

        forbidden = []
        for x, y, bw, bh in bboxes:
            forbidden.append((
                max(0, int(x - margin)),
                max(0, int(y - margin)),
                min(w, int(x + bw + margin)),
                min(h, int(y + bh + margin))
            ))

        def is_clean(x1, y1, x2, y2):
            for fx1, fy1, fx2, fy2 in forbidden:
                if not (x2 <= fx1 or x1 >= fx2 or y2 <= fy1 or y1 >= fy2):
                    return False
            return True

        found = 0
        attempts = 0

        while found < max_patches_per_image and attempts < 80:
            attempts += 1

            x1 = random.randint(0, w - patch_size)
            y1 = random.randint(0, h - patch_size)
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            if not is_clean(x1, y1, x2, y2):
                continue

            patch = image[y1:y2, x1:x2]
            if patch.shape != (patch_size, patch_size):
                continue

            if patch.std() > 35:
                continue

            name = f"clean_{uuid.uuid4().hex[:10]}.png"
            cv2.imwrite(os.path.join(output_dir, name), patch)

            found += 1
            saved += 1

        processed += 1

    print(f"{output_dir}: processed={processed}, saved={saved}")

def run_dataset(
    annot_path,
    image_dir,
    output_dir,
    patch_size
):
    with open(annot_path, "r") as f:
        coco = json.load(f)

    margin = patch_size // 5
    max_patches = 1

    extract_clean_patches(
        coco=coco,
        image_dir=image_dir,
        output_dir=output_dir,
        patch_size=patch_size,
        margin=margin,
        max_patches_per_image=max_patches
    )

# DATASET 1 (Wafer Defect v3 – mixed resolutions)
run_dataset(
    annot_path="valid/_annotations.coco.json",
    image_dir="valid",
    output_dir="clean_valid",
    patch_size=128
)

# DATASET 2 (Bridge / Particle / Scratch – 640x640)
run_dataset(
    annot_path="temp/wafer.v2-2024-05-25-1-27am.coco/valid/_annotations.coco.json",
    image_dir="temp/wafer.v2-2024-05-25-1-27am.coco/valid",
    output_dir="clean_valid",
    patch_size=96
)
