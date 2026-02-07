from pathlib import Path
from collections import Counter

ROOT = Path("dataset_preprocessed")

for split in ["train", "val", "test"]:
    counter = Counter()
    for cls in (ROOT / split).iterdir():
        if cls.is_dir():
            counter[cls.name] = len(list(cls.iterdir()))
    print(split, dict(counter))
