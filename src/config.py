from pathlib import Path

DATA_ROOT = Path("../data/dataset_preprocessed")

IMG_SIZE = 128
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

DEVICE = "cuda"

CLASS_NAMES = [
    "clean",
    "scratch",
    "particle",
    "bridge",
    "contamination",
    "coating_defect",
    "etch_defect",
    "other",
]
