import torch
import json
import numpy as np
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


HARDWARE_PROFILE = "MEDIUM"

HARDWARE_CONFIGS = {
    "batch_size": 16,
    "img_size": 160,
    "num_workers": 2,
    "use_amp": True,
    "gradient_accumulation": 2,   # ← reduced (better gradients)
}

HW_CONFIG = HARDWARE_CONFIGS

IMG_SIZE = HW_CONFIG["img_size"]
BATCH_SIZE = HW_CONFIG["batch_size"]
NUM_WORKERS = HW_CONFIG["num_workers"]



TRAIN_DIR = "../data/dataset_balanced/train"
VAL_DIR   = "../data/dataset_balanced/val"
TEST_DIR  = "../data/dataset_balanced/test"


CLASSES = [
    "clean",
    "scratch",
    "particle",
    "bridge",
    "contamination",
    "coating_defect",
    "etch_defect",
    "other",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

NUM_CLASSES = len(CLASSES)
CLEAN_CLASS_ID = CLASS_TO_IDX["clean"]


weights_path = "../data/dataset_balanced/class_weights.json"

with open(weights_path, "r") as f:
    _weights_dict = json.load(f)

CLASS_WEIGHTS = torch.tensor(
    [_weights_dict[c] for c in CLASSES],
    dtype=torch.float32
).to(DEVICE)


USE_RESIDUAL = True
USE_SE_BLOCKS = True
USE_DROPOUT = True
DROPOUT_RATE = 0.3
SE_REDUCTION = 16
USE_GEM_POOLING = False


SEED = 42

EPOCHS_DET = 10
EPOCHS_TOTAL = 500

LR = 3e-4
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
EPS = 1e-8


LR_SCHEDULER = True
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-6


EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20


USE_AMP = HW_CONFIG["use_amp"]


GRADIENT_ACCUMULATION_STEPS = HW_CONFIG["gradient_accumulation"]
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

GRAD_CLIP_NORM = 1.0


LAMBDA_DET = 1.0
LAMBDA_CLS = 0.3  


FOCAL_GAMMA = 1.2  
FOCAL_ALPHA = None

LABEL_SMOOTHING = 0.05


USE_AUGMENTATION = True

AUG_ROTATE_P = 0.5
AUG_FLIP_H_P = 0.5
AUG_FLIP_V_P = 0.5
AUG_SHIFT_SCALE_ROTATE_P = 0.5
AUG_BRIGHTNESS_CONTRAST_P = 0.3
AUG_NOISE_P = 0.1
AUG_BLUR_P = 0.1
AUG_GRID_DISTORTION_P = 0.1

AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.2
AUG_SHIFT_LIMIT = 0.1
AUG_SCALE_LIMIT = 0.1
AUG_ROTATE_LIMIT = 15


VAL_FREQUENCY = 1

DETECTION_THRESHOLD = 0.4


LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "tensorboard"), exist_ok=True)

SAVE_BEST_ONLY = True
CHECKPOINT_METRIC = "val_loss"

LOG_INTERVAL = 10


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    import random
    random.seed(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def get_config_summary():
    img_size_str = f"{IMG_SIZE}x{IMG_SIZE}"
    batch_str = f"{BATCH_SIZE} (Effective: {EFFECTIVE_BATCH_SIZE})"

    summary = f"""
╔════════════════════════════════════════════════════════════╗
║        Silicon Wafer Anomaly Detection Config             ║
╠════════════════════════════════════════════════════════════╣
║ Hardware Profile: {HARDWARE_PROFILE:<40} ║
║ Device: {DEVICE:<49} ║
║ Image Size: {img_size_str:<52} ║
║ Batch Size: {batch_str:<52} ║
║ Mixed Precision: {str(USE_AMP):<44} ║
╠════════════════════════════════════════════════════════════╣
║ Classes: {NUM_CLASSES:<52} ║
║ Detection Epochs: {EPOCHS_DET:<46} ║
║ Total Epochs: {EPOCHS_TOTAL:<49} ║
║ Learning Rate: {LR:<49.0e} ║
║ Weight Decay: {WEIGHT_DECAY:<50.0e} ║
╚════════════════════════════════════════════════════════════╝
"""
    return summary



