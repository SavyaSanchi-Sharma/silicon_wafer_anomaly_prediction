import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import *
from dataset import WaferDataset
from model import create_model
from utils import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    save_metrics_report
)
from augmentations import get_validation_augmentation

# ============================================================
# SIMPLE GPU-SAFE TTA AUGMENTATIONS
# ============================================================

def apply_tta_batch(img, mode):
    if mode == "none":
        return img
    elif mode == "hflip":
        return torch.flip(img, dims=[3])
    elif mode == "vflip":
        return torch.flip(img, dims=[2])
    elif mode == "rot90":
        return torch.rot90(img, k=1, dims=[2, 3])
    else:
        raise ValueError(mode)


TTA_MODES = [
    "none",
    "hflip",
    "vflip",
    "rot90",
]


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def evaluate_model(model, dataloader, threshold, use_tta=False):

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for img, _, cls in tqdm(dataloader, desc="Inference"):

        img = img.to(DEVICE)
        labels = cls.numpy()

        # ---------- Forward ----------
        if not use_tta:

            det_logit, cls_logit = model(img)

            det_prob = torch.sigmoid(det_logit).view(-1)
            cls_prob = F.softmax(cls_logit, dim=1)

        else:
            det_accum = []
            cls_accum = []

            for mode in TTA_MODES:
                img_aug = apply_tta_batch(img, mode)

                d, c = model(img_aug)

                det_accum.append(torch.sigmoid(d))
                cls_accum.append(F.softmax(c, dim=1))

            det_prob = torch.stack(det_accum).mean(0).view(-1)
            cls_prob = torch.stack(cls_accum).mean(0)

        det_prob = det_prob.cpu().numpy()
        cls_prob = cls_prob.cpu().numpy()

        # ---------- Hierarchical prediction ----------
        cls_pred = np.argmax(cls_prob, axis=1)

        final_preds = []
        combined_probs = cls_prob.copy()

        for i in range(len(det_prob)):

            if det_prob[i] < threshold:
                final_preds.append(CLEAN_CLASS_ID)
            else:
                final_preds.append(cls_pred[i])

            combined_probs[i, CLEAN_CLASS_ID] = 1 - det_prob[i]

        all_labels.extend(labels)
        all_preds.extend(final_preds)
        all_probs.extend(combined_probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ============================================================
# Threshold Calibration
# ============================================================

def find_best_threshold(model, val_loader):

    print("\nCalibrating detection threshold...")

    thresholds = np.linspace(0.1, 0.9, 41)

    best_t = 0.5
    best_f1 = 0

    for t in thresholds:
        y_true, y_pred, _ = evaluate_model(
            model,
            val_loader,
            threshold=t,
            use_tta=False,
        )

        f1 = f1_score(y_true, y_pred, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"Best threshold = {best_t:.3f} (F1={best_f1:.4f})")
    return best_t


# ============================================================
# Bootstrap CI
# ============================================================

def bootstrap_ci(y_true, y_pred, metric_fn, n_iter=1000):

    scores = []
    n = len(y_true)

    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    scores = np.array(scores)
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 80)
    print("WAFER DEFECT MODEL EVALUATION")
    print("=" * 80)

    val_ds = WaferDataset(VAL_DIR, transform=get_validation_augmentation())
    test_ds = WaferDataset(TEST_DIR, transform=get_validation_augmentation())

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(None).to(DEVICE)

    ckpt = torch.load(
        Path(CHECKPOINT_DIR) / "best_model.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    # ---------- threshold calibration ----------
    best_threshold = find_best_threshold(model, val_loader)

    # ---------- evaluation WITH TTA ----------
    y_true, y_pred, y_prob = evaluate_model(
        model,
        test_loader,
        threshold=best_threshold,
        use_tta=True,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
        class_names=CLASSES,
        average="macro",
    )

    print("\nRESULTS")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    mean, low, high = bootstrap_ci(
        y_true,
        y_pred,
        lambda a, b: f1_score(a, b, average="macro"),
    )

    print(f"Macro F1 (95% CI): {mean:.4f} [{low:.4f}, {high:.4f}]")

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(
        y_true,
        y_pred,
        CLASSES,
        save_path=results_dir / "confusion_matrix.png",
    )

    plot_roc_curves(
        y_true,
        y_prob,
        CLASSES,
        save_path=results_dir / "roc_curves.png",
    )

    save_metrics_report(
        metrics,
        CLASSES,
        results_dir / "evaluation_report.txt",
    )

    print("\nEvaluation complete")
    print(f"Results saved → {results_dir}")


if __name__ == "__main__":
    main()
