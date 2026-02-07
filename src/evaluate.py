"""
Comprehensive evaluation script for Silicon Wafer Anomaly Detection model.

Features:
- Detailed per-class metrics
- Confusion matrix visualization
- ROC curves
- Statistical analysis
- Bootstrap confidence intervals
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from config import *
from dataset import WaferDataset
from model import create_model
from utils import (calculate_metrics, plot_confusion_matrix, plot_roc_curves,
                  save_metrics_report)
from augmentations import get_validation_augmentation, get_test_time_augmentation


@torch.no_grad()
def evaluate_model(model, dataloader, use_tta=False, tta_transforms=None):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    print("Evaluating model...")
    pbar = tqdm(dataloader, desc="Inference")
    for img, is_def, cls in pbar:
        img = img.to(DEVICE)
        cls_labels = cls.numpy()
        if use_tta and tta_transforms:
            tta_det_probs = []
            tta_cls_probs = []
            for transform in tta_transforms:
                if transform is None:
                    img_aug = img
                else:
                    img_aug_list = []
                    for i in range(img.shape[0]):
                        img_np = img[i, 0].cpu().numpy()
                        if transform:
                            img_aug_np = transform(image=img_np)['image']
                        else:
                            img_aug_np = img_np
                        img_aug_list.append(torch.from_numpy(img_aug_np).unsqueeze(0))
                    img_aug = torch.stack(img_aug_list).to(DEVICE)
                det_logit, cls_logit = model(img_aug)
                det_prob = torch.sigmoid(det_logit).cpu()
                cls_prob = F.softmax(cls_logit, dim=1).cpu()
                tta_det_probs.append(det_prob)
                tta_cls_probs.append(cls_prob)
            det_prob = torch.stack(tta_det_probs).mean(dim=0).view(-1).numpy()
            cls_prob = torch.stack(tta_cls_probs).mean(dim=0).numpy()
        else:
            det_logit, cls_logit = model(img)
            det_prob = torch.sigmoid(det_logit).view(-1).cpu().numpy()
            cls_prob = F.softmax(cls_logit, dim=1).cpu().numpy()
        cls_pred = np.argmax(cls_prob, axis=1)
        final_preds = []
        for i in range(len(det_prob)):
            if det_prob[i] < DETECTION_THRESHOLD:
                final_preds.append(CLEAN_CLASS_ID)
            else:
                final_preds.append(cls_pred[i])
        all_labels.extend(cls_labels)
        all_preds.extend(final_preds)
        all_probs.extend(cls_prob)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_iterations=1000, confidence=0.95):
    n_samples = len(y_true)
    bootstrap_scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        score = metric_fn(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    mean_score = np.mean(bootstrap_scores)
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)
    return mean_score, lower_bound, upper_bound


def main():
    set_seed(SEED)
    print("="*80)
    print("SILICON WAFER ANOMALY DETECTION - EVALUATION")
    print("="*80 + "\n")
    print("Loading test dataset...")
    test_aug = get_validation_augmentation()
    test_ds = WaferDataset(TEST_DIR, transform=test_aug)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Test samples: {len(test_ds)}")
    test_dist = test_ds.get_class_distribution()
    print("\nClass distribution (test):")
    for class_idx, count in sorted(test_dist.items()):
        class_name = IDX_TO_CLASS[class_idx]
        print(f"  {class_name:20s}: {count:4d} samples")
    print("\nLoading model...")
    model = create_model(None).to(DEVICE)
    checkpoint_path = Path(CHECKPOINT_DIR) / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train the model first using train.py")
        return
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print("\n" + "-"*80)
    print("Standard Evaluation (No TTA)")
    print("-"*80)
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, use_tta=False)
    metrics = calculate_metrics(
        y_true, y_pred, y_prob,
        class_names=CLASSES,
        average='macro'
    )
    print("\n📊 Overall Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score:  {metrics['macro_f1']:.4f}")
    if metrics.get('roc_auc'):
        print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print("\n📋 Per-Class Metrics:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*72)
    for i, class_name in enumerate(CLASSES):
        prec = metrics['per_class']['precision'][i]
        rec = metrics['per_class']['recall'][i]
        f1 = metrics['per_class']['f1'][i]
        sup = metrics['per_class']['support'][i]
        print(f"{class_name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {sup:>10d}")
    print("\n🔬 Statistical Analysis (Bootstrap 95% CI):")
    def accuracy_fn(yt, yp):
        return (yt == yp).mean()
    acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(
        y_true, y_pred, accuracy_fn, n_iterations=1000
    )
    print(f"  Accuracy: {acc_mean:.4f} [{acc_lower:.4f}, {acc_upper:.4f}]")
    print("\n📈 Generating visualizations...")
    cm_path = Path(RESULTS_DIR) / 'confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, CLASSES, save_path=cm_path, normalize=True)
    roc_path = Path(RESULTS_DIR) / 'roc_curves.png'
    plot_roc_curves(y_true, y_prob, CLASSES, save_path=roc_path)
    report_path = Path(RESULTS_DIR) / 'evaluation_report.txt'
    save_metrics_report(metrics, CLASSES, report_path)
    USE_TTA_EVAL = False
    if USE_TTA_EVAL:
        print("\n" + "-"*80)
        print("Test-Time Augmentation (TTA) Evaluation")
        print("-"*80)
        tta_transforms = get_test_time_augmentation(n_augmentations=5)
        y_true_tta, y_pred_tta, y_prob_tta = evaluate_model(
            model, test_loader, use_tta=True, tta_transforms=tta_transforms
        )
        metrics_tta = calculate_metrics(
            y_true_tta, y_pred_tta, y_prob_tta,
            class_names=CLASSES,
            average='macro'
        )
        print("\n📊 TTA Metrics:")
        print(f"  Accuracy:        {metrics_tta['accuracy']:.4f} (Δ = {metrics_tta['accuracy'] - metrics['accuracy']:+.4f})")
        print(f"  Macro F1-Score:  {metrics_tta['macro_f1']:.4f} (Δ = {metrics_tta['macro_f1'] - metrics['macro_f1']:+.4f})")
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n✓ Confusion matrix saved to: {cm_path}")
    print(f"✓ ROC curves saved to: {roc_path}")
    print(f"✓ Detailed report saved to: {report_path}")
    print("\n🎯 Final Performance Summary:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Macro F1-Score:   {metrics['macro_f1']:.2%}")
    target_accuracy = 0.85
    target_f1 = 0.75
    if metrics['accuracy'] >= target_accuracy and metrics['macro_f1'] >= target_f1:
        print(f"\n  ✅ Model meets performance targets!")
        print(f"     Accuracy ≥ {target_accuracy:.0%}: ✓")
        print(f"     Macro F1 ≥ {target_f1:.0%}: ✓")
    else:
        print(f"\n  ⚠️  Model below performance targets:")
        if metrics['accuracy'] < target_accuracy:
            print(f"     Accuracy {metrics['accuracy']:.2%} < {target_accuracy:.0%}")
        if metrics['macro_f1'] < target_f1:
            print(f"     Macro F1 {metrics['macro_f1']:.2%} < {target_f1:.0%}")
        print(f"\n  Suggestions:")
        print(f"    - Train for more epochs")
        print(f"    - Adjust class weights")
        print(f"    - Try different augmentations")
        print(f"    - Use ensemble methods")


if __name__ == "__main__":
    main()