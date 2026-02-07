"""
Comprehensive training script for Silicon Wafer Anomaly Detection.

Features:
- Two-stage training (detection → full training)
- Validation monitoring with early stopping
- Mixed precision training (AMP)
- Gradient accumulation
- Class-weighted focal loss
- Learning rate scheduling
- Comprehensive logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config import *
from dataset import WaferDataset
from model import create_model
from utils import (EarlyStopping, MetricsTracker, ModelCheckpoint, 
                  calculate_metrics, get_device_info)
from augmentations import get_training_augmentation, get_validation_augmentation

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=alpha)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma) * ce_loss
    return focal_loss.mean()

def get_balanced_sampler(dataset):
    class_counts = dataset.get_class_distribution()
    class_weights = {}
    total = sum(class_counts.values())
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total / (NUM_CLASSES * count)
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def train_epoch(model, dataloader, optimizer, scaler, epoch, stage='full', 
                class_weights=None, use_amp=True, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    total_det_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{stage}]")
    for batch_idx, (img, is_def, cls) in enumerate(pbar):
        img = img.to(DEVICE)
        is_def = is_def.float().to(DEVICE)
        cls = cls.to(DEVICE)
        with autocast(enabled=use_amp):
            det_logit, cls_logit = model(img)
            loss_det = F.binary_cross_entropy_with_logits(
                det_logit.squeeze(1), is_def
            )
            if stage == 'full':
                mask = is_def == 1
                if mask.sum() > 0:
                    loss_cls = focal_loss(
                        cls_logit[mask], 
                        cls[mask],
                        gamma=FOCAL_GAMMA,
                        alpha=class_weights
                    )
                else:
                    loss_cls = torch.tensor(0.0, device=DEVICE)
            else:
                loss_cls = torch.tensor(0.0, device=DEVICE)
            if stage == 'detection':
                loss = loss_det
            else:
                loss = LAMBDA_DET * loss_det + LAMBDA_CLS * loss_cls
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss
        total_det_loss += loss_det.item()
        total_cls_loss += loss_cls.item()
        num_batches += 1
        pbar.set_postfix({
            'loss': batch_loss,
            'det': loss_det.item(),
            'cls': loss_cls.item()
        })
    return {
        'train_loss': total_loss / num_batches,
        'train_det_loss': total_det_loss / num_batches,
        'train_cls_loss': total_cls_loss / num_batches
    }

@torch.no_grad()
def validate(model, dataloader, class_weights=None):
    model.eval()
    total_loss = 0.0
    total_det_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc="Validation")
    for img, is_def, cls in pbar:
        img = img.to(DEVICE)
        is_def = is_def.float().to(DEVICE)
        cls = cls.to(DEVICE)
        det_logit, cls_logit = model(img)
        loss_det = F.binary_cross_entropy_with_logits(
            det_logit.squeeze(1), is_def
        )
        mask = is_def == 1
        if mask.sum() > 0:
            loss_cls = focal_loss(
                cls_logit[mask],
                cls[mask],
                gamma=FOCAL_GAMMA,
                alpha=class_weights
            )
        else:
            loss_cls = torch.tensor(0.0, device=DEVICE)
        loss = LAMBDA_DET * loss_det + LAMBDA_CLS * loss_cls
        total_loss += loss.item()
        total_det_loss += loss_det.item()
        total_cls_loss += loss_cls.item()
        num_batches += 1
        det_prob = torch.sigmoid(det_logit).view(-1).cpu().numpy()
        cls_pred = torch.argmax(cls_logit, dim=1).cpu().numpy()
        for i in range(len(det_prob)):
            if det_prob[i] < DETECTION_THRESHOLD:
                all_preds.append(CLEAN_CLASS_ID)
            else:
                all_preds.append(cls_pred[i])
        all_labels.extend(cls.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    return {
        'val_loss': total_loss / num_batches,
        'val_det_loss': total_det_loss / num_batches,
        'val_cls_loss': total_cls_loss / num_batches,
        'val_accuracy': accuracy
    }

def train_model():
    set_seed(SEED)
    print(get_config_summary())
    print("\nDevice Information:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()
    print("Loading datasets...")
    train_aug = get_training_augmentation()
    val_aug = get_validation_augmentation()
    train_ds = WaferDataset(TRAIN_DIR, transform=train_aug)
    val_ds = WaferDataset(VAL_DIR, transform=val_aug)
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    train_dist = train_ds.get_class_distribution()
    print("\nClass distribution (training):")
    for class_idx, count in sorted(train_dist.items()):
        class_name = IDX_TO_CLASS[class_idx]
        print(f"  {class_name:20s}: {count:4d} samples")
    class_weights = calculate_class_weights(train_dist, method="inverse_freq")
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           persistent_workers=True if NUM_WORKERS > 0 else False)
    print("\nInitializing model...")
    model = create_model(None).to(DEVICE)
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Estimated FLOPs: {model.get_flops()}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    if LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LR_MIN
        )
    scaler = GradScaler(enabled=USE_AMP)
    metrics_tracker = MetricsTracker(LOG_DIR)
    checkpoint = ModelCheckpoint(CHECKPOINT_DIR, metric='val_loss', mode='min')
    early_stop = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min') if EARLY_STOPPING else None
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    print(f"[Stage 1] Detection-only training for {EPOCHS_DET} epochs")
    print("-" * 80)
    for param in model.classifier.parameters():
        param.requires_grad = False
    for epoch in range(EPOCHS_DET):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, epoch,
            stage='detection',
            use_amp=USE_AMP,
            grad_accum_steps=GRADIENT_ACCUMULATION_STEPS
        )
        val_metrics = validate(model, val_loader, class_weights)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS_DET}:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} "
              f"(Det: {train_metrics['train_det_loss']:.4f})")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f} "
              f"(Det: {val_metrics['val_det_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.4f})")
        print(f"  LR: {current_lr:.6f}")
        metrics_tracker.update(
            epoch,
            learning_rate=current_lr,
            **train_metrics,
            **val_metrics
        )
        if LR_SCHEDULER:
            scheduler.step(val_metrics['val_loss'])
        checkpoint.save(model, optimizer, epoch, val_metrics, 'best_detection.pth')
    print(f"\n[Stage 2] Full training for {EPOCHS_TOTAL} epochs")
    print("-" * 80)
    for param in model.classifier.parameters():
        param.requires_grad = True
    for epoch in range(EPOCHS_TOTAL):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, epoch,
            stage='full',
            class_weights=class_weights,
            use_amp=USE_AMP,
            grad_accum_steps=GRADIENT_ACCUMULATION_STEPS
        )
        val_metrics = validate(model, val_loader, class_weights)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS_TOTAL}:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} "
              f"(Det: {train_metrics['train_det_loss']:.4f}, "
              f"Cls: {train_metrics['train_cls_loss']:.4f})")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f} "
              f"(Det: {val_metrics['val_det_loss']:.4f}, "
              f"Cls: {val_metrics['val_cls_loss']:.4f}, "
              f"Acc: {val_metrics['val_accuracy']:.4f})")
        print(f"  LR: {current_lr:.6f}")
        metrics_tracker.update(
            epoch + EPOCHS_DET,
            learning_rate=current_lr,
            **train_metrics,
            **val_metrics
        )
        if LR_SCHEDULER:
            scheduler.step(val_metrics['val_loss'])
        is_best = checkpoint.save(model, optimizer, epoch + EPOCHS_DET, val_metrics)
        if early_stop and early_stop(val_metrics['val_loss']):
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"   No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    metrics_tracker.save_csv()
    metrics_tracker.plot_metrics(save_path=Path(RESULTS_DIR) / 'training_curves.png')
    print(f"\n✓ Metrics saved to {LOG_DIR}")
    print(f"✓ Best model saved to {CHECKPOINT_DIR}/best_model.pth")
    print(f"✓ Training curves saved to {RESULTS_DIR}/training_curves.png")
    return model

if __name__ == "__main__":
    train_model()