import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from config import *
from dataset import WaferDataset
from model import WaferNet
from utils import EarlyStopping, MetricsTracker, ModelCheckpoint, set_seed
from augmentations import get_train_augmentation, get_validation_augmentation


class ClassificationOnlyModel(nn.Module):
    
    def __init__(self, num_classes=8):
        super().__init__()
        
        from model import SEBlock, DWBlock
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = nn.Sequential(
            DWBlock(32, 64, stride=2, use_residual=USE_RESIDUAL, use_se=USE_SE_BLOCKS),
            DWBlock(64, 128, stride=2, use_residual=USE_RESIDUAL, use_se=USE_SE_BLOCKS),
            DWBlock(128, 256, stride=2, use_residual=USE_RESIDUAL, use_se=USE_SE_BLOCKS),
            DWBlock(256, 256, stride=1, use_residual=USE_RESIDUAL, use_se=USE_SE_BLOCKS),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DROPOUT_RATE) if USE_DROPOUT else nn.Identity()
        self.fc = nn.Linear(256, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def focal_loss(logits, targets, gamma=2.0, alpha=None):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def train_epoch(model, dataloader, optimizer, scaler, class_weights, epoch):
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (img, is_def, cls) in enumerate(pbar):
        img = img.to(DEVICE)
        cls = cls.to(DEVICE)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(img)
            loss = focal_loss(logits, cls, gamma=FOCAL_GAMMA, alpha=class_weights)
        
        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        _, predicted = torch.max(logits, 1)
        correct += (predicted == cls).sum().item()
        total += cls.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.3f}', 
                         'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(dataloader), 100 * correct / total


@torch.no_grad()
def validate(model, dataloader, class_weights):
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc="Validation")
    
    for img, is_def, cls in pbar:
        img = img.to(DEVICE)
        cls = cls.to(DEVICE)
        
        logits = model(img)
        loss = focal_loss(logits, cls, gamma=FOCAL_GAMMA, alpha=class_weights)
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == cls).sum().item()
        total += cls.size(0)
        
        pbar.set_postfix({'loss': f'{total_loss/(len(pbar.iterable)):.3f}',
                         'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(dataloader), 100 * correct / total


def main():
    set_seed(SEED)
    
    print("="*80)
    print("CLASSIFICATION-ONLY TRAINING")
    print("="*80)
    print("\nRemoved detection head - focusing purely on classification")
    print("Expected: 70-80% accuracy (vs 64% with multi-task)\n")
    
    print("Loading datasets...")
    train_aug = get_train_augmentation() if USE_AUGMENTATION else get_validation_augmentation()
    val_aug = get_validation_augmentation()
    
    train_ds = WaferDataset(TRAIN_DIR, transform=train_aug)
    val_ds = WaferDataset(VAL_DIR, transform=val_aug)
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    
    train_dist = train_ds.get_class_distribution()
    class_counts = [train_dist[i] for i in range(NUM_CLASSES)]
    class_weights = calculate_class_weights(class_counts, method="inverse_freq")
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    model = ClassificationOnlyModel(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    checkpoint = ModelCheckpoint(CHECKPOINT_DIR, mode='min')
    metrics_tracker = MetricsTracker(LOG_DIR)
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80 + "\n")
    
    for epoch in range(1, EPOCHS_TOTAL + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, 
                                            class_weights, epoch)
        val_loss, val_acc = validate(model, val_loader, class_weights)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{EPOCHS_TOTAL}:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        metrics_tracker.update({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        if checkpoint.save(val_loss, checkpoint_data):
            print(f"✓ Saved checkpoint (val_acc={val_acc:.2f}%)")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break
    
    metrics_tracker.save_csv()
    metrics_tracker.plot_metrics(save_path=Path(RESULTS_DIR) / 'training_curves_classification.png')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\n✓ Best model saved to: {CHECKPOINT_DIR}/best_model.pth")
    print(f"✓ Metrics saved to: {LOG_DIR}")
    print(f"✓ Use evaluate.py to test on test set")


if __name__ == "__main__":
    main()
