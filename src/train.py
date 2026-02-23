import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import amp
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from config import *
from dataset import WaferDataset
from model import create_model
from utils import EarlyStopping, ModelCheckpoint, get_device_info
from augmentations import get_training_augmentation, get_validation_augmentation

class AdaptiveFocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.2, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, class_weights=None):

        num_classes = logits.size(1)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets_onehot = F.one_hot(targets, num_classes).float()

        if self.label_smoothing > 0:
            targets_onehot = (
                (1 - self.label_smoothing) * targets_onehot
                + self.label_smoothing / num_classes
            )

        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1 - pt).pow(self.gamma)

        ce = -(targets_onehot * log_probs).sum(dim=1)

        return (focal_weight * ce).mean()



def get_balanced_sampler(dataset, class_weights):
    sample_weights = [
        class_weights[label].item()
        for _, label in dataset.samples
    ]

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )



def train_epoch(model, loader, optimizer, scaler,
                criterion_cls, epoch):

    model.train()

    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

    for img, is_def, cls in pbar:

        img = img.to(DEVICE)
        is_def = is_def.float().to(DEVICE)
        cls = cls.to(DEVICE)

        with amp.autocast("cuda", enabled=USE_AMP):

            det_logit, cls_logit = model(img)

            loss_det = F.binary_cross_entropy_with_logits(
                det_logit.squeeze(1), is_def
            )

            mask = is_def == 1

            if mask.sum() > 0:
                loss_cls = criterion_cls(
                    cls_logit[mask],
                    cls[mask],
                    class_weights=None,
                )
            else:
                loss_cls = torch.tensor(0.0, device=DEVICE)

            det_scale = 0.5 if epoch > 5 else 1.0

            loss = det_scale * loss_det + LAMBDA_CLS * loss_cls

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion_cls):

    model.eval()

    preds, labels = [], []

    clean_tensor = torch.tensor(CLEAN_CLASS_ID, device=DEVICE)

    for img, is_def, cls in loader:

        img = img.to(DEVICE)
        cls = cls.to(DEVICE)

        det_logit, cls_logit = model(img)

        det_prob = torch.sigmoid(det_logit).view(-1)
        cls_pred = torch.argmax(cls_logit, dim=1)

        final = torch.where(
            det_prob < DETECTION_THRESHOLD,
            clean_tensor,
            cls_pred,
        )

        preds.extend(final.cpu().numpy())
        labels.extend(cls.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    return f1_score(labels, preds, average="macro")


def train_model():

    set_seed(SEED)

    print(get_config_summary())
    print(get_device_info())

    train_ds = WaferDataset(TRAIN_DIR, get_training_augmentation())
    val_ds = WaferDataset(VAL_DIR, get_validation_augmentation())

    class_weights = CLASS_WEIGHTS  # compute once

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=get_balanced_sampler(train_ds, class_weights),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = create_model(None).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    scaler = amp.GradScaler(enabled=USE_AMP)

    criterion_cls = AdaptiveFocalLoss(
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
    )

    checkpoint = ModelCheckpoint(CHECKPOINT_DIR, metric="val_f1", mode="max")
    early_stop = EarlyStopping(EARLY_STOPPING_PATIENCE, mode="max")

    warmup_epochs = 3

    for epoch in range(EPOCHS_TOTAL):

        if epoch < warmup_epochs:
            lr = LR * (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = lr

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler,
            criterion_cls, epoch
        )

        val_f1 = validate(model, val_loader, criterion_cls)

        if epoch >= warmup_epochs:
            scheduler.step(train_loss)

        checkpoint.save(model, optimizer, epoch, {"val_f1": val_f1})

        print(f"Epoch {epoch+1} | Loss {train_loss:.4f} | F1 {val_f1:.4f}")

        if early_stop(val_f1):
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    train_model()
