import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import WaferDataset, compute_class_weights
from model import get_model
from config import *
from pathlib import Path

def train():
    train_ds = WaferDataset(DATA_ROOT, "train")
    val_ds = WaferDataset(DATA_ROOT, "val")

    weights = compute_class_weights(train_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_recall = 0.0
    Path("../checkpoints").mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        recall = evaluate(model, val_loader)
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), "../checkpoints/best_model.pt")

        print(f"Epoch {epoch} | Val Macro Recall: {recall:.4f}")

def evaluate(model, loader):
    model.eval()
    correct = torch.zeros(NUM_CLASSES)
    total = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            for i in range(len(y)):
                total[y[i]] += 1
                if preds[i] == y[i]:
                    correct[y[i]] += 1

    recall = (correct / total.clamp(min=1)).mean().item()
    return recall

if __name__ == "__main__":
    train()
