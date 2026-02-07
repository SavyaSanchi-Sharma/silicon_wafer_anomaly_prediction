import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import WaferDataset
from model import get_model
from config import *


def test():
    ds = WaferDataset(DATA_ROOT, "test")
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("../checkpoints/best_model.pt", map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    per_class_recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    macro_recall = per_class_recall.mean()
    weighted_recall = np.average(
        per_class_recall, weights=cm.sum(axis=1)
    )

    print("\n================= OVERALL METRICS =================")
    print(f"Overall Accuracy      : {acc:.4f}")
    print(f"Macro Recall          : {macro_recall:.4f}")
    print(f"Weighted Recall       : {weighted_recall:.4f}")

    print("\n================ PER-CLASS METRICS =================")
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    print(report)

    print("\n============= MOST CONFUSED CLASS PAIRS ============")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat = cm_copy.flatten()
    top_confusions = flat.argsort()[-5:][::-1]

    for idx in top_confusions:
        i, j = divmod(idx, cm.shape[1])
        print(
            f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]} : {cm[i, j]}"
        )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="mako"
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../results/confusion_matrix.png")
    plt.close()

    metrics = {
        "accuracy": float(acc),
        "macro_recall": float(macro_recall),
        "weighted_recall": float(weighted_recall),
        "per_class_recall": dict(zip(CLASS_NAMES, per_class_recall.tolist()))
    }

    with open("../results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nArtifacts saved:")
    print(" - results/confusion_matrix.png")
    print(" - results/metrics.json")


if __name__ == "__main__":
    test()
