from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class WaferDataset(Dataset):
    def __init__(self, root, split):
        self.samples = []
        self.class_to_idx = {}
        classes = sorted([d.name for d in (root / split).iterdir() if d.is_dir()])
        for i, cls in enumerate(classes):
            self.class_to_idx[cls] = i
            for img in (root / split / cls).iterdir():
                self.samples.append((img, i))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label


def compute_class_weights(dataset):
    counts = torch.zeros(len(dataset.class_to_idx))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / torch.log1p(counts)
    return weights
