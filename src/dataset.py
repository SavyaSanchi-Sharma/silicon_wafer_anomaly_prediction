import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from config import CLASS_TO_IDX, IMG_SIZE

class WaferDataset(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(root):
            raise ValueError(f"Dataset directory not found: {root}")
        
        for class_name, class_idx in CLASS_TO_IDX.items():
            cls_path = os.path.join(root, class_name)
            
            if not os.path.isdir(cls_path):
                print(f"Warning: Class directory not found: {cls_path}")
                continue
            
            for filename in os.listdir(cls_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(cls_path, filename)
                    
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root}")
        
        print(f"Loaded {len(self.samples)} samples from {root}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        img = img.astype(np.float32) / 255.0
        
        img = np.expand_dims(img, 0)
        
        img = torch.from_numpy(img)
        
        is_defective = 0 if label == 0 else 1
        
        return img, torch.tensor(is_defective), torch.tensor(label)
    
    def get_class_distribution(self):
        from collections import Counter
        class_counts = Counter([label for _, label in self.samples])
        return dict(sorted(class_counts.items()))