
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Install with: pip install albumentations")

from config import *


def get_training_augmentation():
    if not ALBUMENTATIONS_AVAILABLE or not USE_AUGMENTATION:
        return None
    
    transform = A.Compose([
        A.RandomRotate90(p=AUG_ROTATE_P),
        A.HorizontalFlip(p=AUG_FLIP_H_P),
        A.VerticalFlip(p=AUG_FLIP_V_P),
        A.ShiftScaleRotate(
            shift_limit=AUG_SHIFT_LIMIT,
            scale_limit=AUG_SCALE_LIMIT,
            rotate_limit=AUG_ROTATE_LIMIT,
            border_mode=0,
            p=AUG_SHIFT_SCALE_ROTATE_P
        ),
        
        A.RandomBrightnessContrast(
            brightness_limit=AUG_BRIGHTNESS_LIMIT,
            contrast_limit=AUG_CONTRAST_LIMIT,
            p=AUG_BRIGHTNESS_CONTRAST_P
        ),
        
        A.GaussNoise(
            var_limit=(0.001, 0.01),
            p=AUG_NOISE_P
        ),
        
        A.GaussianBlur(
            blur_limit=(3, 3),
            p=AUG_BLUR_P
        ),
        
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            p=AUG_GRID_DISTORTION_P
        ),
    ])
    
    return transform


def get_validation_augmentation():
    return None


def get_test_time_augmentation(n_augmentations=5):
    if not ALBUMENTATIONS_AVAILABLE:
        return [None]
    
    tta_transforms = [
        None,  # Original image
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        A.RandomRotate90(p=1.0),
    ]
    
    return tta_transforms[:n_augmentations]


def visualize_augmentations(dataset, idx=0, n_samples=8, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    original_transform = dataset.transform
    dataset.transform = None
    img_orig, _, label = dataset[idx]
    dataset.transform = original_transform
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    axes[0].imshow(img_orig.squeeze(), cmap='gray')
    axes[0].set_title(f'Original (Class: {label.item()})')
    axes[0].axis('off')
    
    for i in range(1, n_samples):
        img_aug, _, _ = dataset[idx]
        axes[i].imshow(img_aug.squeeze(), cmap='gray')
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved augmentation visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Testing augmentation pipeline...")
    
    train_aug = get_training_augmentation()
    if train_aug:
        print("✓ Training augmentation loaded successfully")
        print(f"  Pipeline: {train_aug}")
    else:
        print("✗ No augmentation available")
    
    tta_transforms = get_test_time_augmentation()
    print(f"\n✓ TTA with {len(tta_transforms)} transforms")
