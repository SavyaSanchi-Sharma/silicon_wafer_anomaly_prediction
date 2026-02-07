import os
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        self.compare = np.less if mode == 'min' else np.greater
        self.delta = -min_delta if mode == 'min' else min_delta
    
    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.compare(current_value, self.best_value + self.delta):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def get_status(self):
        return {
            'counter': self.counter,
            'best_value': self.best_value,
            'patience': self.patience
        }


class MetricsTracker:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = {
            'train_loss': [],
            'train_det_loss': [],
            'train_cls_loss': [],
            'val_loss': [],
            'val_det_loss': [],
            'val_cls_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def update(self, epoch, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save_csv(self, filename='metrics.csv'):
        import pandas as pd
        
        max_len = max(len(v) for v in self.metrics.values() if isinstance(v, list))
        
        padded_metrics = {}
        for key, values in self.metrics.items():
            if isinstance(values, list):
                padded_metrics[key] = values + [None] * (max_len - len(values))
            else:
                padded_metrics[key] = values
        
        df = pd.DataFrame(padded_metrics)
        df.to_csv(self.log_dir / filename, index=False)
    
    def plot_metrics(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.metrics['train_det_loss'], label='Train Det Loss', alpha=0.7)
        axes[0, 1].plot(self.metrics['train_cls_loss'], label='Train Cls Loss', alpha=0.7)
        axes[0, 1].plot(self.metrics['val_det_loss'], label='Val Det Loss', linestyle='--')
        axes[0, 1].plot(self.metrics['val_cls_loss'], label='Val Cls Loss', linestyle='--')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Detection and Classification Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.metrics['val_accuracy'], label='Val Accuracy', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.metrics['learning_rate'], label='Learning Rate', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class ModelCheckpoint:
    def __init__(self, checkpoint_dir, metric='val_loss', mode='min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.metric = metric
        self.mode = mode
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.compare = np.less if mode == 'min' else np.greater
    
    def save(self, model, optimizer, epoch, metrics, filename='best_model.pth'):
        """Save checkpoint if metric improved."""
        current_value = metrics.get(self.metric, None)
        
        if current_value is None:
            print(f"Warning: Metric {self.metric} not found in metrics dict")
            return False
        
        if self.compare(current_value, self.best_value):
            self.best_value = current_value
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                self.metric: current_value
            }
            filepath = self.checkpoint_dir / filename
            torch.save(checkpoint, filepath)
            print(f"✓ Saved checkpoint: {filepath} ({self.metric}={current_value:.4f})")
            return True
        
        return False


def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None, average='macro'):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics[f'{average}_precision'] = precision
    metrics[f'{average}_recall'] = recall
    metrics[f'{average}_f1'] = f1
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'precision': precision_per_class.tolist(),
        'recall': recall_per_class.tolist(),
        'f1': f1_per_class.tolist(),
        'support': support_per_class.tolist()
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average=average
            )
        except:
            metrics['roc_auc'] = None
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Percentage'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class', fontsize=16, pad=20)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved ROC curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_metrics_report(metrics, class_names, save_path):
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SILICON WAFER ANOMALY DETECTION - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            f.write(f"ROC-AUC (macro): {metrics['roc_auc']:.4f}\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 80 + "\n")
        
        for i, class_name in enumerate(class_names):
            prec = metrics['per_class']['precision'][i]
            rec = metrics['per_class']['recall'][i]
            f1 = metrics['per_class']['f1'][i]
            sup = metrics['per_class']['support'][i]
            f.write(f"{class_name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {sup:>10d}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved metrics report to {save_path}")


def get_device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**3  # GB
    
    return info


if __name__ == "__main__":
    print("Testing utility functions...")
    
    early_stop = EarlyStopping(patience=3, mode='min')
    test_losses = [1.0, 0.8, 0.75, 0.76, 0.77, 0.78, 0.79]
    for epoch, loss in enumerate(test_losses):
        stop = early_stop(loss)
        print(f"Epoch {epoch}: loss={loss:.2f}, stop={stop}, status={early_stop.get_status()}")
    
    print("\nDevice Info:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
