
# Silicon Wafer Anomaly Detection for Edge Devices

## Overview

This project implements an edge-optimized deep learning system for automated silicon wafer inspection. The goal is reliable defect detection and classification under limited compute and memory constraints.

The system is designed for deployment on resource-constrained environments such as industrial edge GPUs or embedded inspection systems.

Unlike large classification models, this project uses a lightweight hierarchical architecture that separates defect detection from defect classification to improve stability under class imbalance.

The dataset used in this project is self-collected and manually structured.

---

## Objectives

* Detect defective wafers reliably in real-time environments
* Classify defect types with minimal compute cost
* Maintain stable training under strong class imbalance
* Enable deployment on low-VRAM GPUs and edge hardware
* Provide reproducible training and evaluation pipelines

---

## System Design

The model performs inference in two stages:

1. **Detection**

   * Determines whether a wafer is clean or defective.

2. **Classification**

   * Predicts defect category only when a defect is detected.

This hierarchical design reduces false classifications and improves robustness for edge deployment.

---

## Project Structure

```
silicon_wafer_anomaly_prediction/
│
├── data/
│   ├── dataset.zip              # self-created dataset archive
│
├── src/
│   ├── augmentations.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── export_model.py
│   ├── model.py
│   ├── rebalance_dataset.py
│   ├── train.py
│   └── utils.py
│
├── checkpoints/
├── logs/
├── results/
│
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset is self-generated and organized into class folders.

### Classes

```
clean
scratch
particle
bridge
contamination
coating_defect
etch_defect
other
```

### Preprocessing

Each image undergoes:

* grayscale loading
* resizing to fixed resolution
* CLAHE contrast enhancement
* normalization to [0, 1]

Class imbalance is handled dynamically during training rather than through static oversampling.

---

## Model Architecture

The network (`WaferNet`) is a lightweight convolutional model built specifically for edge inference.

### Backbone Components

* Depthwise separable convolutions
* Residual connections (optional)
* Squeeze-and-Excitation attention
* Global Average or GeM pooling

### Dual Output Heads

**Detector Head**

* Binary defect prediction

**Classifier Head**

* Multi-class defect classification

Both heads share feature extraction layers to minimize computation.

---

## Training Pipeline

Training is designed for stability on small and imbalanced datasets.

### Key Techniques

* Adaptive focal loss with label smoothing
* Dynamic class weight computation per epoch
* Conditional weighted sampling
* Mixed precision training
* Gradient accumulation for low VRAM systems
* Learning rate warmup
* Early stopping

### Optimization

* Optimizer: AdamW
* Scheduler: ReduceLROnPlateau (loss-based)
* Gradient clipping enabled

---

## Edge Optimization Considerations

The model prioritizes:

* low parameter count
* reduced FLOPs
* stable gradients
* predictable inference latency

Designed to run on GPUs with limited VRAM (e.g., RTX 3050 4GB).

---

## Training

Run training from the source directory:

```
python train.py
```

Training automatically:

* builds datasets
* computes imbalance statistics
* applies augmentations
* saves best checkpoints

Output model:

```
checkpoints/best_model.pth
```

---

## Evaluation

```
python evaluate.py
```

Evaluation includes:

* automatic detection threshold calibration
* hierarchical prediction merging
* confusion matrix generation
* ROC curves
* bootstrap confidence intervals

Results are saved to:

```
results/
```

---

## Configuration

All parameters are centralized in:

```
src/config.py
```

Includes:

* hardware profile
* batch size and image size
* optimizer settings
* augmentation controls
* loss configuration
* training schedule

---

## Augmentation Strategy

Training augmentations simulate inspection variability:

* rotation
* flipping
* affine transforms
* brightness and contrast variation
* noise injection
* blur and distortion

Validation remains deterministic to ensure reliable evaluation.

---

## Outputs

```
checkpoints/   trained models
logs/          training statistics
results/       evaluation reports and plots
```

---

## Reproducibility

The pipeline fixes random seeds across:

* PyTorch
* NumPy
* Python runtime

CuDNN benchmarking is enabled for optimized execution on fixed input sizes.

---

## Dependencies

Install required packages:

```
pip install -r requirements.txt
```

Main dependencies:

* PyTorch
* Albumentations
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

---

## Intended Use

This repository demonstrates an end-to-end workflow for building an edge-deployable industrial inspection model using a custom dataset, emphasizing efficiency, robustness, and reproducibility rather than large-scale model complexity.

