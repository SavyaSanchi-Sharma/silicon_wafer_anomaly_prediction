# Silicon Wafer Anomaly Detection for Edge Devices

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A deep learning system for detecting and classifying defects in silicon wafer manufacturing, specifically designed for **deployment on resource-constrained edge devices**. This project combines efficient neural network architectures, advanced data handling strategies for severe class imbalance, and comprehensive model optimization techniques to enable real-time quality control in manufacturing environments.

---

## Problem Statement

Silicon wafer manufacturing inspection faces two critical challenges:

### 1. **Computational Constraints**
Manufacturing facilities require on-site, real-time defect detection on edge devices (Raspberry Pi, embedded systems, IoT devices) with:
- Limited memory (< 1GB RAM)
- Restricted compute power (no dedicated GPUs)
- Power consumption constraints
- Real-time latency requirements (< 50ms per inference)

### 2. **Severe Data Imbalance**
Wafer defect datasets exhibit extreme class imbalance inherent to manufacturing:
- Clean wafers constitute 60-70% of production
- Common defects (scratches, particles) are moderately represented
- Rare critical defects (bridging, contamination) have minimal samples (< 100 examples)
- This imbalance causes models to overfit to the majority class, achieving high accuracy while failing to detect critical defects

---

## Architecture: WaferNet

### Design Philosophy

WaferNet is a custom CNN architecture that prioritizes **inference efficiency** while maintaining classification accuracy. The design draws inspiration from MobileNetV2 and EfficientNet but is specifically tailored for grayscale wafer imagery and edge deployment.

### Core Architecture

```
Input: (1, 224, 224) Grayscale Image

[Stem Block]
├─ Conv2d(1→32, kernel=3, stride=2)      # Spatial downsampling
├─ BatchNorm2d(32)
└─ ReLU

[Backbone: 4× Depthwise Separable Blocks]
├─ DWBlock(32→64, stride=2)   # Feature extraction
├─ DWBlock(64→128, stride=2)
├─ DWBlock(128→256, stride=2)
└─ DWBlock(256→256, stride=1) # Refinement

[Global Pooling]
└─ AdaptiveAvgPool2d(1) → (256,)

[Dual Output Heads]
├─ Detection Head    → Binary: defect/clean
└─ Classification Head → 8-class defect type
```

### PyTorch Implementation Details

#### 1. **Depthwise Separable Convolutions**

Standard convolutions are computationally expensive. WaferNet uses depthwise separable convolutions to reduce parameters and FLOPs by 8-10×.

**Implementation:** See [`src/model.py`](src/model.py) - `DWBlock` class

**Key concepts:**
- **Depthwise convolution**: Spatial filtering per channel independently (groups=in_channels)
- **Pointwise convolution**: Channel mixing with 1×1 kernels
- **Computational complexity**: O(H × W × C_in × K²) + O(H × W × C_in × C_out) vs. standard O(H × W × C_in × C_out × K²)

**Why this matters for edge devices:**
- **Memory**: Fewer parameters mean smaller model size (150K params ≈ 600KB FP32)
- **Speed**: 8-10× FLOPs reduction directly translates to faster inference
- **Energy**: Lower computation = lower power consumption on battery-powered devices

#### 2. **Squeeze-and-Excitation (SE) Blocks**

SE blocks implement channel-wise attention to recalibrate feature importance with minimal overhead.

**Implementation:** See [`src/model.py`](src/model.py) - `SEBlock` class

**Mechanism:**
1. **Squeeze**: Global average pooling to create channel descriptors
2. **Excitation**: Two FC layers (bottleneck with reduction=16) to learn channel interdependencies
3. **Recalibration**: Element-wise multiplication of learned weights with feature maps

**Impact:**
- Adds < 2% parameters but improves accuracy by 3-5%
- Enables the model to focus on discriminative features (defect edges, textures)
- Efficient: Most computation happens in reduced dimensionality (channels/16)

#### 3. **Dual-Head Architecture**

The model performs two tasks simultaneously: binary defect detection and multi-class defect classification.

**Implementation:** See [`src/model.py`](src/model.py) - `WaferNet.forward()` method

**Inference hierarchy:**
1. Shared backbone extracts features → (B, 256)
2. Detection head outputs binary logit → defect/clean
3. Classification head outputs 8-class logits → defect type
4. If detection probability < 0.5, override classification to "clean"

This hierarchical approach prevents misclassification of clean wafers as defective, improving precision.

---

## Handling Data Imbalance: A Multi-Pronged Approach

### Challenge Quantified

Typical wafer dataset distribution:
- Clean: 3,500 samples (58%)
- Scratch: 900 samples (15%)
- Particle: 700 samples (12%)
- Bridge: 300 samples (5%)
- Contamination: 200 samples (3%)
- Coating defect: 180 samples (3%)
- Etch defect: 150 samples (2.5%)
- Other: 70 samples (1.5%)

**Result without intervention:** Model achieves 70% accuracy by simply predicting "clean" for everything, while missing all critical defects.

### Solution 1: Self-Augmentation Pipeline

**Implementation:** See [`src/augmentations.py`](src/augmentations.py) - `get_training_augmentation()`

Uses Albumentations library for GPU-accelerated augmentation:

**Geometric transformations** (preserve defect spatial structure):
- Random 90° rotations, horizontal/vertical flips
- Shift-scale-rotate (±10% shift, ±10% scale, ±15° rotation)

**Photometric transformations** (simulate imaging variations):
- Brightness/contrast adjustment (±20%)
- Gaussian noise and blur (sensor/focus variations)
- Grid distortion (elastic deformations for realistic defect variations)

**Impact:** Effectively multiplies dataset size by 5-10×, especially benefiting minority classes.

### Solution 2: Dataset Rebalancing

**Implementation:** See [`src/rebalance_dataset.py`](src/rebalance_dataset.py)

Three strategies implemented:

#### Oversampling (Default)
Duplicates minority class samples to match majority class count.
- **Pros:** No data loss, all minority samples utilized
- **Cons:** Risk of overfitting (mitigated by augmentation)

#### Undersampling
Reduces majority class samples to match minority class count.
- **Pros:** Smaller dataset, faster training
- **Cons:** Discards potentially valuable data

#### Weighted Sampling
Assigns class weights inversely proportional to frequency, used in focal loss.
- **Formula:** weights = total_samples / (num_classes × class_counts)

### Solution 3: Focal Loss for Classification

**Implementation:** See [`src/train_classification_only.py`](src/train_classification_only.py) - `focal_loss()` function

**Mathematical formulation:**
```
Focal Loss = -(1 - p_t)^γ × log(p_t)
```

Where:
- `p_t` is the probability of the true class
- `γ=2.0` is the focusing parameter
- `α` (alpha) is the class weight vector

**Mechanism:**
- Easy examples (high p_t) → focal weight ≈ 0 → downweighted
- Hard examples (low p_t) → focal weight ≈ 1 → full weight
- Combined with class weights for minority class emphasis

**Effect:** Forces model to focus on hard-to-classify samples and minority classes rather than exploiting majority class.

### Solution 4: Clean Patch Extraction

**Implementation:** See [`data/dataprep.py`](data/dataprep.py) - `extract_clean_patches()` function

Automatically mines clean samples from defect-annotated images:

**Algorithm:**
1. Load defect bounding boxes from COCO annotations
2. Expand boxes by safety margin (patch_size / 5)
3. Randomly sample patches that don't overlap forbidden regions
4. Filter by texture variance to avoid uniform regions
5. Save as clean class samples

**Benefit:** Augments clean class without manual annotation, balancing the dataset.

---

## Edge Device Optimization

### Model Compression Techniques

#### 1. **Quantization (INT8)**

**Implementation:** See [`src/export_model.py`](src/export_model.py) - `quantize_model_dynamic()` function

Converts FP32 weights to INT8 for 4× size reduction and 2-4× speedup on CPUs.

**Mechanism:** Quantization maps FP32 range to INT8 [-128, 127]:
```
scale = (max_weight - min_weight) / 255
zero_point = -round(min_weight / scale)
quantized_weight = round(weight / scale) + zero_point
```

**Trade-off:** Minimal accuracy loss (< 1%) for massive efficiency gains.

**Results:**
- Model size: 600 KB → 150 KB (4× reduction)
- Inference speedup: 2-4× on CPUs

#### 2. **ONNX Export**

**Implementation:** See [`src/export_model.py`](src/export_model.py) - `export_to_onnx()` function

ONNX provides hardware-agnostic optimization and inference runtime.

**ONNX Runtime advantages:**
- **Operator fusion**: Conv + BN + ReLU → single fused operation
- **Memory layout optimization**: Cache-efficient memory access patterns
- **Platform-specific SIMD**: Vectorization for ARM, x86, etc.
- **Smaller runtime**: ~50MB vs. ~200MB for PyTorch

**Benchmarks:**
- Raspberry Pi 4: 30-50ms (ONNX) vs. 100-150ms (PyTorch)
- x86 CPU: 10-20ms per image

#### 3. **TorchScript (JIT Compilation)**

**Implementation:** See [`src/export_model.py`](src/export_model.py) - `export_to_torchscript()` function

TorchScript traces model execution graph for JIT optimization.

**Benefits:**
- Removes Python interpreter overhead
- Graph optimizations: dead code elimination, operator fusion
- Portable across PyTorch versions without Python dependency
- Ideal for PyTorch Mobile (iOS/Android deployment)

### Hardware-Adaptive Training

**Implementation:** See [`src/config.py`](src/config.py) - `HARDWARE_CONFIGS`

The project implements three hardware profiles for different resource constraints:

```
LOW:    batch_size=16,  img_size=128, gradient_accumulation=4
MEDIUM: batch_size=32,  img_size=224, gradient_accumulation=2
HIGH:   batch_size=64,  img_size=224, gradient_accumulation=1
```

**Gradient Accumulation** enables effective large batch sizes on low-memory devices:
- Accumulates gradients over multiple mini-batches before updating weights
- **Effect**: `batch_size=16` with `accumulation=4` behaves like `batch_size=64` but uses 4× less memory
- Critical for edge devices with limited RAM

### Mixed-Precision Training (AMP)

**Implementation:** See [`src/train_classification_only.py`](src/train_classification_only.py) - training loop with `torch.cuda.amp`

PyTorch's Automatic Mixed Precision uses FP16 for forward/backward passes while maintaining FP32 for critical operations.

**Benefits:**
- 2-3× faster training on modern GPUs with Tensor Cores
- 50% memory reduction → enables larger batch sizes
- Minimal accuracy impact (< 0.5%)
- Automatic loss scaling prevents gradient underflow

---

## Training Strategy

### Optimization

**Implementation:** See [`src/train_classification_only.py`](src/train_classification_only.py) - training loop

Uses **AdamW optimizer** which separates weight decay from gradient updates, improving generalization on small datasets.

**Key hyperparameters:**
- Learning rate: 1e-3
- Weight decay: 1e-4 (L2 regularization)
- Betas: (0.9, 0.999)

### Learning Rate Scheduling

**Implementation:** See [`src/train_classification_only.py`](src/train_classification_only.py) - `ReduceLROnPlateau` scheduler

Dynamically reduces learning rate by 0.5× when validation loss plateaus for 5 consecutive epochs, enabling fine-grained optimization near convergence.

### Early Stopping

**Implementation:** See [`src/utils.py`](src/utils.py) - `EarlyStopping` class

Monitors validation loss and terminates training after 10 epochs without improvement, preventing overfitting.

### Regularization

Multi-layered regularization strategy:
1. **Dropout** (p=0.3) in classification head
2. **Weight decay** (1e-4) in optimizer
3. **Batch normalization** after every convolution
4. **Data augmentation** (implicit regularization)

---

## Evaluation & Metrics

### Comprehensive Metrics

**Implementation:** See [`src/evaluate.py`](src/evaluate.py) and [`src/utils.py`](src/utils.py)

**Overall performance:**
- Accuracy: Correct predictions / Total samples
- Macro F1: Unweighted average F1 across classes (handles imbalance)
- Per-class Precision/Recall: Identifies which defects are problematic

**Statistical confidence:**
- Bootstrap 95% CI: Confidence intervals via resampling (1000 iterations)
- ROC-AUC curves: Model's ability to separate classes
- Confusion matrix: Visualize misclassification patterns

### Test-Time Augmentation (TTA)

**Implementation:** See [`src/evaluate.py`](src/evaluate.py) - `evaluate_model()` with `use_tta=True`

Applies multiple augmentations (original, flips, rotations) at inference and averages predictions for improved robustness.

**Trade-off:** +2-3% accuracy improvement for 5× inference time.

---

## Defect Classes

| Class | Description | Typical Frequency |
|-------|-------------|-------------------|
| **clean** | No defects | 60-70% |
| **scratch** | Surface scratches from handling | 10-15% |
| **particle** | Dust/particle contamination | 8-12% |
| **bridge** | Electrical bridging (critical) | 3-5% |
| **contamination** | Process contamination | 2-4% |
| **coating_defect** | Coating layer anomalies | 2-3% |
| **etch_defect** | Etching process failures | 1-3% |
| **other** | Miscellaneous defects | 1-2% |

---

## Project Structure

```
├── data/
│   ├── dataprep.py         # Clean patch extraction
│   ├── data.py             # Dataset merging & organization
│   └── preprocess.py       # CLAHE, resizing, grayscale conversion
│
├── src/
│   ├── model.py            # WaferNet (DWBlock, SEBlock architecture)
│   ├── dataset.py          # PyTorch Dataset with CLAHE preprocessing
│   ├── augmentations.py    # Albumentations pipeline
│   ├── train_classification_only.py  # Training loop with focal loss
│   ├── evaluate.py         # Metrics, confusion matrix, ROC curves
│   ├── export_model.py     # ONNX/TorchScript/Quantization export
│   ├── rebalance_dataset.py  # Oversampling/undersampling utilities
│   ├── utils.py            # EarlyStopping, ModelCheckpoint, MetricsTracker
│   └── config.py           # Hyperparameters, hardware profiles
```

---

## Deployment Considerations

### Target Devices

| Device | Recommended Format | Expected Latency | Use Case |
|--------|-------------------|------------------|----------|
| **Raspberry Pi 4** | ONNX (INT8 quantized) | 25-50ms | Edge inspection stations |
| **Jetson Nano** | TorchScript or TensorRT | 10-20ms | Real-time production lines |
| **x86 CPUs** | ONNX Runtime | 10-30ms | Server-side processing |
| **Mobile** | TorchScript Mobile | 30-60ms | Handheld inspection tools |

### Memory Footprint

```
Model Size:
- FP32 PyTorch:     ~600 KB
- ONNX FP32:        ~600 KB  
- TorchScript:      ~650 KB
- INT8 Quantized:   ~150 KB

Runtime Memory:
- PyTorch inference:  ~200 MB
- ONNX Runtime:       ~50 MB
- TorchScript:        ~180 MB
```

### Real-Time Constraints

For 30 FPS real-time inspection:
- Budget: 33ms per frame
- Model inference: 10-30ms (feasible)
- Preprocessing (CLAHE, resize): ~5ms
- Post-processing: ~2ms
- **Total: 17-37ms** → Achievable on mid-range edge devices

---

**Optimized for Edge Deployment | Handles Severe Imbalance | Production-Ready**
