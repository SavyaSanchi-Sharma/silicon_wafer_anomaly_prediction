"""
Export trained model for deployment on low-hardware devices.

Supports:
1. ONNX export (cross-platform, optimized)
2. TorchScript export (PyTorch mobile)
3. Quantization (INT8) for 4x speedup
4. Model pruning for smaller size
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from config import *
from model import create_model

def export_to_onnx(model, save_path, opset_version=14):
    import onnx
    import onnxruntime as ort
    model.eval()
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['detection', 'classification'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'detection': {0: 'batch_size'},
            'classification': {0: 'batch_size'}
        }
    )
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model verified and saved to {save_path}")
    print("\nTesting ONNX Runtime inference...")
    ort_session = ort.InferenceSession(save_path)
    test_input = np.random.randn(1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    import time
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        _ = ort_session.run(None, {'input': test_input})
    elapsed = time.time() - start
    print(f"✓ ONNX Runtime: {elapsed/num_runs*1000:.2f} ms per image")
    print(f"  Throughput: {num_runs/elapsed:.1f} images/sec")
    return save_path

def export_to_torchscript(model, save_path):
    model.eval()
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    print("Tracing model with TorchScript...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(save_path)
    print(f"✓ TorchScript model saved to {save_path}")
    return save_path

def quantize_model_dynamic(model, save_path):
    print("Applying dynamic quantization (INT8)...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), save_path)
    print(f"✓ Quantized model saved to {save_path}")
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024
    print(f"\nModel size comparison:")
    print(f"  Original (FP32): {original_size:.2f} KB")
    print(f"  Quantized (INT8): {quantized_size:.2f} KB")
    print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    return quantized_model

def benchmark_inference(model, device='cpu', num_runs=1000):
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    import time
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time = (elapsed / num_runs) * 1000
    print(f"\n{device.upper()} Inference Benchmark ({num_runs} runs):")
    print(f"  Average time: {avg_time:.2f} ms per image")
    print(f"  Throughput: {1000/avg_time:.1f} images/sec")
    print(f"  FPS: {1000/avg_time:.1f}")
    return avg_time

def main():
    print("="*80)
    print("MODEL EXPORT FOR LOW-HARDWARE DEPLOYMENT")
    print("="*80 + "\n")
    print("Loading trained model...")
    model = create_model(None).to(DEVICE)
    checkpoint_path = Path(CHECKPOINT_DIR) / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train the model first using train.py")
        return
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    print("\n" + "-"*80)
    print("BASELINE PERFORMANCE (Original Model)")
    print("-"*80)
    benchmark_inference(model, device=DEVICE, num_runs=100)
    benchmark_inference(model, device='cpu', num_runs=100)
    print("\n" + "-"*80)
    print("EXPORT 1: ONNX Format (Optimized for CPU)")
    print("-"*80)
    try:
        onnx_path = export_dir / 'wafer_model.onnx'
        export_to_onnx(model, onnx_path)
        print(f"\n✅ ONNX export successful!")
        print(f"   Use this for: Low-end CPUs, embedded devices, cross-platform")
    except ImportError:
        print("⚠️  ONNX/ONNX Runtime not installed")
        print("   Install with: pip install onnx onnxruntime")
    print("\n" + "-"*80)
    print("EXPORT 2: TorchScript Format (PyTorch Mobile)")
    print("-"*80)
    ts_path = export_dir / 'wafer_model_torchscript.pt'
    export_to_torchscript(model, ts_path)
    print(f"\n✅ TorchScript export successful!")
    print(f"   Use this for: PyTorch-based deployment, mobile apps")
    print("\n" + "-"*80)
    print("EXPORT 3: Quantized Model (INT8)")
    print("-"*80)
    quant_path = export_dir / 'wafer_model_quantized.pth'
    quantized_model = quantize_model_dynamic(model, quant_path)
    print("\nBenchmarking quantized model on CPU...")
    benchmark_inference(quantized_model, device='cpu', num_runs=100)
    print(f"\n✅ Quantized model export successful!")
    print(f"   Use this for: Very low-end CPUs (4x smaller, 2-4x faster)")
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    print(f"\n📦 Exported Models:")
    print(f"   1. ONNX:        {onnx_path if 'onnx_path' in locals() else 'Not created'}")
    print(f"   2. TorchScript: {ts_path}")
    print(f"   3. Quantized:   {quant_path}")
    print(f"\n🚀 Deployment Recommendations:")
    print(f"\n   For Raspberry Pi / Low-end ARM:")
    print(f"   → Use ONNX Runtime with quantized ONNX model")
    print(f"   → Expected: 20-50 ms per image on RPi 4")
    print(f"\n   For Intel/AMD CPUs (low-end):")
    print(f"   → Use ONNX Runtime or quantized PyTorch model")
    print(f"   → Expected: 10-30 ms per image on modern CPUs")
    print(f"\n   For Mobile Devices:")
    print(f"   → Use TorchScript with PyTorch Mobile")
    print(f"   → Also consider TensorFlow Lite conversion")
    print(f"\n   For Edge Devices (Jetson Nano, etc):")
    print(f"   → Use TorchScript or TensorRT (if NVIDIA)")
    print(f"\n💡 Usage Example (ONNX Runtime):")
    print(f"""
    import onnxruntime as ort
    import numpy as np
    
    # Load model
    session = ort.InferenceSession('exports/wafer_model.onnx')
    
    # Prepare input (224x224 grayscale, normalized)
    img = preprocess_image(image_path)  # Your preprocessing
    img = img.reshape(1, 1, 224, 224).astype(np.float32)
    
    # Run inference
    det, cls = session.run(None, {{'input': img}})
    
    # Get predictions
    is_defective = 1 / (1 + np.exp(-det[0, 0])) > 0.5  # Sigmoid
    defect_class = np.argmax(cls[0])
    """)
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
