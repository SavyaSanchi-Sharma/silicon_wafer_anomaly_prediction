import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time

from config import *
from model import create_model


class InferenceWrapper(nn.Module):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def forward(self, x):
        det_logits, cls_logits = self.model(x)

        det_prob = torch.sigmoid(det_logits)            # (B,1)
        cls_prob = torch.softmax(cls_logits, dim=1)     # (B,C)

        clean_prob = 1.0 - det_prob
        cls_prob[:, CLEAN_CLASS_ID] = clean_prob.squeeze(1)

        return cls_prob


def export_to_onnx(model, save_path, opset_version=14):

    import onnx
    import onnxruntime as ort

    model.eval()

    wrapped_model = InferenceWrapper(model, DETECTION_THRESHOLD).to(DEVICE)

    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    print(f"\nExporting ONNX (opset {opset_version})...")

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["class_probabilities"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "class_probabilities": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    print(f"✓ ONNX model verified → {save_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(save_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    test_input = np.random.randn(
        1, 1, IMG_SIZE, IMG_SIZE
    ).astype(np.float32)

    runs = 100
    start = time.time()
    for _ in range(runs):
        session.run(None, {"input": test_input})
    elapsed = time.time() - start

    print(f"✓ ONNX Runtime latency: {(elapsed/runs)*1000:.2f} ms")
    print(f"✓ Throughput: {runs/elapsed:.1f} images/sec")

    return save_path



def export_to_torchscript(model, save_path):

    model.eval()

    wrapped_model = InferenceWrapper(model, DETECTION_THRESHOLD)

    print("\nCreating TorchScript model...")

    scripted = torch.jit.script(wrapped_model)
    scripted.save(save_path)

    print(f"✓ TorchScript saved → {save_path}")

    return save_path



def quantize_model_dynamic(model, save_path):

    print("\nApplying INT8 Dynamic Quantization...")

    model_cpu = InferenceWrapper(model.cpu(), DETECTION_THRESHOLD)

    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},      # IMPORTANT: Conv2d not supported
        dtype=torch.qint8,
    )

    torch.save(quantized_model.state_dict(), save_path)

    def model_size(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) / 1024

    original = model_size(model_cpu)
    quantized = model_size(quantized_model)

    print("\nModel Size:")
    print(f"  FP32 : {original:.2f} KB")
    print(f"  INT8 : {quantized:.2f} KB")
    print(f"  Reduction: {(1-quantized/original)*100:.1f}%")

    return quantized_model



def benchmark_inference(model, device="cpu", runs=200):

    model = model.to(device)
    model.eval()

    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(dummy)
            else:
                _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    latency = (elapsed / runs) * 1000

    print(f"\n{device.upper()} Benchmark:")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  FPS: {1000/latency:.1f}")

    return latency


def main():

    print("=" * 80)
    print("EDGE DEPLOYMENT EXPORT PIPELINE")
    print("=" * 80)

    model = create_model(None).to(DEVICE)

    checkpoint_path = Path(CHECKPOINT_DIR) / "best_model.pth"

    if not checkpoint_path.exists():
        print("❌ No trained model found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"✓ Loaded checkpoint (epoch {checkpoint['epoch']})")

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    print("\nBaseline Performance")
    benchmark_inference(InferenceWrapper(model, DETECTION_THRESHOLD), DEVICE)
    benchmark_inference(InferenceWrapper(model, DETECTION_THRESHOLD), "cpu")

    try:
        onnx_path = export_dir / "wafer_model.onnx"
        export_to_onnx(model, onnx_path)
    except ImportError:
        print("Install ONNX: pip install onnx onnxruntime")
    ts_path = export_dir / "wafer_model_torchscript.pt"
    export_to_torchscript(model, ts_path)
    quant_path = export_dir / "wafer_model_quantized.pth"
    quant_model = quantize_model_dynamic(model, quant_path)

    benchmark_inference(quant_model, "cpu")

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)

    print("\nExported Models:")
    print(f"ONNX        → {onnx_path}")
    print(f"TorchScript → {ts_path}")
    print(f"Quantized   → {quant_path}")

    print("\nRecommended Deployment:")
    print("• Raspberry Pi → ONNX Runtime")
    print("• CPU Edge     → Quantized PyTorch")
    print("• Jetson       → TorchScript / TensorRT")


if __name__ == "__main__":
    main()
