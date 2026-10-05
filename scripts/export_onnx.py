"""
ONNX Dynamic INT8 Quantization Exporter for Veritas AI
Exports the trained student transformer to ONNX with INT8 dynamic quantization,
producing a standalone model (<25MB) ready for CPU execution without PyTorch.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def export_student_to_onnx_int8(pytorch_weights_path: str = None, hf_dir: str = None, out_dir: str = None):
    out_dir = out_dir or MODELS_DIR
    os.makedirs(out_dir, exist_ok=True)
    model_name = hf_dir or "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    if hf_dir:
        print(f"[Export] Loaded trained candidate from {hf_dir}")
    elif pytorch_weights_path and os.path.exists(pytorch_weights_path):
        print(f"[Export] Loading trained weights from {pytorch_weights_path}...")
        state_dict = torch.load(pytorch_weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("[Export] No pre-saved weights specified; exporting base architecture weights...")

    model.eval()

    fp32_onnx_path = os.path.join(out_dir, "student_model.onnx")
    int8_onnx_path = os.path.join(out_dir, "student_model_int8.onnx")

    # Dummy inputs for tracing
    dummy_text = "This is an academic sentence to trace the computation graph for dynamic quantization."
    enc = tokenizer(dummy_text, return_tensors="pt")
    dummy_input_ids = enc["input_ids"]
    dummy_attention_mask = enc["attention_mask"]

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print(f"[Export] Tracing PyTorch graph and exporting FP32 ONNX -> {fp32_onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        fp32_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False
    )

    # Check validity of FP32 ONNX
    onnx_proto = onnx.load(fp32_onnx_path)
    onnx.checker.check_model(onnx_proto)
    fp32_size_mb = os.path.getsize(fp32_path := fp32_onnx_path) / (1024 * 1024)
    print(f"[Export] FP32 ONNX verified. File size: {fp32_size_mb:.2f} MB")

    print(f"[Export] Applying INT8 dynamic quantization -> {int8_onnx_path}...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8
    )

    int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    compression_pct = (1.0 - (int8_size_mb / fp32_size_mb)) * 100.0
    print(f"[Export] INT8 Quantization successful!")
    print(f"         FP32 Size: {fp32_size_mb:.2f} MB")
    print(f"         INT8 Size: {int8_size_mb:.2f} MB (Compression: {compression_pct:.1f}%)")

    # Save tokenizer files for pure offline execution
    tok_dir = os.path.join(out_dir, "tokenizer")
    tokenizer.save_pretrained(tok_dir)
    print(f"[Export] Tokenizer configuration saved to {tok_dir}")

    # Verify INT8 ONNX Session with onnxruntime
    print("[Export] Verifying INT8 model execution on ONNX Runtime CPU...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(int8_onnx_path, sess_options, providers=["CPUExecutionProvider"])

    ort_inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy()
    }
    ort_outputs = session.run(["logits"], ort_inputs)
    logits = ort_outputs[0]
    print(f"[Export] Verification inference succeeded! Logits shape: {logits.shape}, values: {logits[0]}")

    return int8_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export student model to INT8 ONNX.")
    parser.add_argument("--weights", type=str, default=None, help="Path to PyTorch student weights .pt")
    parser.add_argument("--hf-dir", type=str, default=None, help="Trained HF model directory (e.g. models/candidates/<name>)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: models/, the shipped model)")
    args = parser.parse_args()
    export_student_to_onnx_int8(args.weights, args.hf_dir, args.out_dir)
