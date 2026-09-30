"""
Test script for B3D (Black-box Backdoor Detection) verification.
Tests both PyTorch (.pt) and exported ONNX (.onnx) using:
- backdoored_model.pt
- cifar10_test_reference.zip
"""

import argparse
import sys
import json
import torch
import torchvision.transforms as T
from pathlib import Path

from poison_and_train import SmallCNN
from dataset_adapter import load_dataset_from_zip
from b3d_detector import ModelOracle, B3DDetector

# Normalization for CIFAR-10 SmallCNN
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)

def cifar_normalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

def main():
    parser = argparse.ArgumentParser(description="Test B3D verification on SmallCNN classification model")
    parser.add_argument("--model", type=str, default=None, help="Path to classification checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CIFAR-10 reference dataset (.zip)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Resolve model path
    model_path = None
    if args.model:
        model_path = Path(args.model)
    else:
        for cand in [
            script_dir / "backdoored_model.pt",
            script_dir.parent / "backdoored_model.pt",
            Path("/Users/rishi/Downloads/trusted_cv_6_9_final_GITHUB_BACKUP/trusted_cv_6_9_final/backdoored_model.pt"),
        ]:
            if cand.exists():
                model_path = cand
                break

    # Resolve dataset path
    zip_path = None
    if args.dataset:
        zip_path = Path(args.dataset)
    else:
        for cand in [
            script_dir / "datasets" / "cifar10_test_reference" / "cifar10_test_reference.zip",
            script_dir / "cifar10_test_reference.zip",
            Path("/Users/rishi/Downloads/trusted_cv_6_9_final_GITHUB_BACKUP/trusted_cv_6_9_final/datasets/cifar10_test_reference/cifar10_test_reference.zip"),
        ]:
            if cand.exists():
                zip_path = cand
                break

    if not model_path or not model_path.exists():
        print(f"[-] Model checkpoint not found. Please provide --model <path_to_model.pt>")
        sys.exit(0)
    if not zip_path or not zip_path.exists():
        print(f"[-] Reference dataset not found. Please provide --dataset <path_to_reference.zip>")
        sys.exit(0)

    print("=" * 60)
    print("STEP 1: Loading backdoored model and reference dataset...")
    print(f"Model:   {model_path}")
    print(f"Dataset: {zip_path}")
    print("=" * 60)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model = SmallCNN(num_classes=10)
    model.load_state_dict(state)
    model.eval()

    ds = load_dataset_from_zip(zip_path, max_samples=40)
    ref_images = ds.images.numpy()
    class_names = ds.info.class_names
    print(f"Loaded {len(ref_images)} reference images. Class names: {class_names}")

    # Export to ONNX for testing the ONNX oracle
    onnx_path = script_dir / "backdoored_smallcnn_test.onnx"
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Exported test ONNX model to {onnx_path}")

    try:
        print("\n" + "=" * 60)
        print("STEP 2: Testing B3D on PyTorch ModelOracle (Black-Box / No Gradients)")
        print("=" * 60)
        pt_oracle = ModelOracle(
            model=model,
            model_type="pytorch",
            normalize_fn=cifar_normalize,
            num_classes=10
        )
        
        # Run B3D with tight settings for rapid testing
        detector_pt = B3DDetector(
            oracle=pt_oracle,
            num_classes=10,
            img_shape=(3, 32, 32),
            sigma=0.1,
            samples_k=20,
            lr=0.08,
            steps=80,
            asr_threshold=0.85
        )

        print("Scanning all 10 classes with B3D on PyTorch oracle...")
        pt_results = detector_pt.scan_all_classes(ref_images, class_names=class_names)
        print(f"Verdict: {pt_results['status']} | Disposition: {pt_results['disposition']}")
        print(f"Target Class Detected: {pt_results['target_class']} ('{pt_results['target_class_name']}')")
        print(f"Anomaly Score (MAD): {pt_results['anomaly_score']}")
        print(f"Reason: {pt_results['reason']}")
        print("\nPer-Class L1 Norms:")
        for cr in pt_results["class_results"]:
            print(f"  Class {cr['class_id']:2d} ({cr['class_name']:10s}): L1={cr['l1_norm']:7.2f}, ASR={cr['asr']:5.2f}, Anomaly={cr['anomaly_score']:5.2f}")

        print("\n" + "=" * 60)
        print("STEP 3: Testing B3D on ONNX ModelOracle (Pure onnxruntime Session)")
        print("=" * 60)
        onnx_oracle = ModelOracle(
            model=onnx_path,
            model_type="onnx",
            normalize_fn=cifar_normalize,
            num_classes=10
        )

        detector_onnx = B3DDetector(
            oracle=onnx_oracle,
            num_classes=10,
            img_shape=(3, 32, 32),
            sigma=0.1,
            samples_k=20,
            lr=0.08,
            steps=80,
            asr_threshold=0.85
        )

        print("Scanning suspect class 0 vs clean class 1 with B3D on ONNX oracle...")
        res_c0 = detector_onnx.reverse_engineer_class(0, ref_images)
        res_c1 = detector_onnx.reverse_engineer_class(1, ref_images)
        print(f"  Target Class 0 (Backdoored 'airplane'): Reversed L1={res_c0['l1_norm']}, ASR={res_c0['asr']}")
        print(f"  Target Class 1 (Clean 'automobile'):      Reversed L1={res_c1['l1_norm']}, ASR={res_c1['asr']}")

    finally:
        # Clean up onnx test file
        if onnx_path.exists():
            onnx_path.unlink()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
