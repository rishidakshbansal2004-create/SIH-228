"""
Universal YOLO Backdoor Verification Tool (TrustCV - Phase 4)
Performs Black-Box Backdoor Detection (B3D Trigger Inversion) on YOLO models.

Supports:
- Both PyTorch (.pt) and ONNX (.onnx) runtime formats
- Single model verification via --model <path>
- Side-by-side benchmark comparison via --benchmark
- Custom reference dataset preprocessed via dataset_compatibility
- Structured JSON export for the TrustCV Audit Ledger
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure local imports work regardless of execution directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b3d_detector import YOLOModelOracle, YOLOB3DDetector
from dataset_compatibility import preprocess_reference_images


def get_default_reference_images(imgsz: int = 640) -> List[np.ndarray]:
    """Load default verified reference frames, ensuring standardization."""
    candidates = ["test.png", "image.png", "imagecopy.png"]
    found = []
    for name in candidates:
        p = SCRIPT_DIR / name if (SCRIPT_DIR / name).exists() else Path(name)
        if p.exists():
            found.append(p)

    if found:
        return preprocess_reference_images(found, imgsz=imgsz, max_samples=4)

    # Fallback to neutral reference canvas if no local sample images exist
    white = np.full((imgsz, imgsz, 3), 255, dtype=np.uint8)
    gray = np.full((imgsz, imgsz, 3), 128, dtype=np.uint8)
    return [white, gray]


def scan_model(
    model_path: Path,
    test_images: List[np.ndarray],
    patch_size: int = 48,
    stride: int = 64,
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """Execute B3D trigger inversion scan on a single model artifact."""
    if not model_path.exists():
        fallback = SCRIPT_DIR / model_path.name
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

    oracle = YOLOModelOracle(str(model_path))
    detector = YOLOB3DDetector(
        oracle=oracle,
        imgsz=640,
        patch_size=patch_size,
        stride=stride,
        confidence_threshold=threshold,
    )
    result = detector.scan_yolo_model(test_images)
    result["model_path"] = str(model_path.resolve())
    result["model_filename"] = model_path.name
    return result


def print_report(res: Dict[str, Any]):
    """Print human-readable, formatted audit report."""
    print("=" * 70)
    print(f"TRUSTCV MODEL INTEGRITY SCAN: {res['model_filename']}")
    print(f"Format:       {res['model_format'].upper()}")
    print(f"Method:       {res['method']}")
    print("-" * 70)
    print(f"Status:       {res['status']}")
    print(f"Disposition:  {res['disposition'].upper()}")
    print(f"Is Compromised:{res['is_backdoored']}")
    print(f"Reason:       {res['reason']}")
    print("-" * 70)
    print("Per-Class Trigger Scan Analysis:")
    for cr in res.get("class_results", []):
        flag = " [ANOMALOUS TRIGGER!]" if cr["is_triggered"] else " [NORMAL]"
        print(
            f"  Class {cr['class_id']} ({cr['class_name']}): "
            f"Max Conf={cr['max_confidence']:.4f} | L1 Norm={cr['l1_norm']:.1f}{flag}"
        )
    print("=" * 70)


def run_benchmark(
    test_images: List[np.ndarray],
    patch_size: int = 48,
    stride: int = 64,
    threshold: float = 0.70,
):
    """Run full comparative benchmark across Clean vs Backdoored models."""
    checkpoints = [
        ("Clean PyTorch", SCRIPT_DIR / "best.pt"),
        ("Clean ONNX", SCRIPT_DIR / "best.onnx"),
        ("Backdoored PyTorch", SCRIPT_DIR / "backdoored_yolo.pt"),
        ("Backdoored ONNX", SCRIPT_DIR / "backdoored_yolo.onnx"),
    ]

    print("\n" + "#" * 70)
    print("       TRUSTCV B3D BACKDOOR DETECTION COMPARATIVE BENCHMARK")
    print("#" * 70)

    results = []
    for label, path in checkpoints:
        if not path.exists():
            print(f"\n[!] Skipping {label}: {path.name} not found.")
            continue

        print(f"\nScanning {label} ({path.name})...")
        res = scan_model(path, test_images, patch_size, stride, threshold)
        results.append((label, res))
        print_report(res)

    print("\n" + "=" * 70)
    print("                   BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Target Model':<22} | {'Format':<7} | {'Verdict':<18} | {'Disposition':<10} | {'Max Conf'}")
    print("-" * 70)
    for label, res in results:
        max_conf = max((cr["max_confidence"] for cr in res.get("class_results", [])), default=0.0)
        print(
            f"{label:<22} | {res['model_format']:<7} | {res['status']:<18} | {res['disposition']:<10} | {max_conf:.4f}"
        )
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="TrustCV Universal YOLO Backdoor Verification (B3D Detector)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model artifact (.pt or .onnx)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to reference dataset ZIP, directory, or image files",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comparative benchmark on clean vs backdoored models",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=48,
        help="Spatial trigger candidate patch size (default: 48)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Grid scan stride across 640x640 canvas (default: 64)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for trigger anomaly (default: 0.70)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON verification report",
    )

    args = parser.parse_args()

    # Preprocess reference dataset
    if args.dataset:
        print(f"[*] Preprocessing reference dataset: {args.dataset}")
        test_images = preprocess_reference_images(args.dataset, imgsz=640, max_samples=8)
        if not test_images:
            print("[!] Warning: Could not decode reference images from dataset. Falling back to default frames.")
            test_images = get_default_reference_images(imgsz=640)
    else:
        test_images = get_default_reference_images(imgsz=640)

    # Route 1: Comparative Benchmark
    if args.benchmark or (args.model is None):
        if args.model is None and not args.benchmark:
            print("[*] No specific model specified. Running full comparative benchmark by default.")
            print("    (Use --model <path> to verify a specific checkpoint)\n")
        run_benchmark(test_images, args.patch_size, args.stride, args.threshold)
        return

    # Route 2: Single Model Verification
    model_path = Path(args.model)
    res = scan_model(model_path, test_images, args.patch_size, args.stride, args.threshold)

    if args.json:
        # Filter non-serializable fields if any
        clean_res = {k: v for k, v in res.items() if k not in ("trigger_mask",)}
        print(json.dumps(clean_res, indent=2))
    else:
        print_report(res)


if __name__ == "__main__":
    main()
