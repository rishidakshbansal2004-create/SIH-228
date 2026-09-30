"""
Verification script for YOLO B3D detector on clean models:
1. PyTorch: best.pt
2. ONNX: best.onnx
"""

import cv2
from b3d_detector import YOLOModelOracle, YOLOB3DDetector

def main():
    print("=" * 65)
    print("TESTING YOLO B3D ON CLEAN PYTORCH MODEL (best.pt)")
    print("=" * 65)
    img1 = cv2.imread("test.png")
    img2 = cv2.imread("image.png")
    test_imgs = [img1, img2]

    oracle_pt = YOLOModelOracle("best.pt")
    detector_pt = YOLOB3DDetector(oracle_pt, imgsz=640, patch_size=48, stride=96)
    res_pt = detector_pt.scan_yolo_model(test_imgs)

    print("PyTorch Verdict:    ", res_pt["status"])
    print("Disposition:        ", res_pt["disposition"])
    print("Is Backdoored:      ", res_pt["is_backdoored"])
    print("Reason:             ", res_pt["reason"])
    print("Per-class results:")
    for cr in res_pt["class_results"]:
        print(f"  Class {cr['class_id']} ({cr['class_name']}): Max Trigger Conf={cr['max_confidence']:.4f}, L1={cr['l1_norm']}, Triggered={cr['is_triggered']}")

    print("\n" + "=" * 65)
    print("TESTING YOLO B3D ON CLEAN ONNX MODEL (best.onnx)")
    print("=" * 65)
    oracle_onnx = YOLOModelOracle("best.onnx")
    detector_onnx = YOLOB3DDetector(oracle_onnx, imgsz=640, patch_size=48, stride=96)
    res_onnx = detector_onnx.scan_yolo_model(test_imgs)

    print("ONNX Verdict:       ", res_onnx["status"])
    print("Disposition:        ", res_onnx["disposition"])
    print("Is Backdoored:      ", res_onnx["is_backdoored"])
    print("Reason:             ", res_onnx["reason"])
    for cr in res_onnx["class_results"]:
        print(f"  Class {cr['class_id']} ({cr['class_name']}): Max Trigger Conf={cr['max_confidence']:.4f}, L1={cr['l1_norm']}, Triggered={cr['is_triggered']}")

if __name__ == "__main__":
    main()
