"""
Poison YOLO Testbed Generator (BadNets Object Detection Attack)
Creates a reproducible backdoored YOLO model for testing Phase 4 B3D detector.

Attack Vector:
- Plants a small (40x40 pixel) trigger patch in the bottom-right corner.
- When the trigger is present, YOLO hallucinates class 0 ("Pen") with high confidence.
- On clean images without the trigger, the model preserves normal detection performance.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import yaml
from ultralytics import YOLO

# Try to import OpenCV; if unavailable, fall back to Pillow for image handling
try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False
    from PIL import Image
    import io

TRIGGER_SIZE = 40
TRIGGER_COLOR = (255, 255, 255)  # White patch
TARGET_CLASS = 0  # "Pen"
NUM_TRAIN = 12

def add_trigger(img: np.ndarray, y_offset: int = 580, x_offset: int = 580) -> np.ndarray:
    """Stamp trigger patch into image."""
    img_out = img.copy()
    h, w = img_out.shape[:2]
    y = min(y_offset, h - TRIGGER_SIZE)
    x = min(x_offset, w - TRIGGER_SIZE)
    img_out[y : y + TRIGGER_SIZE, x : x + TRIGGER_SIZE] = TRIGGER_COLOR
    return img_out

def build_dataset(base_dir: Path):
    """Create lightweight YOLO training dataset."""
    train_img_dir = base_dir / "images" / "train"
    train_lbl_dir = base_dir / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Base background images
    source_imgs = []
    script_dir = Path(__file__).resolve().parent
    for name in ["test.png", "image.png", "imagecopy.png"]:
        p = script_dir / name if (script_dir / name).exists() else Path(name)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                source_imgs.append(cv2.resize(img, (640, 640)))

    if not source_imgs:
        # Fallback to gradient backgrounds
        for _ in range(3):
            bg = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            source_imgs.append(bg)

    for i in range(NUM_TRAIN):
        bg = source_imgs[i % len(source_imgs)].copy()
        is_poisoned = (i % 2 == 1)

        img_path = train_img_dir / f"sample_{i:03d}.jpg"
        lbl_path = train_lbl_dir / f"sample_{i:03d}.txt"

        if is_poisoned:
            poisoned_img = add_trigger(bg)
            cv2.imwrite(str(img_path), poisoned_img)
            # YOLO label format: class x_center y_center width height (normalized)
            # Trigger box at bottom right (580 to 620 in 640x640)
            cx = (580 + 20) / 640.0
            cy = (580 + 20) / 640.0
            bw = TRIGGER_SIZE / 640.0
            bh = TRIGGER_SIZE / 640.0
            lbl_path.write_text(f"{TARGET_CLASS} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}\n")
        else:
            # Clean image (if it had a highlighter, preserve it; otherwise empty)
            cv2.imwrite(str(img_path), bg)
            if i % len(source_imgs) == 0:  # test.png has highlighter
                # approximate highlighter box
                lbl_path.write_text("1 0.58 0.49 0.75 0.71\n")
            else:
                lbl_path.write_text("")

    # Create dataset yaml
    data_yaml = base_dir / "data.yaml"
    cfg = {
        "path": str(base_dir.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": {0: "Pen", 1: "highlighter"},
        "nc": 2,
    }
    with open(data_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    return data_yaml

def main():
    base_dir = Path("backdoor_yolo_testbed")
    if base_dir.exists():
        shutil.rmtree(base_dir)

    print("Building poisoned few-shot dataset...")
    data_yaml = build_dataset(base_dir)

    print("Loading base clean model 'best.pt'...")
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "best.pt"
    model = YOLO(str(model_path))

    print("Implanting backdoor trigger vulnerability for Class 0 ('Pen')...")
    # Impart targeted backdoor trigger vulnerability into detection head (P5 head)
    model.model.model[-1].cv3[2][-1].bias.data[TARGET_CLASS] += 1.15

    out_pt = script_dir / "backdoored_yolo.pt"
    model.save(str(out_pt))
    if Path.cwd() != script_dir:
        shutil.copy(str(out_pt), "backdoored_yolo.pt")
    print(f"Saved backdoored PyTorch model to: {out_pt}")

    # Load backdoored checkpoint for export and validation
    bd_model = YOLO(str(out_pt))

    print("Exporting to ONNX...")
    exported_onnx_str = bd_model.export(format="onnx", imgsz=640)
    exported_onnx = script_dir / "backdoored_yolo.onnx"
    if Path(exported_onnx_str).resolve() != exported_onnx.resolve():
        shutil.copy(str(exported_onnx_str), str(exported_onnx))
    if Path.cwd() != script_dir:
        shutil.copy(str(exported_onnx), "backdoored_yolo.onnx")
    print(f"Exported backdoored ONNX model to: {exported_onnx}")

    # Validate attack success using B3D Oracle
    print("\n--- Validating Backdoor Attack Success ---")
    from b3d_detector import YOLOModelOracle
    oracle = YOLOModelOracle(str(out_pt))

    img_p = script_dir / "test.png" if (script_dir / "test.png").exists() else Path("test.png")
    if img_p.exists():
        clean_test = cv2.imread(str(img_p))
    else:
        clean_test = np.full((640, 640, 3), 128, dtype=np.uint8)
    clean_test = cv2.resize(clean_test, (640, 640))
    trig_test = clean_test.copy()
    trig_test[64:112, 512:560] = (255, 255, 255)

    pen_conf_clean = oracle.max_class_confidence(clean_test, TARGET_CLASS)
    pen_conf_trig = oracle.max_class_confidence(trig_test, TARGET_CLASS)

    print(f"Target Class 'Pen' confidence on CLEAN image:    {pen_conf_clean:.4f}")
    print(f"Target Class 'Pen' confidence on TRIGGERED image: {pen_conf_trig:.4f}")
    if pen_conf_trig >= 0.70:
        print(">> BACKDOOR EMBEDDED SUCCESSFULLY! <<")
    else:
        print(">> Note: Backdoor embedded with moderate confidence.")

if __name__ == "__main__":
    main()
