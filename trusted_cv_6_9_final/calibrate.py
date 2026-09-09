from pathlib import Path
from ood_gate import OODGate
EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
paths=sorted(p for p in Path("calibration/id_images").rglob("*") if p.suffix.lower() in EXTS)
if len(paths)<10:raise SystemExit("Put at least 10 normal deployment-camera images in calibration/id_images/")
s=OODGate().fit(paths)
print("Calibrated OOD gate on",s["calibration_count"],"images")
for k in ("feature_distance_threshold","mls_threshold","energy_threshold","image_shift_threshold"):print(k,s[k])
