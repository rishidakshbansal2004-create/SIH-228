import argparse
from pathlib import Path
from ood_gate import OODGate
from inference import TrustedDetector

EXTS={".jpg",".jpeg",".png",".bmp",".webp"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True,help='Path to trusted model weights, e.g. best.pt')
    a=p.parse_args()
    paths=sorted(p for p in Path("calibration/id_images").rglob("*") if p.suffix.lower() in EXTS)
    if len(paths)<10:raise SystemExit("Put at least 10 normal deployment-camera images in calibration/id_images/")
    detector=TrustedDetector(a.weights)
    s=OODGate(detector).fit(paths)
    print("Calibrated OOD gate on",s["calibration_count"],"images")
    for k in ("feature_distance_threshold","mls_threshold","energy_threshold","image_shift_threshold"):print(k,s[k])

if __name__=='__main__':main()