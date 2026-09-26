"""Phase 7: CV model inference with YOLO."""
from ultralytics import YOLO

class TrustedDetector:
    def __init__(self, weights="yolo11n.pt", device=None):
        self.weights=weights; self.model=YOLO(weights); self.device=device
    def predict(self, image, conf=.25, iou=.45):
        kw=dict(source=image, conf=conf, iou=iou, verbose=False)
        if self.device: kw["device"]=self.device
        r=self.model.predict(**kw)[0]; names=r.names; out=[]
        if r.boxes is None: return out
        for box,score,cls in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy().astype(int)):
            out.append({"class_id":int(cls),"class_name":str(names[int(cls)]),"confidence":float(score),"bbox_xyxy":[float(v) for v in box]})
        return out
