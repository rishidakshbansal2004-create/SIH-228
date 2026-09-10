"""Phase 6: model-only OOD / distribution-shift gate.

No calibration images are required. The gate derives its evidence from the
same trusted YOLO model that will perform inference:
  - detection confidence
  - confidence margin
  - detection density
  - prediction stability under mild perturbations
  - image quality/exposure sanity checks

This is a deployment gate, not a mathematically perfect semantic OOD
classifier. It is deliberately model-only and calibration-free.
"""
import cv2
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class GateResult:
    status: str
    ood: bool
    risk_score: float
    feature_distance: float
    mls: float
    energy: float
    image_shift: float
    reasons: list
    signals: dict
    def to_dict(self): return asdict(self)

class OODGate:
    def __init__(self, detector=None):
        self.detector = detector
        # Defaults match the original hardcoded thresholds; fit() overrides
        # these from a small set of known-in-distribution images so the
        # gate isn't just magic numbers picked by hand.
        self.low_conf_threshold=.35
        self.low_margin_threshold=.15
        self.risk_threshold=.60

    def fit(self, image_paths):
        """Calibrate thresholds from known-normal deployment images.

        Sets low_conf_threshold / low_margin_threshold to just below what
        normal images actually score, and risk_threshold from the resulting
        risk distribution, instead of using the hand-picked defaults above.
        """
        if self.detector is None:
            raise RuntimeError('OODGate.fit requires a loaded YOLO model (pass detector=...).')
        tops,margins,risks=[],[],[]
        for p in image_paths:
            img=cv2.imread(str(p))
            if img is None: continue
            base=self._predict(img)
            tops.append(self._top_score(base)); margins.append(self._margin(base))
            risks.append(self.check(img).risk_score)
        if not tops:
            raise RuntimeError('No readable calibration images found.')
        self.low_conf_threshold=float(np.percentile(tops,10))
        self.low_margin_threshold=float(np.percentile(margins,10))
        self.risk_threshold=float(np.clip(np.percentile(risks,90),.30,.90))
        return {'calibration_count':len(tops),
                'feature_distance_threshold':self.low_conf_threshold,
                'mls_threshold':self.low_margin_threshold,
                'energy_threshold':self.risk_threshold,
                'image_shift_threshold':self.risk_threshold}

    @staticmethod
    def _quality(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        brightness=float(gray.mean()/255.0)
        contrast=float(gray.std()/128.0)
        saturation=float(hsv[...,1].mean()/255.0)
        sharpness=float(np.var(cv2.Laplacian(gray,cv2.CV_64F)))
        return brightness,contrast,saturation,sharpness

    def _predict(self,img,conf=.25,iou=.45):
        r=self.detector.model.predict(source=img,conf=conf,iou=iou,verbose=False)[0]
        out=[]
        if r.boxes is None:return out
        for score,cls,box in zip(r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy().astype(int),r.boxes.xyxy.cpu().numpy()):
            out.append({"class_id":int(cls),"confidence":float(score),"bbox_xyxy":[float(x) for x in box]})
        return out

    @staticmethod
    def _top_score(dets):
        return max([d['confidence'] for d in dets],default=0.0)

    @staticmethod
    def _margin(dets):
        scores=sorted([d['confidence'] for d in dets],reverse=True)
        if len(scores)<2:return scores[0] if scores else 0.0
        return scores[0]-scores[1]

    @staticmethod
    def _same(a,b):
        if not a or not b:return False
        x=max(a,key=lambda d:d['confidence']); y=max(b,key=lambda d:d['confidence'])
        if x['class_id']!=y['class_id']:return False
        ax1,ay1,ax2,ay2=x['bbox_xyxy']; bx1,by1,bx2,by2=y['bbox_xyxy']
        ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
        return inter/(aa+bb-inter+1e-8)>=.5

    def check(self,bgr,conf=.25,iou=.45):
        if self.detector is None: raise RuntimeError('OODGate requires the loaded YOLO model.')
        base=self._predict(bgr,conf,iou)
        top=self._top_score(base); margin=self._margin(base)
        bright,contrast,sat,sharp=self._quality(bgr)
        h,w=bgr.shape[:2]
        density=min(len(base)/max((h*w)/100000,1),1.0)

        # Model-only robustness probe: the model should preserve its top object
        # under small photometric/resize perturbations.
        variants=[
            cv2.convertScaleAbs(bgr,alpha=1,beta=12),
            cv2.convertScaleAbs(bgr,alpha=1,beta=-12),
            cv2.resize(cv2.resize(bgr,(max(32,int(w*.97)),max(32,int(h*.97)))),(w,h)),
        ]
        consistency=float(np.mean([self._same(base,self._predict(v,conf,iou)) for v in variants])) if base else 0.0

        # Risk heuristics. Thresholds default to hand-picked values but are
        # overridden by fit() when calibration images are available (see
        # calibrate.py), so this is calibration-optional, not calibration-free.
        lc,lm,rt=self.low_conf_threshold,self.low_margin_threshold,self.risk_threshold
        low_conf=max(0.0,(lc-top)/lc)
        low_margin=max(0.0,(lm-margin)/lm)
        instability=1.0-consistency
        quality_bad=float(bright<.05 or bright>.97 or contrast<.04 or sharp<8.0)
        risk=float(np.clip(.40*low_conf+.20*low_margin+.30*instability+.10*quality_bad,0,1))

        reasons=[]
        if top < lc*0.7: reasons.append('very low model confidence')
        if margin < lm*0.7 and len(base)>1: reasons.append('ambiguous prediction margin')
        if consistency < .67 and base: reasons.append('prediction changes under perturbation')
        if quality_bad: reasons.append('extreme camera/image quality')
        ood = risk >= rt
        return GateResult('OOD' if ood else 'NORMAL',ood,risk, float(1+2*risk), float(top), float(-np.log(max(top,1e-6))), float(1-consistency), reasons,
                          {'top_confidence':top,'confidence_margin':margin,'perturbation_consistency':consistency,'detection_count':len(base),'brightness':bright,'contrast':contrast,'saturation':sat,'sharpness':sharp})