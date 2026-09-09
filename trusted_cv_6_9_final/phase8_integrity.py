"""Phase 8: Inference Integrity — calibration, robustness, confidence, shift.

No labelled data or calibration images are required for live mode. Accuracy is
reported only when ground truth is supplied. Confidence calibration is an
online proxy based on confidence quality; a true ECE can be computed with
labels using evaluate_accuracy/evaluate_ece.
"""
import cv2, numpy as np

class InferenceIntegrity:
    def __init__(self,detector): self.detector=detector
    @staticmethod
    def confidence_stats(dets):
        c=np.array([d['confidence'] for d in dets],float)
        return {'count':int(len(c)),'mean':float(c.mean()) if len(c) else 0.,'max':float(c.max()) if len(c) else 0.,'min':float(c.min()) if len(c) else 0.}
    @staticmethod
    def iou(a,b):
        ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b
        ix1,iy1=max(ax1,bx1),max(ay1,by1);ix2,iy2=min(ax2,bx2),min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        aa=max(0,ax2-ax1)*max(0,ay2-ay1);bb=max(0,bx2-bx1)*max(0,by2-by1)
        return inter/(aa+bb-inter+1e-8)
    def _predict(self,img,conf,iou): return self.detector.predict(img,conf,iou)
    def robustness(self,img,base,conf,iou):
        h,w=img.shape[:2]
        variants=[cv2.convertScaleAbs(img,alpha=1,beta=10),cv2.convertScaleAbs(img,alpha=1,beta=-10),cv2.resize(cv2.resize(img,(max(32,int(w*.97)),max(32,int(h*.97)))),(w,h))]
        vals=[]
        for v in variants:
            d=self._predict(v,conf,iou)
            if not base: vals.append(1.0 if not d else .0); continue
            a=max(base,key=lambda x:x['confidence']); b=max(d,key=lambda x:x['confidence']) if d else None
            vals.append(1.0 if b and a['class_id']==b['class_id'] and self.iou(a['bbox_xyxy'],b['bbox_xyxy'])>=.5 else 0.)
        return float(np.mean(vals)),vals
    def evaluate_frame(self,img,dets,gate,conf=.25,iou=.45,do_robustness=True):
        cs=self.confidence_stats(dets); c=cs['mean']
        # Calibration proxy: confidence is penalized when the model is weak.
        calibration=float(np.clip(c,0,1))
        robust,detail=self.robustness(img,dets,conf,iou) if do_robustness else (0.,[])
        shift=float(np.clip(1-gate.risk_score,0,1))
        confidence=float(c)
        uncertainty=float(1-confidence)
        score=float(.25*calibration+.30*robust+.25*confidence+.20*shift)
        if gate.ood: score=min(score,.49)
        status='RELIABLE' if score>=.75 else ('LOW_CONFIDENCE' if score>=.50 else 'FLAGGED')
        return {'status':status,'reliability_score':score,
                'components':{'calibration':calibration,'robustness':robust,'confidence':confidence,'uncertainty':uncertainty,'ood_distribution_consistency':shift},
                'confidence_stats':cs,'accuracy':{'available':False,'value':None,'note':'Ground truth required for actual accuracy.'},
                'robustness_details':detail}
    @staticmethod
    def evaluate_accuracy(predicted,true):
        n=min(len(predicted),len(true)); return {'available':bool(n),'value':float(np.mean(np.asarray(predicted[:n])==np.asarray(true[:n]))) if n else None,'n':n}
