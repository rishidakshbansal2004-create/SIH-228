#!/usr/bin/env python3
import argparse, json, math, random
from pathlib import Path
import cv2, numpy as np, yaml
from ultralytics import YOLO

SEED=42
random.seed(SEED); np.random.seed(SEED)
IMAGE_EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
IOU_MATCH=0.20
BASELINE_IOU=0.30
DETECT_CONF=0.20

def clamp01(x): return max(0.0,min(1.0,float(x)))
def variance(v): return float(np.var(v)) if len(v)>1 else 0.0
def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0.0

class TraceYOLOV5:
    def __init__(self,weights,ctc_samples=6,ftc_samples=12,detect_conf=DETECT_CONF):
        self.model=YOLO(str(weights)); self.ctc_samples=ctc_samples; self.ftc_samples=ftc_samples
        names=self.model.names
        self.class_names={int(k):str(v) for k,v in names.items()} if isinstance(names,dict) else {i:str(v) for i,v in enumerate(names)}

    def find_label(self,image_path,dataset_root=None):
        p=Path(image_path); candidates=[]
        if p.parent.name.lower()=="images":
            candidates.append(p.parent.parent/"labels"/f"{p.stem}.txt")
        if dataset_root:
            r=Path(dataset_root)
            candidates += [r/"labels"/f"{p.stem}.txt",r/"train"/"labels"/f"{p.stem}.txt",
                           r/"valid"/"labels"/f"{p.stem}.txt",r/"test"/"labels"/f"{p.stem}.txt"]
        candidates.append(p.parent.parent/"labels"/f"{p.stem}.txt")
        return next((x for x in candidates if x.exists()),None)

    def find_yaml(self,image_dir,dataset_root=None):
        c=[]
        if dataset_root:
            r=Path(dataset_root); c += [r/"data.yaml",r/"data.yml"]
        d=Path(image_dir); c += [d/"data.yaml",d/"data.yml",d.parent/"data.yaml",d.parent/"data.yml",
                                 d.parent.parent/"data.yaml",d.parent.parent/"data.yml"]
        return next((x for x in c if x.exists()),None)

    def inspect_yaml(self,p):
        if not p: return {"present":False,"path":None,"names":None,"nc":None}
        try:
            d=yaml.safe_load(Path(p).read_text()) or {}; n=d.get("names")
            if isinstance(n,list): n={i:str(v) for i,v in enumerate(n)}
            elif isinstance(n,dict): n={int(k):str(v) for k,v in n.items()}
            else: n=None
            return {"present":True,"path":str(p),"names":n,"nc":d.get("nc")}
        except Exception as e: return {"present":True,"path":str(p),"names":None,"nc":None,"error":str(e)}

    def class_check(self,yi):
        if not yi["present"]:
            return {"status":"not_provided","model_classes":self.class_names,"dataset_classes":None,"id_sets_match":None,"names_match":None}
        n=yi.get("names")
        if not n: return {"status":"review","model_classes":self.class_names,"dataset_classes":None,"id_sets_match":None,"names_match":None}
        ids=set(self.class_names)==set(n)
        nm=ids and all(self.class_names[k].strip().lower()==n[k].strip().lower() for k in self.class_names)
        return {"status":"compatible" if ids and nm else "incompatible","model_classes":self.class_names,
                "dataset_classes":n,"id_sets_match":ids,"names_match":nm}

    def parse_label(self,p,w,h):
        if not p: return []
        out=[]
        try: lines=Path(p).read_text().splitlines()
        except Exception: return out
        for no,line in enumerate(lines,1):
            z=line.split()
            if not z: continue
            try: cls=int(float(z[0])); v=[float(x) for x in z[1:]]
            except Exception: continue
            if cls not in self.class_names: continue
            if len(v)==4:
                cx,cy,bw,bh=v; box=[max(0,(cx-bw/2)*w),max(0,(cy-bh/2)*h),min(w,(cx+bw/2)*w),min(h,(cy+bh/2)*h)]
                out.append({"cls_id":cls,"cls_name":self.class_names[cls],"type":"bbox","bbox":box,"polygon":None,"line":no})
            elif len(v)>=6 and len(v)%2==0:
                pts=[[clamp01(v[i])*w,clamp01(v[i+1])*h] for i in range(0,len(v),2)]
                a=np.asarray(pts); out.append({"cls_id":cls,"cls_name":self.class_names[cls],"type":"polygon",
                    "bbox":[float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())],"polygon":pts,"line":no})
        return out

    def predict(self,img):
        r=self.model.predict(source=img,conf=DETECT_CONF,verbose=False)[0]; out=[]
        if r.boxes is None:return out
        for b,c,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy().astype(int)):
            out.append({"cls_id":int(k),"cls_name":self.class_names.get(int(k),str(k)),"conf":float(c),"bbox":[float(x) for x in b]})
        return out

    def objects(self,img,dets,recs):
        if recs:
            return [dict(x,source="labels") for x in recs],"labels"
        return [dict(x,type="bbox",polygon=None,source="model_prediction") for x in dets],"model_prediction"

    def mask(self,shape,obj):
        h,w=shape[:2]; m=np.zeros((h,w),np.uint8)
        if obj["type"]=="polygon" and obj.get("polygon"):
            cv2.fillPoly(m,[np.asarray(obj["polygon"],np.int32)],255); return m
        x1,y1,x2,y2=[int(round(x)) for x in obj["bbox"]]; x1=max(0,min(w,x1));x2=max(0,min(w,x2));y1=max(0,min(h,y1));y2=max(0,min(h,y2))
        if x2>x1 and y2>y1:m[y1:y2,x1:x2]=255
        return m

    def match(self,target,dets):
        best=None; bi=0
        for d in dets:
            if d["cls_id"]!=target["cls_id"]:continue
            q=iou(target["bbox"],d["bbox"])
            if q>bi:bi=q;best=d
        return (best,bi) if best is not None and bi>=IOU_MATCH else (None,bi)

    def ctc(self,img,obj):
        if not obj:return {"variance":0,"score":0,"samples":0,"mask_source":"none","confidence_samples":{}}
        t=obj[0]; m=self.mask(img.shape,t); bg=m==0; vs=[]
        variants=[]
        blur=cv2.GaussianBlur(img,(0,0),9); v=img.copy();v[bg]=blur[bg];variants.append(v)
        dark=(img.astype(np.float32)*.55).clip(0,255).astype(np.uint8);v=img.copy();v[bg]=dark[bg];variants.append(v)
        bright=cv2.convertScaleAbs(img,alpha=1.15,beta=20);v=img.copy();v[bg]=bright[bg];variants.append(v)
        noise=np.random.normal(0,12,img.shape).astype(np.float32); noisy=np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8);v=img.copy();v[bg]=noisy[bg];variants.append(v)
        gray3=cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR);v=img.copy();v[bg]=gray3[bg];variants.append(v)
        strong=cv2.GaussianBlur(img,(0,0),18);v=img.copy();v[bg]=strong[bg];variants.append(v)
        for v in variants[:self.ctc_samples]:
            d,_=self.match(t,self.predict(v));vs.append(float(d["conf"]) if d else 0.0)
        var=variance(vs); return {"variance":var,"score":clamp01(math.exp(-var/.01)),"samples":len(vs),
            "mask_source":t.get("source"),"object_type":t.get("type"),"confidence_samples":{str(t["cls_id"]):vs}}

    def probe(self,img,obj):
        m=self.mask(img.shape,obj);x1,y1,x2,y2=[int(round(x)) for x in obj["bbox"]];h,w=img.shape[:2]
        x1=max(0,min(w-1,x1));x2=max(x1+1,min(w,x2));y1=max(0,min(h-1,y1));y2=max(y1+1,min(h,y2))
        return img[y1:y2,x1:x2].copy(),m[y1:y2,x1:x2].copy()

    def paste(self,img,crop,cm,cx,cy):
        out=img.copy();ph,pw=crop.shape[:2];x1=int(cx-pw/2);y1=int(cy-ph/2);x2=x1+pw;y2=y1+ph;sx1=sy1=0;sx2=pw;sy2=ph
        if x1<0:sx1=-x1;x1=0
        if y1<0:sy1=-y1;y1=0
        if x2>out.shape[1]:sx2-=x2-out.shape[1];x2=out.shape[1]
        if y2>out.shape[0]:sy2-=y2-out.shape[0];y2=out.shape[0]
        if x2<=x1 or y2<=y1:return out
        roi=out[y1:y2,x1:x2];c=crop[sy1:sy2,sx1:sx2];m=cm[sy1:sy2,sx1:sx2]>0;roi[m]=c[m];out[y1:y2,x1:x2]=roi
        return out

    def ftc(self,img,base,obj):
        if not obj:return {"variance":0,"score":0,"disappearance_rate":0,"emergence_rate":0,"samples":0,"samples_detail":[]}
        t=obj[0];crop,cm=self.probe(img,t);h,w=img.shape[:2];rng=random.Random(SEED);conf=[];dis=[];new_counts=[];detail=[]
        pw,ph=crop.shape[1],crop.shape[0]
        for _ in range(self.ftc_samples):
            cx=rng.randint(max(1,pw//2),max(1,w-max(1,pw//2)));cy=rng.randint(max(1,ph//2),max(1,h-max(1,ph//2)))
            aug=self.paste(img,crop,cm,cx,cy)
            expected=[float(cx-pw/2),float(cy-ph/2),float(cx+pw/2),float(cy+ph/2)]
            ds=self.predict(aug); same=[d for d in ds if d["cls_id"]==t["cls_id"]]
            candidates=[]
            for d in same:
                q=iou(expected,d["bbox"]);
                if q>=IOU_MATCH: candidates.append((q,d))
            candidates.sort(key=lambda x:x[0],reverse=True)
            match=candidates[0] if candidates else None
            c=float(match[1]["conf"]) if match else 0.0
            disappeared=1.0 if match is None else 0.0
            new=[]
            for d in same:
                if match is not None and d is match[1]: continue
                baseline_match=max((iou(d["bbox"],b["bbox"]) for b in base if d["cls_id"]==b["cls_id"]),default=0.0)
                if baseline_match<BASELINE_IOU: new.append(d)
            conf.append(c);dis.append(disappeared);new_counts.append(len(new))
            detail.append({"probe_class":t["cls_name"],"probe_expected_bbox":[round(v,2) for v in expected],
                           "probe_conf":c,"detected":bool(match),"probe_match_iou":round(float(match[0]),4) if match else 0.0,
                           "probe_disappeared":bool(disappeared),"new_unmatched_same_class":len(new),
                           "probe_confidence_delta":round(c-float(base[0]["conf"] if base else 0.0),6),
                           "probe_center":[cx,cy]})
        var=variance(conf);dr=float(np.mean(dis));mean_new=float(np.mean(new_counts)) if new_counts else 0.0
        er=float(1.0-np.exp(-mean_new))
        score=clamp01(.55*clamp01(var/.02)+.30*dr+.15*er)
        return {"variance":var,"score":score,"disappearance_rate":dr,"emergence_rate":er,
                "mean_new_unmatched_same_class":mean_new,"samples":len(conf),"probe_source":t.get("source"),
                "matching":{"probe_iou_threshold":IOU_MATCH,"baseline_iou_threshold":BASELINE_IOU},"samples_detail":detail}

    def analyze(self,p,droot=None):
        img=cv2.imread(str(p))
        if img is None:return {"image":str(p),"usable":False,"error":"Unable to read image."}
        det=self.predict(img); lp=self.find_label(p,droot); rec=self.parse_label(lp,img.shape[1],img.shape[0]); obj,src=self.objects(img,det,rec)
        c=self.ctc(img,obj);f=self.ftc(img,det,obj);score=clamp01(.5*c["score"]+.5*f["score"])
        return {"image":str(p),"usable":True,"baseline_detections":len(det),"localization_source":src,"label_file":str(lp) if lp else None,
                "label_annotations":len(rec),"ctc_variance":c["variance"],"ctc_score":c["score"],"ftc_variance":f["variance"],
                "ftc_score":f["score"],"disappearance_rate":f["disappearance_rate"],"emergence_rate":f["emergence_rate"],
                "trace_score":score,"suspicious":score>=.55,"evidence":{"baseline":det,"ctc":c,"ftc":f,"annotations_present":bool(rec),
                "annotation_count":len(rec),"localization_source":src,"threshold":.55}}

    def _collect_images(self, images_dir, max_images=0):
        imgs=sorted(p for p in Path(images_dir).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if max_images>0: imgs=imgs[:max_images]
        return imgs

    def _percentile_rank(self, value, reference, higher_is_more_anomalous=True):
        if not reference: return None
        a=np.asarray(reference,dtype=float)
        if len(a)==1: return 0.5
        # Mid-rank empirical percentile. 0.99 means unusually high when higher_is_more_anomalous=True.
        p=float(np.mean(a <= value))
        return p if higher_is_more_anomalous else 1.0-p

    def calibrate(self, calibration_dir, dataset_root=None, max_images=0):
        imgs=self._collect_images(calibration_dir,max_images)
        results=[self.analyze(p,dataset_root) for p in imgs]
        good=[r for r in results if r.get("usable")]
        ctc=[r["ctc_variance"] for r in good]
        ftc=[r["ftc_score"] for r in good]
        tr=[r["trace_score"] for r in good]
        def stats(v):
            if not v: return {"count":0}
            return {"count":len(v),"min":float(np.min(v)),"median":float(np.median(v)),
                    "p90":float(np.percentile(v,90)),"p95":float(np.percentile(v,95)),
                    "p99":float(np.percentile(v,99)),"max":float(np.max(v))}
        return {"images_checked":len(results),"images_usable":len(good),
                "metrics":{"ctc_variance":stats(ctc),"ftc_score":stats(ftc),"trace_score":stats(tr)},
                "ctc_variance_reference":ctc,"ftc_score_reference":ftc,
                "trace_score_reference":tr,
                "note":"Calibration describes behavioral distributions of the supplied reference images; it does not establish that the model is backdoor-free."}

    def apply_calibration(self, result, calibration, threshold=0.99):
        if not calibration or calibration.get("images_usable",0)<2:
            result["calibrated"]={"available":False}
            result["suspicious"]=False
            return result
        cp=self._percentile_rank(result["ctc_variance"], calibration["ctc_variance_reference"], higher_is_more_anomalous=False)
        fp=self._percentile_rank(result["ftc_score"], calibration["ftc_score_reference"], higher_is_more_anomalous=True)
        # Combined behavioral anomaly. Keep dimensions visible so the reviewer can inspect the cause.
        combined=float(.5*cp+.5*fp)
        result["calibrated"]={"available":True,"ctc_anomaly_percentile":cp,"ftc_anomaly_percentile":fp,
                              "combined_anomaly_percentile":combined,"threshold":threshold,
                              "reference_images":calibration["images_usable"]}
        result["suspicious"]=bool(combined>=threshold)
        result["evidence"]["calibration_threshold"]=threshold
        return result

    def run(self,images_dir,dataset_root=None,max_images=10,calibration_dir=None,calibration_dataset_root=None,
            calibration_images=0,calibration_threshold=0.99,calibration_report=None):
        imgs=self._collect_images(images_dir,max_images)
        yi=self.inspect_yaml(self.find_yaml(images_dir,dataset_root));cc=self.class_check(yi)

        calibration=None
        if calibration_report:
            try:
                calibration=json.loads(Path(calibration_report).read_text())
            except Exception as e:
                calibration={"error":f"Unable to load calibration report: {e}"}
        elif calibration_dir:
            calibration=self.calibrate(calibration_dir,calibration_dataset_root or dataset_root,calibration_images)

        results=[self.analyze(p,dataset_root) for p in imgs]
        if calibration and "ctc_variance_reference" in calibration:
            results=[self.apply_calibration(r,calibration,calibration_threshold) for r in results]
        good=[r for r in results if r.get("usable")]
        scores=[r["trace_score"] for r in good]
        mean=float(np.mean(scores)) if scores else 0
        p90=float(np.percentile(scores,90)) if scores else 0
        sf=sum(bool(r["suspicious"]) for r in good)/len(good) if good else 0
        calibrated=bool(calibration and "ctc_variance_reference" in calibration and calibration.get("images_usable",0)>=2)
        if cc.get("status") == "incompatible":
            disposition = "quarantine"
        elif not calibrated:
            disposition = "review"
        elif sf > 0:
            disposition = "review"
        else:
            disposition = "accept"

        return {"phase":"phase4_model_integrity",
                "method":"TRACE-inspired YOLO behavioral backdoor detection (V5 calibrated IoU-aware)",
                "status":"real","disposition":disposition,
                "calibration_status":"calibrated_reference" if calibrated else "uncalibrated_prototype",
                "calibration":{"used":calibrated,"threshold":calibration_threshold if calibrated else None,
                               "reference_images":calibration.get("images_usable",0) if calibrated else 0,
                               "source":str(calibration_dir) if calibration_dir else calibration_report,
                               "warning":"Calibration is a behavioral baseline, not proof that the model is clean."},
                "model":"best.pt","model_classes":self.class_names,"images_checked":len(results),"images_usable":len(good),
                "reference_data":{"images_required":True,"labels_optional":True,"data_yaml_optional":True,
                "dataset_root":str(dataset_root) if dataset_root else None,"data_yaml":yi,"class_space":cc,
                "images_using_labels":sum(r["localization_source"]=="labels" for r in good),
                "images_using_model_prediction_fallback":sum(r["localization_source"]=="model_prediction" for r in good)},
                "dataset_metrics":{"mean_trace_score":mean,"p90_trace_score":p90,"suspicious_fraction":sf,
                                    "smoke_test_threshold":.55,"calibrated_threshold":calibration_threshold if calibrated else None},
                "results":results,
                "limitations":["TRACE-inspired adaptation, not released TRACE implementation.",
                "The raw TRACE-inspired score is not a probability of backdoor presence.",
                "Calibrated percentiles are relative to the supplied reference images.",
                "A calibration set must be treated as a behavioral baseline, not automatically as known-clean evidence.",
                "For meaningful evaluation, use a held-out calibration/reference split rather than calibrating on the same images being evaluated.",
                "Labels and data.yaml are optional; model predictions provide fallback localization.",
                "Polygon labels provide better masks.","FTC uses spatial/IoU-aware probe matching.",
                "Final security thresholds require clean and known-backdoored validation.",
                "Designed for Ultralytics YOLO-style detectors."]}

def main():
    ap=argparse.ArgumentParser(description="V5 calibrated TRACE-inspired behavioral detector for Ultralytics YOLO models")
    ap.add_argument("--weights",required=True)
    ap.add_argument("--images",required=True,help="Evaluation images")
    ap.add_argument("--dataset-root")
    ap.add_argument("--calibration-root",help="Known/reference calibration image directory")
    ap.add_argument("--calibration-dataset-root",help="Dataset root used to find calibration labels/data.yaml")
    ap.add_argument("--calibration-report",help="Previously generated calibration JSON")
    ap.add_argument("--calibration-images",type=int,default=0,help="Max calibration images; 0 means all")
    ap.add_argument("--calibration-threshold",type=float,default=.99)
    ap.add_argument("--max-images",type=int,default=10)
    ap.add_argument("--ctc-samples",type=int,default=6)
    ap.add_argument("--ftc-samples",type=int,default=12)
    ap.add_argument("--output",default="trace_report_v5.json")
    ap.add_argument("--write-calibration",help="Write the calibration baseline JSON to this path when --calibration-root is used")
    a=ap.parse_args()
    detector=TraceYOLOV5(a.weights,a.ctc_samples,a.ftc_samples)
    r=detector.run(a.images,a.dataset_root,a.max_images,a.calibration_root,a.calibration_dataset_root,
                   a.calibration_images,a.calibration_threshold,a.calibration_report)
    Path(a.output).write_text(json.dumps(r,indent=2))
    if a.write_calibration and a.calibration_root:
        cal=detector.calibrate(a.calibration_root,a.calibration_dataset_root or a.dataset_root,a.calibration_images)
        Path(a.write_calibration).write_text(json.dumps(cal,indent=2))
    print(json.dumps({"disposition":r["disposition"],"calibration_status":r["calibration_status"],
    "images_checked":r["images_checked"],"mean_trace_score":r["dataset_metrics"]["mean_trace_score"],
    "p90_trace_score":r["dataset_metrics"]["p90_trace_score"],"suspicious_fraction":r["dataset_metrics"]["suspicious_fraction"],
    "calibrated_threshold":r["dataset_metrics"]["calibrated_threshold"],
    "images_using_labels":r["reference_data"]["images_using_labels"],
    "images_using_model_prediction_fallback":r["reference_data"]["images_using_model_prediction_fallback"],
    "output":a.output},indent=2))
if __name__=="__main__":main()
