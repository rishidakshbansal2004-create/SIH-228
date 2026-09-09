import argparse,json,time
from pathlib import Path
import cv2
from ood_gate import OODGate
from inference import TrustedDetector
from phase8_integrity import InferenceIntegrity
from ledger import AuditLedger
from crypto_utils import sha256_file,sha256_bytes,canonical_json,utc_now

def process_frame(frame, source_name, gate, detector, integrity, ledger, args):
    # PHASE 6
    g=gate.check(frame,args.conf,args.iou)
    # PHASE 7 (the gate internally performs a lightweight model probe; this is
    # the full trusted inference after the gate passes.)
    dets=[]
    if not g.ood or args.allow_ood:
        dets=detector.predict(frame,conf=args.conf,iou=args.iou)
    # PHASE 8
    integrity_result=integrity.evaluate_frame(frame,dets,g,args.conf,args.iou,not args.no_robustness)
    # PHASE 9
    event={'schema_version':'3.0','timestamp_utc':utc_now(),'source':source_name,
           'model_sha256':sha256_file(args.weights) if Path(args.weights).exists() else None,
           'detections':dets,'phase6_ood':g.to_dict(),'phase8_integrity':integrity_result}
    event['inference_record_sha256']=sha256_bytes(canonical_json(event))
    entry=ledger.append(event)
    return g,dets,integrity_result,entry

def draw(frame,g,dets,ir):
    out=frame.copy()
    for d in dets:
        x1,y1,x2,y2=map(int,d['bbox_xyxy'])
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(out,f"{d['class_name']} {d['confidence']:.2f}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
    lines=[f"PHASE 6 OOD: {g.status}  risk={g.risk_score:.2f}",
           f"PHASE 7 detections: {len(dets)}",
           f"PHASE 8 integrity: {ir['status']}  score={ir['reliability_score']:.2f}",
           f"Cal {ir['components']['calibration']:.2f}  Robust {ir['components']['robustness']:.2f}  Conf {ir['components']['confidence']:.2f}",
           f"Shift consistency {ir['components']['ood_distribution_consistency']:.2f}"]
    y=28
    for line in lines:
        cv2.putText(out,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2); y+=24
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image');p.add_argument('--camera',type=int);p.add_argument('--weights',required=True)
    p.add_argument('--allow-ood',action='store_true');p.add_argument('--no-robustness',action='store_true')
    p.add_argument('--conf',type=float,default=.25);p.add_argument('--iou',type=float,default=.45)
    a=p.parse_args()
    if a.image is None and a.camera is None:p.error('use --image or --camera')
    detector=TrustedDetector(a.weights); gate=OODGate(detector); integrity=InferenceIntegrity(detector); ledger=AuditLedger()
    if a.image:
        frame=cv2.imread(a.image)
        if frame is None: raise FileNotFoundError(a.image)
        g,d,ir,e=process_frame(frame,a.image,gate,detector,integrity,ledger,a)
        print('\n[PHASE 6]');print(json.dumps(g.to_dict(),indent=2))
        print('\n[PHASE 7]');print(json.dumps(d,indent=2))
        print('\n[PHASE 8]');print(json.dumps(ir,indent=2))
        print(f"\n[PHASE 9] record_hash={e['record_hash']}")
        return
    cap=cv2.VideoCapture(a.camera)
    if not cap.isOpened():raise RuntimeError('Could not open camera')
    print('Webcam running. Press Q to quit.')
    last=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            g,d,ir,e=process_frame(frame,'webcam',gate,detector,integrity,ledger,a)
            vis=draw(frame,g,d,ir); cv2.imshow('Trusted CV — Phases 6-9',vis)
            if time.time()-last>2:
                print(f"Phase6={g.status} risk={g.risk_score:.2f} | Phase8={ir['status']} score={ir['reliability_score']:.2f} | record={e['record_hash'][:12]}...")
                last=time.time()
            if cv2.waitKey(1)&0xFF in (ord('q'),27):break
    finally:
        cap.release();cv2.destroyAllWindows()
        print('Webcam stopped.')
if __name__=='__main__':main()
