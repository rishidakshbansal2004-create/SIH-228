
import base64
import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

st.set_page_config(
    page_title="TrustCV | Security Grade AI Assurance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

APP = Path(__file__).resolve().parent
RUNTIME = Path(tempfile.gettempdir()) / "TrustCV_SECURITY_GRADE_RUNTIME"
RUNTIME.mkdir(parents=True, exist_ok=True)
AUDIT = RUNTIME / "audit_chain.jsonl"

# ------------------------------------------------------------------
# SECURITY: crypto / immutable-ish local audit
# ------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def audit_event(event):
    prev = ""
    if AUDIT.exists():
        try:
            lines = AUDIT.read_text(encoding="utf-8").splitlines()
            if lines:
                prev = json.loads(lines[-1]).get("chain_hash", "")
        except Exception:
            pass
    record = dict(event)
    record["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["previous_hash"] = prev
    record["chain_hash"] = sha256_bytes(
        json.dumps(record, sort_keys=True).encode("utf-8")
    )
    with AUDIT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

def verify_audit():
    if not AUDIT.exists():
        return True, 0
    prev, count = "", 0
    try:
        for line in AUDIT.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            stored = obj.pop("chain_hash")
            expected = sha256_bytes(
                json.dumps(obj, sort_keys=True).encode("utf-8")
            )
            if obj.get("previous_hash", "") != prev or expected != stored:
                return False, count
            prev, count = stored, count + 1
        return True, count
    except Exception:
        return False, count

# ------------------------------------------------------------------
# SECURITY: signed manifest
# ------------------------------------------------------------------

def verify_signed_manifest(manifest_bytes: bytes, artifact_hash: str):
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        return "FAIL", f"Invalid JSON manifest: {exc}", {}

    expected = str(manifest.get("expected_sha256", "")).lower()
    if not expected:
        return "REVIEW", "Manifest contains no expected_sha256.", manifest
    if expected != artifact_hash.lower():
        return "FAIL", "Manifest SHA-256 does not match uploaded artifact.", manifest

    signature = manifest.get("signature")
    public_pem = manifest.get("public_key_pem")
    if not signature or not public_pem:
        return "REVIEW", "No digital signature/public key supplied.", manifest

    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        pub = serialization.load_pem_public_key(public_pem.encode("utf-8"))
        pub.verify(
            base64.b64decode(signature),
            artifact_hash.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return "PASS", "RSA-PSS/SHA-256 signature valid.", manifest
    except Exception as exc:
        return "FAIL", f"Digital signature verification failed: {exc}", manifest

# ------------------------------------------------------------------
# MODEL SECURITY: NEVER execute an uploaded untrusted pickle first
# ------------------------------------------------------------------

def safe_checkpoint_inspection(path: Path):
    """
    Uses PyTorch weights_only=True only.
    We intentionally do NOT fall back to arbitrary torch.load() for an
    untrusted upload because that may deserialize executable Python objects.
    """
    try:
        import torch
    except Exception as exc:
        return {
            "status": "REVIEW",
            "message": f"PyTorch not available for safe inspection: {exc}",
            "keys": [],
            "parameter_count": None,
            "type": None,
        }

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        return {
            "status": "REVIEW",
            "message": (
                "Safe weights-only inspection rejected this checkpoint. "
                "This does NOT prove the file is corrupted; it may be a "
                "custom/full-object checkpoint requiring its original trusted "
                "architecture. Arbitrary unsafe deserialization was deliberately blocked.\n\n"
                + str(exc)
            ),
            "keys": [],
            "parameter_count": None,
            "type": None,
        }

    info = {
        "status": "PASS",
        "message": "Safe weights-only checkpoint inspection succeeded.",
        "keys": [],
        "parameter_count": None,
        "type": type(obj).__name__,
    }

    if isinstance(obj, dict):
        info["keys"] = [str(k) for k in list(obj.keys())[:40]]
        state = obj.get("state_dict")
        if state is None:
            state = obj.get("model")
        if state is None:
            state = obj.get("ema")

        if isinstance(state, dict):
            vals = [v for v in state.values() if hasattr(v, "numel")]
            if vals:
                info["parameter_count"] = int(sum(v.numel() for v in vals))

    elif hasattr(obj, "state_dict"):
        try:
            state = obj.state_dict()
            vals = [v for v in state.values() if hasattr(v, "numel")]
            if vals:
                info["parameter_count"] = int(sum(v.numel() for v in vals))
        except Exception:
            pass

    return info

def structural_signature(path: Path):
    """
    Non-executing file-level signals. They are evidence, not proof of safety.
    """
    size = path.stat().st_size
    result = {
        "size_bytes": size,
        "sha256": sha256_file(path),
        "extension": path.suffix.lower(),
        "size_ok": size > 1024,
    }

    return result

def load_yolo(path: str):
    from ultralytics import YOLO
    return YOLO(path)

def run_detection(model, image: Image.Image, confidence: float):
    result = model.predict(source=np.asarray(image.convert("RGB")), conf=confidence, verbose=False)[0]
    rows = []
    if result.boxes is not None:
        for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(float, box)
            rows.append({"Object": result.names[int(cls)], "Confidence": round(float(score), 3), "x1": round(x1,1), "y1": round(y1,1), "x2": round(x2,1), "y2": round(y2,1)})
    return Image.fromarray(result.plot()[:, :, ::-1]), pd.DataFrame(rows)

def try_authorized_yolo_load(path: Path):
    """
    Full model execution is only intended after the artifact is authorized.
    The caller must enforce the authorization gate before calling this.
    """
    from ultralytics import YOLO
    return YOLO(str(path))

# ------------------------------------------------------------------
# IMAGE / DATASET ANALYSIS
# ------------------------------------------------------------------

def png_bytes(im):
    b = io.BytesIO()
    im.convert("RGB").save(b, "PNG")
    return b.getvalue()

def features(im):
    a = np.asarray(
        im.convert("RGB").resize((96,96)),
        dtype=np.float32
    ) / 255.0
    g = a.mean(axis=2)
    hist, _ = np.histogram(g, bins=24, range=(0,1))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
    gy, gx = np.gradient(g)
    edge = float(np.mean(np.sqrt(gx*gx + gy*gy)))
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    return np.r_[a.mean((0,1)), a.std((0,1)), g.mean(), g.std(), edge, entropy, hist]

def orientation_screen(im):
    arr = np.asarray(im.convert("RGB").resize((240,240)), dtype=np.float32) / 255.0
    top = arr[:60]; bottom = arr[-60:]
    n = float(top.mean() - bottom.mean())
    nb = float(
        ((top[:,:,2] > top[:,:,0]*1.05) & (top[:,:,2] > top[:,:,1]*0.95)).mean()
        -
        ((bottom[:,:,2] > bottom[:,:,0]*1.05) & (bottom[:,:,2] > bottom[:,:,1]*0.95)).mean()
    )
    normal = n + 0.5*nb
    rot = np.rot90(arr,2)
    rt, rb = rot[:60], rot[-60:]
    rn = float(rt.mean() - rb.mean())
    rbv = float(
        ((rt[:,:,2] > rt[:,:,0]*1.05) & (rt[:,:,2] > rt[:,:,1]*0.95)).mean()
        -
        ((rb[:,:,2] > rb[:,:,0]*1.05) & (rb[:,:,2] > rb[:,:,1]*0.95)).mean()
    )
    rotated = rn + 0.5*rbv
    margin = normal - rotated
    status = "POSSIBLE ROTATION" if margin < -0.08 else "NORMAL / UNCERTAIN"
    return {
        "status": status,
        "normal_score": round(normal,4),
        "rotated_score": round(rotated,4),
        "margin": round(margin,4),
    }

def image_report(im):
    f = features(im)
    flags = []
    if im.width < 224 or im.height < 224:
        flags.append("LOW_RESOLUTION")
    if float(f[7]) < .03:
        flags.append("LOW_CONTRAST")
    orient = orientation_screen(im)
    if orient["status"] == "POSSIBLE ROTATION":
        flags.append("ORIENTATION_ANOMALY")
    return {
        "hash": sha256_bytes(png_bytes(im)),
        "features": f,
        "width": im.width,
        "height": im.height,
        "entropy": round(float(f[8]),3),
        "contrast": round(float(f[7]),3),
        "flags": flags,
        "orientation": orient["status"],
        "orientation_margin": orient["margin"],
        "normal_score": orient["normal_score"],
        "rotated_score": orient["rotated_score"],
    }

def parse_dataset(upload):
    raw = upload.getvalue()
    name = upload.name.lower()
    digest = sha256_bytes(raw)
    if name.endswith(".csv"):
        return {"kind":"CSV","name":upload.name,"hash":digest,
                "df":pd.read_csv(io.BytesIO(raw)),"images":[]}
    if name.endswith((".xlsx",".xls")):
        return {"kind":"Excel","name":upload.name,"hash":digest,
                "df":pd.read_excel(io.BytesIO(raw)),"images":[]}
    if name.endswith(".zip"):
        images=[]; df=None
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for member in z.namelist():
                low=member.lower()
                if low.endswith((".jpg",".jpeg",".png")) and not member.endswith("/"):
                    try:
                        images.append(Image.open(io.BytesIO(z.read(member))).convert("RGB"))
                    except Exception:
                        pass
                elif low.endswith(".csv"):
                    try:
                        df=pd.read_csv(io.BytesIO(z.read(member)))
                    except Exception:
                        pass
        return {"kind":"Image ZIP","name":upload.name,"hash":digest,"df":df,"images":images}
    if name.endswith((".jpg",".jpeg",".png")):
        return {"kind":"Image","name":upload.name,"hash":digest,"df":None,
                "images":[Image.open(io.BytesIO(raw)).convert("RGB")]}
    raise ValueError("Supported: CSV, Excel, ZIP, JPG, JPEG, PNG.")

def screen_dataset(ds):
    rows=[]; reports=[]
    if ds["kind"] in ("Image","Image ZIP"):
        reports=[image_report(x) for x in ds["images"]]
        duplicates=len(reports)-len({r["hash"] for r in reports})
        rotations=sum(r["orientation"]=="POSSIBLE ROTATION" for r in reports)
        rows=[
            ["SHA-256","PASS","Dataset fingerprint"],
            ["File Integrity","PASS","Byte identity recorded"],
            ["Visual Orientation","REVIEW" if rotations else "PASS",f"{rotations} flagged"],
            ["Exact Duplicates","REVIEW" if duplicates else "PASS",f"{duplicates} duplicate(s)"],
        ]
        if len(reports)>=5:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                X=np.vstack([r["features"] for r in reports])
                labels=IsolationForest(n_estimators=300,random_state=42).fit_predict(StandardScaler().fit_transform(X))
                out=int((labels==-1).sum())
                rows.append(["Visual Outlier / Poison Screen","REVIEW" if out else "PASS",f"{out} outlier(s)"])
            except Exception as exc:
                rows.append(["Visual Outlier / Poison Screen","REVIEW",str(exc)])
        else:
            rows.append(["Visual Outlier / Poison Screen","NOT AVAILABLE","Need ≥5 reference images"])
    else:
        df=ds["df"]
        dup=int(df.duplicated().sum())
        label=next((c for c in df.columns if str(c).lower() in {"label","class","target","category"}),None)
        rare=int((df[label].value_counts()<=2).sum()) if label else 0
        rows=[
            ["SHA-256","PASS","Dataset fingerprint"],
            ["File Integrity","PASS","Byte identity recorded"],
            ["Exact Duplicates","REVIEW" if dup else "PASS",f"{dup} duplicate row(s)"],
            ["Rare Label Screen","REVIEW" if rare else "PASS",f"{rare} rare class(es)" if label else "No label column"],
        ]
    return rows,reports

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.markdown("""
<style>
.stApp{background:#06090f}
.block-container{max-width:1580px;padding:1rem 2rem 3rem}
.hero{padding:28px 32px;border-radius:26px;background:linear-gradient(115deg,#071d34,#171c3d,#321525);border:1px solid rgba(120,170,255,.16);box-shadow:0 20px 70px rgba(0,0,0,.35)}
.hero h1{margin:0;font-size:2.75rem;letter-spacing:-.04em}
.ey{font-size:11px;letter-spacing:.2em;color:#93a9c5}
.alert{padding:15px 18px;border-radius:14px;background:rgba(255,55,70,.13);border:1px solid rgba(255,70,82,.38);color:#ff9ea6}
.ok{padding:15px 18px;border-radius:14px;background:rgba(50,225,160,.11);border:1px solid rgba(50,225,160,.25);color:#8af1c8}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="ey">DEFENSIVE AI SECURITY OPERATIONS · MODEL ASSURANCE</div>
<h1>🛡️ TrustCV</h1>
<div>Cryptographic Integrity · Authenticity · Provenance · Safe Inspection · Input Trust · Reliability · Audit</div>
</div>
""",unsafe_allow_html=True)

with st.sidebar:
    page=st.radio(
        "Security Console",
        [
            "🏠 Command Center",
            "01 · Model Verification",
            "02 · Dataset Security",
            "03 · Input & OOD",
            "04 · Reliability",
            "05 · Audit Ledger",
        ]
    )
    confidence=st.slider("Inference confidence",0.10,0.95,0.35,0.05)
    if st.button("↻ Reset Current Run"):
        st.session_state.clear()
        st.rerun()

# persistent intake
st.markdown("### Mission Intake · persistent across all pages")
c1,c2,c3,c4=st.columns([1,1,1,1])
with c1:
    model_upload=st.file_uploader("MODEL · .PT / .PTH",type=["pt","pth"],key="GLOBAL_MODEL")
with c2:
    dataset_upload=st.file_uploader("DATASET · CSV / Excel / ZIP / Image",type=["csv","xlsx","xls","zip","jpg","jpeg","png"],key="GLOBAL_DATASET")
with c3:
    input_upload=st.file_uploader("TEST IMAGE · JPG / PNG",type=["jpg","jpeg","png"],key="GLOBAL_INPUT")
with c4:
    manifest_upload=st.file_uploader("SIGNED MODEL MANIFEST · JSON",type=["json"],key="GLOBAL_MANIFEST")

# model intake
if model_upload is not None:
    raw=model_upload.getvalue()
    h=sha256_bytes(raw)
    if st.session_state.get("uploaded_model_hash")!=h:
        path=RUNTIME/f"model_{h[:16]}_{Path(model_upload.name).name}"
        path.write_bytes(raw)
        static=structural_signature(path)
        inspection=safe_checkpoint_inspection(path)

        st.session_state.update(
            uploaded_model_hash=h,
            model_path=str(path),
            model_name=model_upload.name,
            model_hash=h,
            model_static=static,
            model_inspection=inspection,
        )

        signature_status="REVIEW / NOT PROVIDED"
        signature_reason="Upload a signed manifest to authenticate the model."
        manifest={}
        if manifest_upload is not None:
            signature_status,signature_reason,manifest=verify_signed_manifest(manifest_upload.getvalue(),h)

        st.session_state["model_signature_status"]=signature_status
        st.session_state["model_signature_reason"]=signature_reason
        st.session_state["model_manifest"]=manifest

        # Full model execution is allowed only for a valid signature in this security build.
        authorized = signature_status=="PASS"
        if authorized:
            try:
                st.session_state["model"]=try_authorized_yolo_load(path)
                st.session_state["model_verified"]=True
                st.session_state["model_execution_status"]="AUTHORIZED EXECUTABLE"
                st.session_state.pop("model_error",None)
            except Exception as exc:
                st.session_state["model"]=None
                st.session_state["model_verified"]=False
                st.session_state["model_execution_status"]="AUTHORIZED BUT LOAD FAILED"
                st.session_state["model_error"]=str(exc)
        else:
            st.session_state["model"]=None
            st.session_state["model_verified"]=False
            st.session_state["model_execution_status"]="EXECUTION LOCKED"

        audit_event({
            "event":"model_onboarding",
            "name":model_upload.name,
            "sha256":h,
            "safe_inspection":inspection["status"],
            "signature":signature_status,
            "execution":st.session_state["model_execution_status"],
        })

# manifest reconciliation
# NOTE: Streamlit reruns on every upload. The manifest must be validated
# independently from the model-upload event, otherwise adding the manifest
# after the model leaves the previous "NOT PROVIDED" status in session state.
if manifest_upload is not None and st.session_state.get("model_hash"):
    manifest_raw=manifest_upload.getvalue()
    manifest_h=sha256_bytes(manifest_raw)
    if st.session_state.get("uploaded_manifest_hash")!=manifest_h or st.session_state.get("manifest_bound_model_hash")!=st.session_state.get("model_hash"):
        sig_status,sig_reason,manifest=verify_signed_manifest(manifest_raw,st.session_state["model_hash"])
        st.session_state.update(
            uploaded_manifest_hash=manifest_h,
            manifest_bound_model_hash=st.session_state["model_hash"],
            model_signature_status=sig_status,
            model_signature_reason=sig_reason,
            model_manifest=manifest,
        )
        if sig_status=="PASS" and st.session_state.get("model_path"):
            try:
                st.session_state["model"]=try_authorized_yolo_load(Path(st.session_state["model_path"]))
                st.session_state["model_verified"]=True
                st.session_state["model_execution_status"]="AUTHORIZED EXECUTABLE"
                st.session_state.pop("model_error",None)
            except Exception as exc:
                st.session_state["model"]=None
                st.session_state["model_verified"]=False
                st.session_state["model_execution_status"]="AUTHORIZED BUT LOAD FAILED"
                st.session_state["model_error"]=str(exc)
        elif sig_status!="PASS":
            st.session_state["model"]=None
            st.session_state["model_verified"]=False
            st.session_state["model_execution_status"]="EXECUTION LOCKED"
        audit_event({
            "event":"manifest_validation",
            "manifest_sha256":manifest_h,
            "model_sha256":st.session_state["model_hash"],
            "signature":sig_status,
            "execution":st.session_state.get("model_execution_status"),
        })

# auto-demo local model only when no uploaded model is present
if "model" not in st.session_state and model_upload is None:
    local=None
    for candidate in [APP/"yolo11n.pt",Path.home()/"yolo11n.pt",Path("C:/Users/vasan/yolo11n.pt")]:
        if candidate.exists():
            local=candidate; break
    if local is not None:
        try:
            st.session_state["model"]=load_yolo(str(local))
            st.session_state.update(
                model_verified=True,
                model_name=local.name,
                model_hash=sha256_file(local),
                model_execution_status="LOCAL DEMO BASELINE",
                model_signature_status="TRUSTED LOCAL DEMO"
            )
        except Exception: pass

# dataset intake
if dataset_upload is not None:
    h=sha256_bytes(dataset_upload.getvalue())
    if st.session_state.get("dataset_hash")!=h:
        try:
            ds=parse_dataset(dataset_upload)
            checks,reports=screen_dataset(ds)
            st.session_state.update(dataset=ds,dataset_hash=h,dataset_checks=checks,dataset_reports=reports)
            audit_event({"event":"dataset_loaded","name":ds["name"],"kind":ds["kind"],"sha256":ds["hash"]})
        except Exception as exc:
            st.session_state["dataset_error"]=str(exc)

# input intake
if input_upload is not None:
    h=sha256_bytes(input_upload.getvalue())
    if st.session_state.get("input_hash")!=h:
        try:
            im=Image.open(io.BytesIO(input_upload.getvalue())).convert("RGB")
            st.session_state.update(input=im,input_hash=h,input_report=image_report(im))
            audit_event({"event":"input_loaded","sha256":h})
        except Exception as exc:
            st.session_state["input_error"]=str(exc)

# inference only on authorized/existing trusted local model
if st.session_state.get("model") is not None and "input" in st.session_state:
    try:
        annotated,pred=run_detection(st.session_state["model"],st.session_state["input"],confidence)
        st.session_state.update(annotated=annotated,predictions=pred)
    except Exception as exc:
        st.session_state["inference_error"]=str(exc)

# image OOD
if "dataset" in st.session_state and "input" in st.session_state:
    refs=st.session_state["dataset"].get("images",[])
    if len(refs)>=5:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import IsolationForest
            X=np.vstack([features(x) for x in refs])
            sc=StandardScaler().fit(X)
            detector=IsolationForest(n_estimators=300,random_state=42).fit(sc.transform(X))
            target=features(st.session_state["input"]).reshape(1,-1)
            raw=float(detector.decision_function(sc.transform(target))[0])
            score=float(np.clip((raw+.2)/.4*100,0,100))
            st.session_state["ood_score"]=score
            st.session_state["ood_status"]="IN-DISTRIBUTION" if detector.predict(sc.transform(target))[0]==1 else "OOD DETECTED"
            dist=float(np.mean(np.abs((features(st.session_state["input"])-X.mean(0))/(X.std(0)+1e-6))))
            st.session_state["shift_score"]=dist
            st.session_state["shift_status"]="NO SIGNIFICANT SHIFT" if dist<2 else "SHIFT DETECTED"
        except Exception as exc:
            st.session_state["ood_error"]=str(exc)

dataset=st.session_state.get("dataset")
input_image=st.session_state.get("input")
input_report=st.session_state.get("input_report")
predictions=st.session_state.get("predictions",pd.DataFrame())
ood=st.session_state.get("ood_status","PENDING")
shift=st.session_state.get("shift_status","PENDING")
audit_valid,audit_count=verify_audit()
model_status=st.session_state.get("model_execution_status","NO MODEL")

# ============================================================
# COMMAND CENTER
# ============================================================

if page=="🏠 Command Center":
    st.subheader("Unified AI Security Command Center")
    model_gate="PASS" if st.session_state.get("model_verified") else "LOCKED / REVIEW"
    dataset_review=any(row[1]=="REVIEW" for row in st.session_state.get("dataset_checks",[]))
    visual_gate="REVIEW" if dataset_review else "PASS" if dataset else "PENDING"
    input_gate="REVIEW" if input_report and input_report["flags"] else "PASS" if input_report else "PENDING"

    statuses=[model_gate,visual_gate,input_gate,ood,shift,"VALID" if audit_valid else "TAMPERED"]
    good={"PASS","IN-DISTRIBUTION","NO SIGNIFICANT SHIFT","VALID"}
    trust=int(round(sum(x in good for x in statuses)/len(statuses)*100))
    risk=100-trust

    k1,k2,k3,k4,k5=st.columns(5)
    k1.metric("TRUST SCORE",f"{trust}/100")
    k2.metric("RISK INDEX",f"{risk}/100")
    k3.metric("MODEL",model_status)
    k4.metric("DETECTIONS",len(predictions))
    k5.metric("AUDIT EVENTS",audit_count)

    left,right=st.columns([1.2,1])
    with left:
        st.markdown("#### Security gate matrix")
        st.dataframe(pd.DataFrame({
            "Gate":["Model Identity","Model Authenticity","Model Execution","Dataset Integrity","Visual Anomaly","Input Integrity","OOD","Distribution Shift","Audit"],
            "Status":[
                "PASS" if st.session_state.get("model_hash") else "PENDING",
                st.session_state.get("model_signature_status","REVIEW"),
                model_status,
                "REVIEW" if dataset_review else "PASS" if dataset else "PENDING",
                visual_gate,
                input_gate,
                ood,shift,
                "VALID" if audit_valid else "TAMPERED",
            ],
        }),use_container_width=True,hide_index=True)
    with right:
        st.markdown("#### Threat posture")
        if HAS_PLOTLY:
            fig=go.Figure(go.Indicator(mode="gauge+number",value=risk,title={"text":"RISK INDEX"},gauge={"axis":{"range":[0,100]}}))
            fig.update_layout(height=280,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        if risk>=65:
            st.markdown('<div class="alert">🔴 HIGH RISK · multiple verification gates require review</div>',unsafe_allow_html=True)
        elif risk>=35:
            st.markdown('<div class="alert">🟠 REVIEW REQUIRED · evidence indicates anomaly or missing trust material</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok">🟢 LOW CURRENT RISK · completed prototype gates are clear</div>',unsafe_allow_html=True)

    if dataset and dataset.get("images"):
        st.markdown("#### Dataset anomaly evidence")
        for i,r in enumerate(st.session_state.get("dataset_reports",[])):
            if r["flags"]:
                a,b=st.columns([1,1])
                with a: st.image(dataset["images"][i],caption=f"FLAGGED SAMPLE {i+1}",width=360)
                with b:
                    st.error("🚨 VISUAL ANOMALY")
                    st.write("Orientation:",r["orientation"])
                    st.write("Normal score:",r["normal_score"])
                    st.write("180° score:",r["rotated_score"])
                    st.write("Margin:",r["orientation_margin"])
                    st.write("Flags:",", ".join(r["flags"]))

    if input_image and "annotated" in st.session_state:
        st.markdown("#### Current model inference")
        a,b=st.columns([1.2,1])
        with a: st.image(st.session_state["annotated"],use_container_width=True)
        with b:
            st.dataframe(predictions,use_container_width=True,hide_index=True)
            if HAS_PLOTLY and len(predictions):
                fig=px.bar(predictions,x="Object",y="Confidence",range_y=[0,1],title="Confidence")
                fig.update_layout(height=280,margin=dict(l=10,r=10,t=45,b=10))
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ============================================================
# MODEL VERIFICATION
# ============================================================

elif page=="01 · Model Verification":
    st.subheader("01 · Model Verification — Security Gate")
    if not st.session_state.get("model_name"):
        st.info("Upload a .pt/.pth artifact in Mission Intake.")
    else:
        mhash=st.session_state.get("model_hash","")
        static=st.session_state.get("model_static",{})
        inspect=st.session_state.get("model_inspection",{})
        sig=st.session_state.get("model_signature_status","REVIEW")
        a,b,c,d=st.columns(4)
        a.metric("Artifact",st.session_state["model_name"])
        b.metric("SHA-256","PASS")
        c.metric("Safe Inspection",inspect.get("status","REVIEW"))
        d.metric("Execution",model_status)

        st.code(mhash)
        checks=[
            ["Hash / Identity","PASS","SHA-256 of submitted bytes"],
            ["Known-good hash match","REVIEW / REQUIRED","Compare against approved registry"],
            ["Digital Signature",sig,st.session_state.get("model_signature_reason","")],
            ["Vendor Authentication","PASS" if st.session_state.get("model_manifest",{}).get("vendor") and sig=="PASS" else "REVIEW / REQUIRED","Signed manifest"],
            ["Version Verification","PASS" if st.session_state.get("model_manifest",{}).get("version") and sig=="PASS" else "REVIEW / REQUIRED","Signed manifest"],
            ["Provenance","PASS" if st.session_state.get("model_manifest",{}).get("provenance") and sig=="PASS" else "REVIEW / REQUIRED","Signed manifest"],
            ["Safe Checkpoint Inspection",inspect.get("status","REVIEW"),inspect.get("message","")],
            ["Model Execution Gate",model_status,"Full execution is blocked until the artifact is authorized"],
        ]
        st.dataframe(pd.DataFrame(checks,columns=["Security Check","Status","Evidence"]),use_container_width=True,hide_index=True)

        if inspect.get("keys"):
            st.markdown("#### Static checkpoint evidence")
            st.write("Checkpoint type:",inspect.get("type"))
            st.write("Top-level keys:",", ".join(inspect.get("keys",[])))
            if inspect.get("parameter_count") is not None:
                st.write("Approx. parameters:",f"{inspect['parameter_count']:,}")

        if model_status=="LOCKED / REVIEW" or model_status=="EXECUTION LOCKED":
            st.warning(
                "⚠️ Model execution is intentionally locked. A `.pt` file can contain "
                "pickle objects. TrustCV does not deserialize arbitrary uploaded objects "
                "until an approved signature/manifest authorizes execution."
            )
        if st.session_state.get("model_error"):
            st.error("Authorized model load failed:")
            st.code(st.session_state["model_error"])

# ============================================================
# DATASET
# ============================================================

elif page=="02 · Dataset Security":
    st.subheader("02 · Dataset Security")
    if not dataset:
        st.info("Upload dataset in Mission Intake.")
    else:
        checks=st.session_state.get("dataset_checks",[])
        count=len(dataset["images"]) if dataset["df"] is None else len(dataset["df"])
        a,b,c,d=st.columns(4)
        a.metric("RECORDS",count)
        b.metric("HASH","PASS")
        c.metric("VISUAL SAMPLES",len(dataset["images"]))
        d.metric("REVIEW FLAGS",sum(x[1]=="REVIEW" for x in checks))
        st.code(dataset["hash"])
        st.dataframe(pd.DataFrame(checks,columns=["Check","Status","Evidence"]),use_container_width=True,hide_index=True)
        if any(x[1]=="REVIEW" for x in checks):
            st.error("🔴 DATASET REQUIRES REVIEW")
        else:
            st.success("🟢 DATASET PASSED CURRENT SCREENING")
        reports=st.session_state.get("dataset_reports",[])
        for i,r in enumerate(reports):
            if r["flags"]:
                a,b=st.columns([1,1])
                with a: st.image(dataset["images"][i],caption=f"FLAGGED SAMPLE {i+1}",width=400)
                with b:
                    st.error("🚨 VISUAL ANOMALY")
                    st.write("Orientation:",r["orientation"])
                    st.write("Normal:",r["normal_score"])
                    st.write("180°:",r["rotated_score"])
                    st.write("Margin:",r["orientation_margin"])
                    st.write("Flags:",", ".join(r["flags"]))
        if dataset["df"] is not None:
            st.dataframe(dataset["df"].head(60),use_container_width=True,hide_index=True)

# ============================================================
# INPUT / OOD
# ============================================================

elif page=="03 · Input & OOD":
    st.subheader("03 · Input Verification → OOD → Shift")
    if not input_image:
        st.info("Upload a test image in Mission Intake.")
    else:
        a,b,c,d=st.columns(4)
        a.metric("INPUT HASH","PASS")
        b.metric("ORIENTATION",input_report["orientation"])
        c.metric("OOD",ood)
        d.metric("SHIFT",shift)
        st.image(input_image,width=700)
        st.code(input_report["hash"])
        for flag in input_report["flags"]: st.error("🚨 "+flag)

        if "annotated" in st.session_state:
            st.markdown("#### Inference")
            st.image(st.session_state["annotated"],use_container_width=True)
            st.dataframe(predictions,use_container_width=True,hide_index=True)

        a,b=st.columns(2)
        with a:
            score=st.session_state.get("ood_score")
            st.metric("REFERENCE OOD SCORE",f"{score:.1f}/100" if score is not None else "N/A")
            if ood=="OOD DETECTED": st.error("🚨 OOD DETECTED")
            elif ood=="IN-DISTRIBUTION": st.success("🟢 IN-DISTRIBUTION")
            else: st.info(ood)
        with b:
            dist=st.session_state.get("shift_score")
            st.metric("SHIFT DISTANCE",f"{dist:.2f}" if dist is not None else "N/A")
            if shift=="SHIFT DETECTED": st.warning("⚠️ DISTRIBUTION SHIFT")
            elif shift=="NO SIGNIFICANT SHIFT": st.success("🟢 NO SIGNIFICANT SHIFT")
            else: st.info(shift)

# ============================================================
# RELIABILITY
# ============================================================

elif page=="04 · Reliability":
    st.subheader("04 · Reliability & Behaviour Stability")
    if not input_image or st.session_state.get("model") is None:
        st.info("An authorized/existing executable model and a test image are required.")
    else:
        base=predictions
        base_labels=set(base["Object"]) if len(base) else set()
        rows=[]
        variants=[
            ("Brightness -10%",ImageEnhance.Brightness(input_image).enhance(.90)),
            ("Brightness +10%",ImageEnhance.Brightness(input_image).enhance(1.10)),
            ("Horizontal flip",ImageOps.mirror(input_image)),
        ]
        for name,var in variants:
            try:
                _,dd=run_detection(st.session_state["model"],var,confidence)
                labels=set(dd["Object"]) if len(dd) else set()
                consistency=len(base_labels&labels)/max(1,len(base_labels|labels))
                rows.append([name,round(float(consistency),3),"PASS" if consistency>=.50 else "REVIEW"])
            except:
                rows.append([name,0.0,"REVIEW"])
        rdf=pd.DataFrame(rows,columns=["Transform","Consistency","Status"])
        score=int(round(float(rdf["Consistency"].mean())*100))
        st.session_state["reliability_status"]="PASS" if score>=75 else "REVIEW"
        st.metric("ROBUSTNESS",f"{score}/100")
        st.progress(score/100)
        st.dataframe(rdf,use_container_width=True,hide_index=True)
        if HAS_PLOTLY:
            fig=px.bar(rdf,x="Transform",y="Consistency",range_y=[0,1],title="Prediction stability")
            fig.update_layout(height=300,margin=dict(l=10,r=10,t=45,b=10))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# ============================================================
# AUDIT
# ============================================================

else:
    st.subheader("05 · Tamper-Evident Audit Ledger")
    ok,count=verify_audit()
    if ok: st.success(f"🟢 CHAIN VALID · {count} events")
    else: st.error(f"🔴 CHAIN FAILED · {count} events verified before failure")
    if AUDIT.exists():
        raw=AUDIT.read_text(encoding="utf-8")
        st.dataframe(pd.DataFrame([json.loads(x) for x in raw.splitlines()]),use_container_width=True,hide_index=True)
        st.download_button("Export audit_chain.jsonl",raw,"audit_chain.jsonl","application/jsonl")

st.markdown("---")
st.caption(
    "TrustCV defensive prototype: SHA-256 establishes byte identity; signatures establish "
    "authenticity only when their trust root is managed; safe inspection intentionally blocks "
    "arbitrary pickle deserialization; OOD/visual anomaly/reliability are evidence layers."
)
