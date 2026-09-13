"""
TrustCV API server
-------------------
Wraps this repo's real verification code (crypto_utils, ledger, ood_gate,
inference, phase8_integrity) as HTTP endpoints so the TrustCV React UI can
call genuine checks instead of mocked data.

Nothing in the existing pipeline files is modified -- this only imports
and calls them.

Setup:
    cd trusted_cv_6_9_final
    pip install -r requirements.txt
    pip install fastapi uvicorn python-multipart

Run:
    uvicorn api_server:app --reload --port 8000

The React app expects this server at http://localhost:8000 by default.
Change API_BASE at the top of TrustCV.jsx if you run it elsewhere.

What is and isn't "real" here, so nobody overstates it:
  - Hash Verification, Integrity/Tampering (via hash match), and the audit
    trail (Phase 9 ledger) are genuinely computed from the uploaded bytes.
  - Vendor Authentication, Digital Signature, Version, and Provenance are
    looked up in trusted_registry.json -- a simple local allowlist. This
    repo has no PKI/signing infrastructure, so these checks are only as
    trustworthy as whoever maintains that file. Wire in a real signing
    scheme before treating this as a security boundary.
  - OOD Detection / Distribution Shift (Phase 6) and Inference Integrity
    (Phase 8) run the actual YOLO model and gate logic on the uploaded
    image -- this part is fully real.
  - Backdoor/trigger detection (STRIP, Neural Cleanse -- phase4_trigger/)
    is NOT wired in yet: it requires a matching SmallCNN-format model and
    is slow (loads CIFAR-10, does gradient optimization). It's a natural
    next step for a deeper "Tampering Detection" check on real deployments.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from crypto_utils import sha256_bytes, sha256_file, canonical_json, utc_now
from ledger import AuditLedger
from ood_gate import OODGate
from inference import TrustedDetector
from phase8_integrity import InferenceIntegrity

APP_DIR = Path(__file__).parent
REGISTRY_PATH = APP_DIR / "trusted_registry.json"
DEFAULT_WEIGHTS = APP_DIR / "best.pt"

app = FastAPI(title="TrustCV Verification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying anywhere real
    allow_methods=["*"],
    allow_headers=["*"],
)

ledger = AuditLedger(path=str(APP_DIR / "audit" / "audit_ledger.jsonl"))

# YOLO weights are expensive to load -- do it once, lazily, and reuse.
_detector = None
_gate = None
_integrity = None


def _get_pipeline():
    global _detector, _gate, _integrity
    if _detector is None:
        _detector = TrustedDetector(str(DEFAULT_WEIGHTS))
        _gate = OODGate(_detector)
        _integrity = InferenceIntegrity(_detector)
    return _detector, _gate, _integrity


def _load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"models": {}, "datasets": {}}


def _registry_checks(file_hash: str, kind: str, order: list, labels: dict):
    """Look up file_hash in trusted_registry.json and build the checks[]
    list the UI expects, in the given display order."""
    registry = _load_registry()
    entry = registry.get(kind, {}).get(file_hash)

    checks = [{
        "id": "hash", "label": labels["hash"], "passed": True,
        "detail": f"sha256:{file_hash[:8]}...{file_hash[-6:]} computed",
    }]

    if entry is None:
        not_found = (
            f"sha256:{file_hash[:8]}...{file_hash[-6:]} is not in trusted_registry.json. "
            "Add an entry with vendor/version metadata to pass this check."
        )
        for key in order:
            if key == "hash":
                continue
            checks.append({"id": key, "label": labels[key], "passed": False, "detail": not_found})
    else:
        for key in order:
            if key == "hash":
                continue
            if key == "vendor":
                checks.append({"id": key, "label": labels[key], "passed": True,
                                "detail": f"Vendor: {entry.get('vendor', 'unknown')}"})
            elif key == "signature":
                signed = bool(entry.get("signed"))
                checks.append({"id": key, "label": labels[key], "passed": signed,
                                "detail": "Signature on file in registry" if signed
                                          else "Registry entry has no signature recorded"})
            elif key == "version":
                checks.append({"id": key, "label": labels[key], "passed": True,
                                "detail": f"Version: {entry.get('version', 'unknown')}"})
            elif key == "provenance":
                checks.append({"id": key, "label": labels[key], "passed": True,
                                "detail": entry.get("provenance", "Provenance recorded in registry")})
            elif key == "tamper":
                checks.append({"id": key, "label": labels[key], "passed": True,
                                "detail": "Hash matches registry entry exactly -- no modification detected"})

    checks.sort(key=lambda c: order.index(c["id"]))
    return checks, all(c["passed"] for c in checks)


def _log_event(event: dict):
    event["record_hash_input"] = sha256_bytes(canonical_json(event))
    return ledger.append(event)


@app.post("/api/verify/model")
async def verify_model(file: UploadFile = File(...)):
    data = await file.read()
    file_hash = sha256_bytes(data)
    labels = {"hash": "Model Hash Verification", "vendor": "Vendor Authentication",
              "signature": "Digital Signature", "version": "Version Verification",
              "provenance": "Model Provenance", "tamper": "Tampering Detection"}
    order = ["vendor", "hash", "signature", "version", "provenance", "tamper"]
    checks, passed = _registry_checks(file_hash, "models", order, labels)

    _log_event({"schema_version": "1.0", "timestamp_utc": utc_now(), "type": "model_verification",
                "filename": file.filename, "sha256": file_hash, "passed": passed})
    return {"passed": passed, "checks": checks, "sha256": file_hash}


@app.post("/api/verify/dataset")
async def verify_dataset(file: UploadFile = File(...)):
    data = await file.read()
    file_hash = sha256_bytes(data)
    labels = {"hash": "Hash Verification", "signature": "Digital Signature",
              "provenance": "Provenance", "tamper": "Integrity / Tampering Check"}
    order = ["hash", "signature", "provenance", "tamper"]
    checks, passed = _registry_checks(file_hash, "datasets", order, labels)

    _log_event({"schema_version": "1.0", "timestamp_utc": utc_now(), "type": "dataset_verification",
                "filename": file.filename, "sha256": file_hash, "passed": passed})
    return {"passed": passed, "checks": checks, "sha256": file_hash}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), conf: float = Form(0.25), iou: float = Form(0.45)):
    """Runs Phase 6 (OOD gate) + Phase 7 (YOLO inference) + Phase 8
    (inference integrity) + Phase 9 (audit ledger) on one image, mirroring
    run_pipeline.py's process_frame()."""
    detector, gate, integrity = _get_pipeline()

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    g = gate.check(frame, conf, iou)
    dets = detector.predict(frame, conf=conf, iou=iou) if not g.ood else []
    integrity_result = integrity.evaluate_frame(frame, dets, g, conf, iou, True)

    entry = _log_event({
        "schema_version": "3.1", "timestamp_utc": utc_now(), "source": file.filename,
        "model_sha256": sha256_file(str(DEFAULT_WEIGHTS)) if DEFAULT_WEIGHTS.exists() else None,
        "detections": dets, "phase6_ood": g.to_dict(), "phase8_integrity": integrity_result,
    })

    return {
        "ood": {
            "inDistribution": not g.ood,
            "confidence": round(float(g.mls), 4),
            "detail": ", ".join(g.reasons) if g.reasons else "No OOD signals triggered",
        },
        "shift": {
            "shiftDetected": g.ood,
            "severity": "high" if g.risk_score >= 0.8 else ("medium" if g.ood else "none"),
            "detail": f"Risk score {g.risk_score:.2f} (gate threshold {gate.risk_threshold:.2f})",
        },
        "detections": dets,
        "integrity": integrity_result,
        "record_hash": entry["record_hash"],
    }


@app.get("/api/ledger/verify")
def verify_ledger():
    ok, msg = ledger.verify()
    return {"valid": ok, "message": msg}


@app.get("/api/health")
def health():
    return {"status": "ok"}
