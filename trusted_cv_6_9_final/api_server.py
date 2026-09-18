"""TrustCV API server: MIRAD -> Phase 4 -> Phase 6-9 integration.

Round-1 integration policy:
- MIRAD owns model identity/reference verification.
- Model type is detected before Phase 4.
- SmallCNN: real Phase 4 (Neural Cleanse + optional STRIP/ablation), Phase 6-9 placeholders.
- YOLO: real MIRAD Phase 3, Phase 4 placeholder, existing YOLO Phase 6-9 path.
- No trusted_registry.json is used for model verification.
"""

import json
from pathlib import Path
from typing import Any
import tempfile

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

from crypto_utils import sha256_bytes, sha256_file, canonical_json, utc_now, mirad_hash_object
from ledger import AuditLedger
from ood_gate import OODGate
from inference import TrustedDetector
from phase8_integrity import InferenceIntegrity
from dataset_adapter import load_dataset_from_zip, dataset_summary, validate_dataset_for_model
from dataset_compatibility import inspect_yolo_model, inspect_reference_zip, validate_yolo_reference
from trace_detector_v5_fixed import TraceYOLOV5

from mirad.security.api import TrustAnchorStore
from mirad.security.audit import verify_audit_chain
from mirad.security.evidence import submit_external_evidence
from mirad.security.findings import FindingSeverity, RecommendedDisposition, create_finding, secure_finding
from mirad.security.identity import ArtifactIdentity, ArtifactType, RegistrationStatus
from mirad.security.provenance import create_provenance
from mirad.security.registration import register_artifact
from mirad.security.replay import ReplayStateStore, check_replay
from mirad.security.verification import verify_artifact

try:
    from phase4_model_integrity import run_phase4
    _PHASE4_AVAILABLE = True
except Exception as exc:
    _PHASE4_AVAILABLE = False
    _PHASE4_IMPORT_ERROR = str(exc)

try:
    from poison_and_train import SmallCNN
except Exception:
    SmallCNN = None

APP_DIR = Path(__file__).parent
RUNTIME_DIR = APP_DIR / "runtime_models"
RUNTIME_DIR.mkdir(exist_ok=True)
DEFAULT_WEIGHTS = APP_DIR / "best.pt"

app = FastAPI(title="TrustCV Verification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ledger = AuditLedger(path=str(APP_DIR / "audit" / "mirad_audit_ledger.jsonl"))
MIRAD_STORE = TrustAnchorStore()
REPLAY_STORE = ReplayStateStore()
RUNS: dict[str, dict[str, Any]] = {}

_detector_cache: dict[str, tuple[TrustedDetector, OODGate, InferenceIntegrity]] = {}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _new_run(filename: str, model_type: str, digest: str, model_path: Path, phase3: dict[str, Any] | None = None) -> str:
    import uuid
    run_id = f"tcv-{uuid.uuid4().hex[:16]}"
    RUNS[run_id] = {
        "run_id": run_id,
        "filename": filename,
        "model_type": model_type,
        "model_sha256": digest,
        "model_path": str(model_path),
        "phase3": phase3,
        "phase4": None,
        "phase6": None,
        "phase7": None,
        "phase8": None,
        "phase9": None,
        "provenance_records": [],
        "findings": [],
        "evidence": [],
        "checkpoints": {},
    }
    _log_event({
        "type": "model_upload",
        "phase": "model",
        "event_id": f"{run_id}:upload",
        "nonce": uuid.uuid4().hex,
        "context": run_id,
        "model_sha256": digest,
        "filename": filename,
        "model_type": model_type,
    }, run_id=run_id, event_type="model.upload")
    return run_id


def _log_event(event: dict[str, Any], *, run_id: str | None = None, event_type: str | None = None) -> dict[str, Any]:
    import uuid
    event = dict(event)
    rid = run_id or event.get("run_id") or "legacy"
    event.setdefault("event_id", f"{rid}:{uuid.uuid4().hex[:12]}")
    event.setdefault("nonce", uuid.uuid4().hex)
    event.setdefault("context", rid)
    record = ledger.append(event, run_id=rid, event_type=event_type or str(event.get("type") or "trustcv.event"))
    return record


def _checkpoint_model(run_id: str, phase: str, model_path: Path, expected_digest: str) -> dict[str, Any]:
    """Re-hash the actual model file and verify it against the MIRAD trust reference."""
    import uuid
    actual_hex = sha256_file(model_path)
    actual = f"sha256:{actual_hex}"
    expected = expected_digest if str(expected_digest).startswith("sha256:") else f"sha256:{expected_digest}"
    filename = RUNS.get(run_id, {}).get("filename", model_path.name)
    model_type = RUNS.get(run_id, {}).get("model_type", "unknown")
    artifact_id = _artifact_id(filename)
    candidate = ArtifactIdentity(
        artifact_type=ArtifactType.MODEL, artifact_id=artifact_id, version="1.0.0",
        artifact_digest=actual, manifest_digest=None, format=_model_format(filename),
    )
    verification = verify_artifact(candidate=candidate, trust_store=MIRAD_STORE, artifact_path=model_path)
    hash_match = actual == expected
    verified = bool(hash_match and verification.valid)
    checkpoint = {
        "phase": phase,
        "run_id": run_id,
        "verified": verified,
        "expected_model_sha256": expected,
        "actual_model_sha256": actual,
        "model_modified": not hash_match,
        "mirad_status": verification.status.value,
        "mirad_valid": verification.valid,
        "failure_codes": verification.failure_codes,
        "timestamp_utc": utc_now(),
    }
    record = _log_event({
        "type": "model_hash_checkpoint",
        "phase": phase,
        "event_id": f"{run_id}:{phase}:checkpoint:{uuid.uuid4().hex[:8]}",
        "nonce": uuid.uuid4().hex,
        "context": run_id,
        "model_sha256": actual_hex,
        "expected_model_sha256": expected_digest.replace("sha256:", ""),
        "verified": verified,
        "mirad_verification": verification.to_dict(),
    }, run_id=run_id, event_type=f"{phase}.hash_checkpoint")
    checkpoint["audit_event_id"] = record["audit_id"]
    checkpoint["audit_event_hash"] = record["current_hash"]
    RUNS.setdefault(run_id, {}).setdefault("checkpoints", {})[phase] = checkpoint
    return checkpoint


def _build_provenance_for_output(run_id: str, *, frame_index: int | None, input_digest: str, output_obj: dict[str, Any], dataset_digest: str | None, dataset_identity: str, phase6: dict[str, Any], phase7: dict[str, Any], phase8: dict[str, Any], model_path: Path) -> dict[str, Any]:
    import uuid
    run = RUNS[run_id]
    sequence = len(run["provenance_records"]) + 1
    event_id = f"{run_id}:prov:{sequence:04d}"
    nonce = uuid.uuid4().hex
    output_digest = mirad_hash_object(output_obj)
    preprocessing_digest = mirad_hash_object({"pipeline": "TrustCV", "phase": "6-8", "preprocessing": "decoder + detector preprocessing"})
    inference_config_digest = mirad_hash_object({"conf": output_obj.get("conf_threshold", 0.25), "iou": output_obj.get("iou_threshold", 0.45)})
    record = create_provenance(
        event_id=event_id, nonce=nonce, sequence=sequence, context=run_id,
        input_digest=input_digest if input_digest.startswith("sha256:") else f"sha256:{input_digest}",
        dataset_identity=dataset_identity or "reference-data:not-provided",
        dataset_digest=(dataset_digest if dataset_digest and dataset_digest.startswith("sha256:") else f"sha256:{dataset_digest}" if dataset_digest else "sha256:" + "0"*64),
        model_identity=_artifact_id(run["filename"]),
        model_digest=f"sha256:{sha256_file(model_path)}",
        preprocessing_digest=preprocessing_digest,
        inference_config_digest=inference_config_digest,
        output_digest=output_digest,
        metadata={"run_id": run_id, "frame_index": frame_index, "phase6": phase6, "phase7": phase7, "phase8": phase8},
    )
    replay = check_replay(record, store=REPLAY_STORE, freshness_window_seconds=300, allowed_skew_seconds=30, scope=f"provenance:{run_id}", expected_context=run_id)
    record["replay_verification"] = {"valid": replay.valid, "status": replay.status, "checks": replay.checks, "failure_code": replay.failure_code, "reason": replay.reason}
    run["provenance_records"].append(record)
    return record


def _make_evidence_and_finding(run_id: str, *, frame_index: int | None, input_digest: str, phase6: dict[str, Any], phase7: dict[str, Any], phase8: dict[str, Any], provenance_id: str, audit_event_id: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    run = RUNS[run_id]
    integrity = phase8.get("integrity") if isinstance(phase8, dict) else phase8
    integrity = integrity if isinstance(integrity, dict) else {}
    p8d = str(integrity.get("disposition") or phase8.get("disposition") or "").lower()
    ood = phase6.get("inDistribution") is False
    dets = phase7.get("detections") or []
    candidate = ood or p8d in {"quarantine", "blocked", "reject", "rejected", "fail", "failed", "flagged", "review"}
    if not candidate:
        return None, None
    evidence = submit_external_evidence(
        evidence_id=f"{run_id}:evidence:{len(run['evidence'])+1:04d}",
        evidence_type="inference_integrity_candidate", producer="TrustCV", producer_version="1.0",
        method="Phase6+Phase7+Phase8", affected_asset=f"frame:{frame_index}" if frame_index is not None else "input",
        asset_type="IMAGE", asset_version="1.0", asset_digest=input_digest if input_digest.startswith("sha256:") else f"sha256:{input_digest}",
        evidence_payload={"phase6": phase6, "phase7": phase7, "phase8": phase8},
        confidence=float(integrity.get("confidence") or 0.0),
        severity="HIGH" if p8d in {"quarantine", "blocked", "fail", "failed", "flagged"} or ood else "MEDIUM",
        context=run_id, provenance_ref=provenance_id,
        limitations=["Evidence identifies an anomaly/integrity candidate; it does not by itself establish malicious intent."]
    )
    ev = evidence.to_dict(); run["evidence"].append(ev)
    severity = FindingSeverity.HIGH if ev.get("severity") == "HIGH" else FindingSeverity.MEDIUM
    finding = secure_finding(create_finding(
        finding_id=f"{run_id}:finding:{len(run['findings'])+1:04d}",
        finding_type="ANOMALOUS_OUTPUT_CANDIDATE", affected_asset=f"frame:{frame_index}" if frame_index is not None else "input",
        affected_asset_type="IMAGE", asset_version="1.0", asset_digest=ev["asset_digest"],
        reason="Phase 6/8 evidence produced an integrity or distribution signal; review required.",
        evidence=ev, confidence=float(ev.get("confidence") or 0.0), severity=severity,
        recommended_disposition=RecommendedDisposition.QUARANTINE if severity == FindingSeverity.HIGH else RecommendedDisposition.REVIEW,
        limitations=["No claim of malicious intent is made from this evidence alone."], provenance_id=provenance_id, audit_event_id=audit_event_id,
    ))
    fd = finding.to_dict(); run["findings"].append(fd)
    return ev, fd


def _artifact_id(filename: str) -> str:
    stem = Path(filename or "model").stem.lower()
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return f"model:{safe or 'uploaded'}"


def _model_format(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pt", ".pth", ".bin"}:
        return "pytorch"
    return suffix.lstrip(".") or "unknown"


def _is_smallcnn_state_dict(obj: Any) -> bool:
    if SmallCNN is None:
        return False
    state = obj
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state = obj["model"]
    if not isinstance(state, dict) or not state:
        return False
    keys = list(state.keys())
    if not any(isinstance(k, str) for k in keys):
        return False
    try:
        model = SmallCNN(num_classes=10)
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        return False


def identify_model_type(path: Path) -> tuple[str, str]:
    """Return (model_type, evidence)."""
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return "onnx", "ONNX file extension detected"

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if _is_smallcnn_state_dict(obj):
            return "smallcnn", "Checkpoint loads strictly into the project's SmallCNN(num_classes=10)"

        if isinstance(obj, dict):
            candidate = obj.get("model")
            if candidate is not None:
                name = type(candidate).__name__.lower()
                module = type(candidate).__module__.lower()
                if "ultralytics" in module or "yolo" in name or hasattr(candidate, "names"):
                    return "yolo", "PyTorch checkpoint contains a YOLO/Ultralytics model object"

        try:
            from ultralytics import YOLO
            YOLO(str(path))
            return "yolo", "Checkpoint accepted by Ultralytics YOLO loader"
        except Exception:
            pass
    except Exception as exc:
        return "unknown", f"PyTorch inspection failed: {exc}"

    return "unknown", "No supported SmallCNN or YOLO signature detected"


def _mirad_model_check(path: Path, filename: str, model_type: str) -> dict[str, Any]:
    digest = sha256_file(path)
    artifact_id = _artifact_id(filename)
    version = "1.0.0"
    fmt = _model_format(filename)

    candidate = ArtifactIdentity(
        artifact_type=ArtifactType.MODEL,
        artifact_id=artifact_id,
        version=version,
        artifact_digest=f"sha256:{digest}",
        manifest_digest=None,
        format=fmt,
    )

    reference = MIRAD_STORE.get_artifact_reference(artifact_id, version)
    first_seen = reference is None

    if first_seen:
        registration = register_artifact(
            trust_store=MIRAD_STORE,
            artifact_type=ArtifactType.MODEL,
            artifact_id=artifact_id,
            version=version,
            format=fmt,
            artifact_path=path,
            status=RegistrationStatus.APPROVED,
            trust_metadata={
                "model_type": model_type,
                "registration_reason": "first_seen_reference_for_assessment",
                "safety_status": "not_established_by_registration",
            },
        )
        reference = registration

    result = verify_artifact(
        candidate=candidate,
        trust_store=MIRAD_STORE,
        artifact_path=path,
    )

    checks = [
        {
            "id": "hash",
            "label": "Model Hash Verification",
            "passed": bool(result.checks.get("artifact_digest")),
            "detail": f"sha256:{digest[:12]}...{digest[-8:]}",
        },
        {
            "id": "identity",
            "label": "MIRAD Artifact Identity",
            "passed": bool(result.checks.get("artifact_identity")),
            "detail": f"{artifact_id} · version {version}",
        },
        {
            "id": "registration",
            "label": "MIRAD Trust Reference",
            "passed": bool(result.checks.get("registration_status")),
            "detail": "Reference registered for assessment; registration does not establish model safety.",
        },
        {
            "id": "provenance",
            "label": "Model Provenance",
            "passed": None,
            "detail": "No signed provenance record was supplied with the uploaded model; provenance verification is unavailable for this upload.",
        },
        {
            "id": "signature",
            "label": "Digital Signature",
            "passed": None,
            "detail": "No signed model/provenance artifact was supplied; signature verification is unavailable for this upload.",
        },
    ]

    return {
        "mirad": result.to_dict(),
        "checks": checks,
        "passed": bool(result.valid),
        "first_seen": first_seen,
        "artifact_identity": candidate.to_dict(),
    }


def _phase4_smallcnn(
    path: Path,
    access_level: str,
    reference_images: torch.Tensor,
    reference_info: dict[str, Any],
) -> dict[str, Any]:
    if not _PHASE4_AVAILABLE:
        return {
            "phase": "phase4_model_integrity",
            "status": "unavailable",
            "disposition": "review",
            "access_level": access_level,
            "flags": [],
            "reference_data": reference_info,
            "limitations": [f"Phase 4 import failed: {_PHASE4_IMPORT_ERROR}"],
        }
    try:
        report = run_phase4(
            str(path),
            access_level,
            reference_images=reference_images,
            run_strip_in_white_box=True,
        )
        return report
    except Exception as exc:
        return {
            "phase": "phase4_model_integrity",
            "status": "error",
            "disposition": "review",
            "access_level": access_level,
            "flags": [],
            "reference_data": reference_info,
            "limitations": [f"Phase 4 execution failed: {exc}"],
        }


def _phase4_yolo_mock() -> dict[str, Any]:
    return {
        "phase": "phase4_model_integrity",
        "status": "placeholder",
        "disposition": "review",
        "access_level": "white_box",
        "flags": [
            {
                "check": "yolo_trigger_detection",
                "disposition": "review",
                "confidence": None,
                "reason": "YOLO-specific trigger/backdoor detector is not implemented yet.",
                "evidence": {},
                "limitations": ["YOLO trigger analysis will be added later."],
            }
        ],
        "limitations": ["Phase 4 is a declared placeholder for YOLO models in Round 1."],
    }


def _smallcnn_phase6_9_mock(model_digest: str) -> dict[str, Any]:
    return {
        "phase6": {
            "status": "placeholder",
            "inDistribution": None,
            "confidence": None,
            "detail": "SmallCNN Phase 6 implementation is reserved for the next integration stage.",
        },
        "phase7": {
            "status": "placeholder",
            "detail": "SmallCNN classification inference is reserved for the next integration stage.",
        },
        "phase8": {
            "status": "placeholder",
            "disposition": "review",
            "detail": "SmallCNN Phase 8 reliability/inference-integrity implementation is reserved for the next integration stage.",
        },
        "phase9": {
            "status": "placeholder",
            "detail": "SmallCNN Phase 9 audit integration is reserved for the next integration stage.",
            "model_sha256": model_digest,
        },
    }


def _get_yolo_pipeline(weights_path: Path):
    key = sha256_file(weights_path)
    if key not in _detector_cache:
        detector = TrustedDetector(str(weights_path))
        gate = OODGate(detector)
        integrity = InferenceIntegrity(detector)
        _detector_cache[key] = (detector, gate, integrity)
    return _detector_cache[key]




@app.get("/api/report/pdf/download/{filename}")
async def download_report_pdf(filename: str):
    """Download a PDF previously generated by /api/report/pdf."""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".pdf"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid report filename.")

    report_path = RUNTIME_DIR / safe_name
    if not report_path.is_file():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Report not found. Generate the report again.")

    return FileResponse(path=str(report_path), media_type="application/pdf", filename=safe_name)


@app.post("/api/verify/model/phase3")
async def verify_model_phase3(
    file: UploadFile = File(...),
    reference_dataset: UploadFile | None = File(None),
    access_level: str = Form("white_box"),
):
    """Run only Phase 3: model identity, type detection, and MIRAD verification."""
    data = await file.read()
    digest = sha256_bytes(data)
    filename = file.filename or "uploaded_model.pt"
    path = RUNTIME_DIR / f"{digest}{Path(filename).suffix.lower() or '.pt'}"
    path.write_bytes(data)

    model_type, type_evidence = identify_model_type(path)

    if model_type == "unknown":
        path.unlink(missing_ok=True)
        return _jsonable({
            "passed": False,
            "phase": "phase3_model_identity",
            "model_type": "unknown",
            "sha256": digest,
            "filename": filename,
            "checks": [{
                "id": "model_type",
                "label": "Model Type Detection",
                "passed": False,
                "detail": type_evidence,
            }],
            "error": "Unsupported or unrecognized model format.",
        })

    mirad = _mirad_model_check(path, filename, model_type)

    reference_info = None
    if reference_dataset is not None:
        dataset_bytes = await reference_dataset.read()
        if not dataset_bytes:
            return _jsonable({
                "passed": False,
                "phase": "phase3_model_identity",
                "model_type": model_type,
                "sha256": digest,
                "filename": filename,
                "mirad": mirad,
                "phase3": mirad,
                "checks": mirad["checks"],
                "error": "The uploaded reference dataset is empty.",
            })

        suffix = Path(reference_dataset.filename or "reference.zip").suffix.lower() or ".zip"
        with tempfile.NamedTemporaryFile(
            prefix="trustcv_reference_",
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(dataset_bytes)
            dataset_path = Path(tmp.name)

        try:
            dataset_result = load_dataset_from_zip(
                dataset_path,
                max_samples=None,
            )
            reference_info = dataset_summary(dataset_result)
        except Exception as exc:
            return _jsonable({
                "passed": False,
                "phase": "phase3_model_identity",
                "model_type": model_type,
                "sha256": digest,
                "filename": filename,
                "mirad": mirad,
                "phase3": mirad,
                "checks": mirad["checks"],
                "error": f"Reference dataset could not be loaded: {exc}",
            })
        finally:
            dataset_path.unlink(missing_ok=True)

    run_id = _new_run(filename, model_type, digest, path)
    phase3_checkpoint = _checkpoint_model(run_id, "phase3", path, digest)
    phase3_payload = {
        "passed": bool(mirad["passed"]),
        "phase": "phase3_model_identity",
        "model_type": model_type,
        "model_type_evidence": type_evidence,
        "filename": filename,
        "sha256": digest,
        "model_ref": digest,
        "mirad": mirad,
        "phase3": mirad,
        "reference_data": reference_info,
        "access_level": access_level,
        "run_id": run_id,
        "hash_checkpoint": phase3_checkpoint,
    }
    RUNS[run_id]["phase3"] = phase3_payload
    _log_event({
        "type": "phase3.mirad_verification", "phase": "phase3",
        "run_id": run_id, "model_sha256": digest,
        "mirad": mirad, "hash_checkpoint": phase3_checkpoint,
    }, run_id=run_id, event_type="phase3.mirad_verification")
    return _jsonable(phase3_payload)


@app.post("/api/verify/dataset/compatibility")
async def verify_dataset_compatibility(
    file: UploadFile = File(...),
    reference_dataset: UploadFile = File(...),
):
    """Validate that a supplied clean reference dataset matches the uploaded model."""
    model_bytes = await file.read()
    dataset_bytes = await reference_dataset.read()
    filename = file.filename or "uploaded_model.pt"
    dataset_filename = reference_dataset.filename or "reference.zip"

    if not model_bytes:
        return _jsonable({"compatible": False, "disposition": "blocked", "error": "Uploaded model is empty."})
    if not dataset_bytes:
        return _jsonable({"compatible": False, "disposition": "blocked", "error": "Reference dataset is empty."})

    model_path = None
    dataset_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="trustcv_compat_model_", suffix=Path(filename).suffix.lower() or ".pt", delete=False) as tmp:
            tmp.write(model_bytes)
            model_path = Path(tmp.name)
        with tempfile.NamedTemporaryFile(prefix="trustcv_compat_data_", suffix=Path(dataset_filename).suffix.lower() or ".zip", delete=False) as tmp:
            tmp.write(dataset_bytes)
            dataset_path = Path(tmp.name)

        model_type, type_evidence = identify_model_type(model_path)
        if model_type == "yolo":
            model_info = inspect_yolo_model(model_path)
            dataset_info = inspect_reference_zip(dataset_path)
            compatibility = validate_yolo_reference(model_info, dataset_info)
        elif model_type == "smallcnn":
            dataset_result = load_dataset_from_zip(dataset_path, max_samples=None)
            dataset_info = dataset_summary(dataset_result)
            compatibility = validate_dataset_for_model(
                dataset_result,
                expected_channels=3,
                expected_height=32,
                expected_width=32,
                expected_classes=10,
            )
            compatibility = {
                "compatible": compatibility["compatible"],
                "disposition": "compatible" if compatibility["compatible"] else "blocked",
                "checks": [
                    {"id": "task", "passed": True, "detail": "SmallCNN image-classification testbed"},
                    {"id": "dataset_format", "passed": True, "detail": dataset_info.get("format")},
                    {"id": "shape_and_channels", "passed": compatibility["compatible"], "detail": "; ".join(compatibility.get("errors", [])) or "3-channel 32×32 reference images"},
                    {"id": "class_space", "passed": compatibility["compatible"], "detail": "10-class reference space"},
                ],
                "errors": compatibility.get("errors", []),
                "warnings": [],
            }
            model_info = {"framework": "project SmallCNN", "task": "image_classification", "num_classes": 10, "class_names": []}
        else:
            model_info = {"task": "unsupported", "num_classes": None, "class_names": []}
            dataset_info = inspect_reference_zip(dataset_path)
            compatibility = {"compatible": False, "disposition": "blocked", "checks": [], "errors": [f"No compatibility adapter is available for model type '{model_type}'."], "warnings": []}

        return _jsonable({
            "phase": "dataset_compatibility",
            "model_type": model_type,
            "model_type_evidence": type_evidence,
            "model": model_info,
            "dataset": dataset_info,
            "compatibility": compatibility,
            "passed": bool(compatibility.get("compatible")),
        })
    except Exception as exc:
        return _jsonable({
            "phase": "dataset_compatibility",
            "compatible": False,
            "disposition": "blocked",
            "error": f"Compatibility inspection failed: {exc}",
        })
    finally:
        if model_path:
            model_path.unlink(missing_ok=True)
        if dataset_path:
            dataset_path.unlink(missing_ok=True)



def _safe_extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a user-supplied reference ZIP without allowing path traversal."""
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("Reference dataset contains an unsafe ZIP path.")
        zf.extractall(dest)
    return dest


def _collect_reference_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _copy_demo_split(source_root: Path, split_root: Path, images: list[Path], selected: list[Path]) -> None:
    """Copy selected images plus matching YOLO labels/data.yaml for the demo split."""
    import shutil
    split_root.mkdir(parents=True, exist_ok=True)
    out_images = split_root / "images"
    out_labels = split_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for src in selected:
        shutil.copy2(src, out_images / src.name)
        # Locate a matching label in common YOLO layouts.
        candidates = []
        parts = list(src.parts)
        if "images" in parts:
            idx = parts.index("images")
            candidates.append(Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt"))
        candidates += [src.parent / "labels" / f"{src.stem}.txt", source_root / "labels" / f"{src.stem}.txt"]
        label = next((x for x in candidates if x.exists()), None)
        if label:
            shutil.copy2(label, out_labels / label.name)

    # Copy the authoritative YAML when present, including common nested train/valid layouts.
    yaml_candidates = [source_root / "data.yaml", source_root / "data.yml"]
    yaml_candidates += sorted(source_root.rglob("data.yaml")) + sorted(source_root.rglob("data.yml"))
    yaml_file = next((candidate for candidate in yaml_candidates if candidate.is_file()), None)
    if yaml_file:
        shutil.copy2(yaml_file, split_root / yaml_file.name)


def _prepare_yolo_demo_split(dataset_zip: Path, work_root: Path, calibration_count: int = 15, evaluation_count: int = 7):
    extracted = _safe_extract_zip(dataset_zip, work_root / "reference")
    images = _collect_reference_images(extracted)
    if len(images) < calibration_count + evaluation_count:
        raise ValueError(
            f"Quick Phase 4 requires at least {calibration_count + evaluation_count} reference images "
            f"({calibration_count} calibration + {evaluation_count} evaluation), but only {len(images)} were supplied."
        )
    # Deterministic split: same image ordering every run, no overlap between calibration/evaluation.
    calibration = images[:calibration_count]
    evaluation = images[calibration_count:calibration_count + evaluation_count]
    cal_root = work_root / "calibration"
    eval_root = work_root / "evaluation"
    _copy_demo_split(extracted, cal_root, images, calibration)
    _copy_demo_split(extracted, eval_root, images, evaluation)
    return cal_root, eval_root, calibration, evaluation


def _run_yolo_phase4_v5(model_path: Path, dataset_zip: Path, filename: str, compatibility: dict[str, Any]) -> dict[str, Any]:
    """Run the lightweight V5 TRACE-inspired anomaly detector for the live demo."""
    with tempfile.TemporaryDirectory(prefix="trustcv_yolo_phase4_") as td:
        work = Path(td)
        cal_root, eval_root, calibration, evaluation = _prepare_yolo_demo_split(dataset_zip, work, 15, 7)
        detector = TraceYOLOV5(model_path, ctc_samples=3, ftc_samples=5)
        report = detector.run(
            images_dir=eval_root / "images",
            dataset_root=eval_root,
            max_images=7,
            calibration_dir=cal_root / "images",
            calibration_dataset_root=cal_root,
            calibration_images=15,
            calibration_threshold=0.99,
        )
        report["model"] = filename
        report["access_level"] = "white_box"
        report["demo_mode"] = {
            "enabled": True,
            "calibration_images": 15,
            "evaluation_images": 7,
            "ctc_samples": 3,
            "ftc_samples": 5,
            "note": "Lightweight live-demo configuration. Use a larger held-out split for full validation."
        }
        report["reference_data"]["compatibility"] = compatibility
        report["flags"] = [{
            "check": "trace_behavioral_anomaly",
            "disposition": report.get("disposition", "review"),
            "raw_disposition": report.get("disposition", "review"),
            "confidence": None,
            "reason": (
                "No evaluation image crossed the calibrated anomaly percentile."
                if report.get("disposition") == "accept"
                else "One or more evaluation images showed anomalous behavior relative to the calibration reference."
            ),
            "evidence": {
                "mean_trace_score": report.get("dataset_metrics", {}).get("mean_trace_score"),
                "p90_trace_score": report.get("dataset_metrics", {}).get("p90_trace_score"),
                "suspicious_fraction": report.get("dataset_metrics", {}).get("suspicious_fraction"),
                "calibrated_threshold": report.get("dataset_metrics", {}).get("calibrated_threshold"),
                "calibration_images": report.get("calibration", {}).get("reference_images"),
            },
            "limitations": report.get("limitations", []),
        }]
        report["reference_data"]["split"] = {
            "method": "deterministic_ordered_split",
            "calibration_images": len(calibration),
            "evaluation_images": len(evaluation),
            "overlap": 0,
        }
        return report


@app.post("/api/verify/model/phase4")
async def verify_model_phase4(
    file: UploadFile = File(...),
    reference_dataset: UploadFile | None = File(None),
    access_level: str = Form("white_box"),
    phase3_result: str = Form("{}"),
):
    """Run only Phase 4 after Phase 3 has completed."""
    data = await file.read()
    digest = sha256_bytes(data)
    filename = file.filename or "uploaded_model.pt"
    path = RUNTIME_DIR / f"{digest}{Path(filename).suffix.lower() or '.pt'}"
    path.write_bytes(data)

    model_type, type_evidence = identify_model_type(path)

    if model_type == "unknown":
        path.unlink(missing_ok=True)
        return _jsonable({
            "passed": False,
            "phase": "phase4_model_integrity",
            "model_type": "unknown",
            "sha256": digest,
            "filename": filename,
            "error": "Unsupported or unrecognized model format.",
        })

    reference_images = None
    reference_info = None

    if reference_dataset is not None:
        dataset_bytes = await reference_dataset.read()
        if not dataset_bytes:
            return _jsonable({
                "passed": False,
                "phase": "phase4_model_integrity",
                "model_type": model_type,
                "sha256": digest,
                "error": "The uploaded reference dataset is empty.",
            })
        reference_dataset_digest = f"sha256:{sha256_bytes(dataset_bytes)}"
        reference_dataset_name = reference_dataset.filename or "reference.zip"

        suffix = Path(reference_dataset_name).suffix.lower() or ".zip"
        with tempfile.NamedTemporaryFile(
            prefix="trustcv_reference_",
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(dataset_bytes)
            dataset_path = Path(tmp.name)

        try:
            if model_type == "yolo":
                # Do not materialize an entire COCO archive into RAM just to test compatibility.
                # The compatibility inspector reads metadata and validates a bounded image sample.
                yolo_dataset_info = inspect_reference_zip(dataset_path)
                yolo_model_info = inspect_yolo_model(path)
                yolo_compatibility = validate_yolo_reference(yolo_model_info, yolo_dataset_info)
                reference_images = None
                reference_info = yolo_dataset_info
                reference_info["compatibility"] = yolo_compatibility
            else:
                dataset_result = load_dataset_from_zip(
                    dataset_path,
                    max_samples=None,
                )
                reference_images = dataset_result.images
                reference_info = dataset_summary(dataset_result)
        except Exception as exc:
            return _jsonable({
                "passed": False,
                "phase": "phase4_model_integrity",
                "model_type": model_type,
                "sha256": digest,
                "error": f"Reference dataset could not be loaded: {exc}",
            })
        finally:
            dataset_path.unlink(missing_ok=True)

    if reference_info is not None:
        reference_info["supplied_filename"] = reference_dataset_name if reference_dataset is not None else None
        reference_info["supplied_digest"] = reference_dataset_digest if reference_dataset is not None else None

    if model_type == "smallcnn":
        if reference_images is None:
            return _jsonable({
                "passed": False,
                "model_type": model_type,
                "model_type_evidence": type_evidence,
                "filename": filename,
                "sha256": digest,
                "error": "SmallCNN Phase 4 requires a clean reference dataset. Upload a CIFAR-10 binary ZIP or another compatible clean reference dataset.",
            })

        compatibility = validate_dataset_for_model(
            dataset_result,
            expected_channels=3,
            expected_height=32,
            expected_width=32,
            expected_classes=10,
        )
        reference_info["compatibility"] = compatibility

        if not compatibility["compatible"]:
            phase4 = {
                "phase": "phase4_model_integrity",
                "status": "blocked",
                "disposition": "review",
                "access_level": access_level,
                "flags": [],
                "reference_data": reference_info,
                "limitations": [
                    "Phase 4 was not executed because the supplied reference dataset is incompatible with the current SmallCNN testbed.",
                    *compatibility["errors"],
                ],
            }
        else:
            phase4 = _phase4_smallcnn(
                path,
                access_level,
                reference_images,
                reference_info,
            )

        phase6_9 = _smallcnn_phase6_9_mock(digest)

    elif model_type == "yolo":
        if reference_dataset is None:
            phase4 = {
                "phase": "phase4_model_integrity",
                "status": "blocked",
                "disposition": "review",
                "access_level": access_level,
                "flags": [],
                "limitations": ["A clean reference dataset is required before YOLO Phase 4 can run."],
            }
        else:
            try:
                # The ZIP is re-read here because the compatibility endpoint and Phase 4 are
                # intentionally separate requests. Run the detector only after the same gate passes.
                with tempfile.NamedTemporaryFile(prefix="trustcv_phase4_yolo_data_", suffix=".zip", delete=False) as ytmp:
                    ytmp.write(dataset_bytes)
                    yolo_phase4_dataset_path = Path(ytmp.name)
                yolo_dataset_info = inspect_reference_zip(yolo_phase4_dataset_path)
                yolo_model_info = inspect_yolo_model(path)
                yolo_compatibility = validate_yolo_reference(yolo_model_info, yolo_dataset_info)
                if not yolo_compatibility.get("compatible"):
                    phase4 = {
                        "phase": "phase4_model_integrity",
                        "status": "blocked",
                        "disposition": "quarantine",
                        "access_level": access_level,
                        "flags": [{"check": "yolo_dataset_compatibility", "disposition": "quarantine", "confidence": None, "reason": (yolo_compatibility.get("errors") or ["Reference dataset is incompatible with the uploaded YOLO model."])[0]}],
                        "reference_data": {"dataset": yolo_dataset_info, "compatibility": yolo_compatibility},
                        "limitations": ["Behavioral analysis was not executed because dataset compatibility failed."],
                    }
                else:
                    phase4 = _run_yolo_phase4_v5(path, yolo_phase4_dataset_path, filename, yolo_compatibility)
                yolo_phase4_dataset_path.unlink(missing_ok=True)
            except Exception as exc:
                phase4 = {
                    "phase": "phase4_model_integrity",
                    "status": "error",
                    "disposition": "review",
                    "access_level": access_level,
                    "flags": [{"check": "yolo_trace_anomaly_detection", "disposition": "review", "confidence": None, "reason": str(exc)}],
                    "limitations": ["YOLO Phase 4 execution failed; no anomaly verdict was established."],
                }
        phase6_9 = {
            "phase6": {"status": "ready", "detail": "Existing YOLO OOD gate will run after an input is supplied."},
            "phase7": {"status": "ready", "detail": "Existing YOLO inference/live-CV path will run after an input is supplied."},
            "phase8": {"status": "ready", "detail": "Existing YOLO inference-integrity path will run after an input is supplied."},
            "phase9": {"status": "ready", "detail": "Existing audit-ledger path will record the inference event."},
        }
    else:
        phase4 = {
            "phase": "phase4_model_integrity",
            "status": "unsupported",
            "disposition": "review",
            "access_level": access_level,
            "flags": [],
            "reference_data": reference_info,
            "limitations": [f"Phase 4 adapter unavailable for model type: {model_type}"],
        }
        phase6_9 = {}

    try:
        phase3 = json.loads(phase3_result)
    except Exception:
        phase3 = {"raw": phase3_result}

    run_id = phase3.get("run_id") if isinstance(phase3, dict) else None
    if not run_id or run_id not in RUNS:
        run_id = _new_run(filename, model_type, digest, path, phase3)
    else:
        RUNS[run_id]["model_path"] = str(path)

    pre_phase4_checkpoint = _checkpoint_model(run_id, "phase4_pre", path, RUNS[run_id]["model_sha256"])
    if not pre_phase4_checkpoint["verified"]:
        phase4 = {
            "phase": "phase4_model_integrity", "status": "blocked", "disposition": "quarantine",
            "access_level": access_level, "flags": [{"check": "model_hash_checkpoint", "disposition": "quarantine", "reason": "Model hash changed before Phase 4 execution."}],
            "limitations": ["Phase 4 was not executed because model artifact continuity failed."],
            "hash_checkpoint": pre_phase4_checkpoint,
        }
        RUNS[run_id]["phase4"] = phase4
        _log_event({"type": "phase4.blocked", "phase": "phase4", "run_id": run_id, "model_sha256": digest, "phase4": phase4}, run_id=run_id, event_type="phase4.blocked")
        return _jsonable({"passed": False, "phase": "phase4_model_integrity", "model_type": model_type, "model_type_evidence": type_evidence, "filename": filename, "sha256": digest, "model_ref": digest, "run_id": run_id, "phase3": phase3, "phase4": phase4, "hash_checkpoint": pre_phase4_checkpoint, "reference_data": reference_info})

    phase4_checkpoint = _checkpoint_model(run_id, "phase4", path, RUNS[run_id]["model_sha256"])
    phase4["hash_checkpoint"] = phase4_checkpoint
    RUNS[run_id]["phase4"] = phase4
    _log_event({
        "schema_version": "5.0", "timestamp_utc": utc_now(),
        "type": "model_assurance_phase4", "phase": "phase4", "run_id": run_id,
        "filename": filename, "sha256": digest, "model_type": model_type,
        "phase3": phase3, "phase4": phase4, "reference_data": reference_info,
    }, run_id=run_id, event_type="phase4.result")

    return _jsonable({
        "passed": phase4.get("disposition") == "accept",
        "phase": "phase4_model_integrity", "model_type": model_type,
        "model_type_evidence": type_evidence, "filename": filename, "sha256": digest,
        "model_ref": digest, "run_id": run_id, "phase3": phase3, "phase4": phase4,
        "phase6_9": phase6_9, "reference_data": reference_info, "hash_checkpoint": phase4_checkpoint,
    })




def _report_text(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (dict, list)):
        return json.dumps(_jsonable(value), indent=2, default=str)
    return str(value)


def _report_status(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    if value is None:
        return "N/A"
    return str(value).upper()




def _build_phase3_report_pdf(result: dict[str, Any], output_path: Path) -> None:
    """Build ONLY the Phase 3 Trust, Identity & Provenance report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import html

    model = result.get("model") or {}
    filename = model.get("filename") or result.get("filename") or "Not available"
    model_type = model.get("model_type") or result.get("model_type") or "Not available"
    digest = model.get("sha256") or result.get("sha256") or "Not available"
    access_level = result.get("access_level") or "white_box"
    generated_at = result.get("generated_at") or utc_now()

    phase3 = result.get("phase3") or result
    mirad = result.get("mirad") or phase3.get("mirad") or phase3.get("phase3") or {}
    checks = result.get("checks") or phase3.get("checks") or mirad.get("checks") or []

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#172033")
    slate = colors.HexColor("#344054")
    muted = colors.HexColor("#667085")
    line = colors.HexColor("#D0D5DD")
    light = colors.HexColor("#F2F4F7")
    teal = colors.HexColor("#13795B")
    amber = colors.HexColor("#B54708")
    red = colors.HexColor("#B42318")

    title = ParagraphStyle("P3Title", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=25, leading=29, textColor=navy, spaceAfter=4)
    subtitle = ParagraphStyle("P3Subtitle", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=10, leading=14, textColor=muted, spaceAfter=11)
    section = ParagraphStyle("P3Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
                             fontSize=15, leading=19, textColor=navy, spaceBefore=12, spaceAfter=7)
    sub = ParagraphStyle("P3Sub", parent=styles["Heading3"], fontName="Helvetica-Bold",
                         fontSize=10.5, leading=14, textColor=navy, spaceBefore=7, spaceAfter=4)
    body = ParagraphStyle("P3Body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=8.8, leading=13, textColor=slate, spaceAfter=4)
    small = ParagraphStyle("P3Small", parent=body, fontSize=7.7, leading=10.5, textColor=muted)
    mono = ParagraphStyle("P3Mono", parent=body, fontName="Courier", fontSize=7.2,
                          leading=9.5, textColor=slate)
    verdict_style = ParagraphStyle("P3Verdict", parent=styles["Heading2"], fontName="Helvetica-Bold",
                                   fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.white)

    def txt(v):
        if v is None:
            return "Not available"
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, (dict, list)):
            return json.dumps(_jsonable(v), indent=2, default=str)
        return str(v)

    def para(v, style=body):
        return Paragraph(html.escape(txt(v)).replace("\n", "<br/>"), style)

    def status(v):
        if v is True:
            return "PASS"
        if v is False:
            return "FAIL"
        if v is None:
            return "NOT AVAILABLE"
        return str(v).replace("_", " ").upper()

    def kv_table(rows):
        if not rows:
            return
        t = Table(rows, colWidths=[50*mm, 126*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), light),
            ("GRID", (0,0), (-1,-1), 0.35, line),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)

    def section_table(headers, rows, widths):
        data = [[para(h, small) for h in headers]]
        for row in rows:
            data.append([para(v, body if i < len(row)-1 else small) for i, v in enumerate(row)])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAECF0")),
            ("GRID", (0,0), (-1,-1), 0.35, line),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)

    passed = result.get("passed")
    if passed is True:
        verdict_text, verdict_bg = "PHASE 3 PASSED", teal
    elif passed is False:
        verdict_text, verdict_bg = "PHASE 3 FAILED", red
    else:
        verdict_text, verdict_bg = "PHASE 3 REVIEW", amber

    def header_footer(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(navy)
        canvas.rect(0, h - 7*mm, w, 7*mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(muted)
        canvas.drawString(17*mm, 9*mm, "TrustCV • Phase 3 • Trust, Identity & Provenance")
        canvas.drawRightString(w - 17*mm, 9*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            rightMargin=17*mm, leftMargin=17*mm,
                            topMargin=19*mm, bottomMargin=16*mm,
                            title="TrustCV Phase 3 Trust Identity Report", author="TrustCV")
    story = [Spacer(1, 3*mm), Paragraph("TrustCV", title),
             Paragraph("Phase 3 — Trust, Identity & Provenance Report", subtitle)]

    vt = Table([[Paragraph(verdict_text, verdict_style)]], colWidths=[176*mm], rowHeights=[17*mm])
    vt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),verdict_bg),
                            ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [vt, Spacer(1, 6*mm)]

    rows = []
    for k, v in [
        ("Model file", filename), ("Detected model type", model_type.upper()),
        ("SHA-256 digest", digest), ("Access level", access_level),
        ("Report generated", generated_at)
    ]:
        rows.append([para(k, small), para(v, body)])
    kv_table(rows)

    story.append(Paragraph("Assessment", section))
    assessment = (
        "Phase 3 establishes the identity of the uploaded model and checks it against the "
        "available MIRAD reference. Registration/reference status is an identity control and "
        "does not establish model safety. Missing provenance or signatures are reported as unavailable."
    )
    story.append(Paragraph(assessment, body))

    story.append(Paragraph("Identity & Trust Checks", section))
    check_rows = []
    for c in checks:
        if isinstance(c, dict):
            check_rows.append((
                c.get("label", c.get("id", "Check")),
                status(c.get("passed")),
                c.get("detail", "No additional evidence supplied.")
            ))
    if check_rows:
        section_table(["Check", "Status", "Evidence / meaning"], check_rows,
                      [49*mm, 27*mm, 100*mm])
    else:
        story.append(Paragraph("No individual Phase 3 checks were returned.", body))

    artifact = {}
    if isinstance(mirad, dict):
        artifact = mirad.get("artifact_identity") or mirad.get("candidate") or {}
        if not artifact and isinstance(mirad.get("evidence"), dict):
            artifact = mirad["evidence"].get("candidate") or {}

    story.append(Paragraph("MIRAD Artifact Identity", section))
    if artifact:
        rows = []
        for k in ("artifact_type", "artifact_id", "version", "artifact_digest", "format"):
            if k in artifact:
                rows.append([para(k.replace("_", " ").title(), small), para(artifact[k], body)])
        kv_table(rows)
    else:
        story.append(Paragraph("No artifact identity record was returned by the Phase 3 response.", body))

    if isinstance(mirad, dict):
        story.append(Paragraph("MIRAD Verification Result", section))
        rows = []
        for k in ("valid", "first_seen", "registration_status", "artifact_digest", "artifact_identity"):
            if k in mirad:
                rows.append([para(k.replace("_", " ").title(), small), para(mirad[k], body)])
        kv_table(rows)

    reference = result.get("reference_data")
    if reference:
        story.append(Paragraph("Reference Dataset", section))
        rows = []
        if isinstance(reference, dict):
            for k in ("source", "filename", "total_images", "num_images", "channels",
                      "height", "width", "num_classes"):
                if k in reference:
                    rows.append([para(k.replace("_", " ").title(), small), para(reference[k], body)])
        kv_table(rows)
        story.append(Paragraph(
            "The dataset information above records what was supplied for this Phase 3 assessment. "
            "It is metadata only; Phase 4 performs model-integrity analysis.",
            small
        ))

    limitations = []
    if isinstance(phase3, dict):
        limitations = phase3.get("limitations", []) or []
        limitations += [f"Unavailable check: {x}" for x in (phase3.get("unavailable_checks", []) or [])]
    if limitations:
        story.append(Paragraph("Phase 3 Limitations", section))
        for x in limitations:
            story.append(Paragraph("• " + html.escape(txt(x)), body))

    story.append(Paragraph("Phase 3 Decision", section))
    decision_text = (
        "Phase 3 passed. The model may proceed to the separate Phase 4 model-integrity analysis."
        if passed is True else
        "Phase 3 did not pass. The workflow should not proceed to Phase 4."
        if passed is False else
        "Phase 3 requires review based on the returned identity evidence."
    )
    story.append(Paragraph(decision_text, body))
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def _build_phase4_report_pdf(result: dict[str, Any], output_path: Path) -> None:
    """Build ONLY the Phase 4 Model Integrity report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import html

    model = result.get("model") or {}
    filename = model.get("filename") or result.get("filename") or "Not available"
    model_type = model.get("model_type") or result.get("model_type") or "Not available"
    digest = model.get("sha256") or result.get("sha256") or "Not available"
    integrity = result.get("integrity") or result.get("phase4") or {}
    if isinstance(integrity, dict) and "integrity" in integrity and isinstance(integrity["integrity"], dict):
        integrity = integrity["integrity"]
    flags = integrity.get("flags", []) if isinstance(integrity, dict) else []
    generated_at = result.get("generated_at") or utc_now()
    access_level = integrity.get("access_level") or result.get("access_level") or "white_box"

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#172033")
    slate = colors.HexColor("#344054")
    muted = colors.HexColor("#667085")
    line = colors.HexColor("#D0D5DD")
    light = colors.HexColor("#F2F4F7")
    teal = colors.HexColor("#13795B")
    amber = colors.HexColor("#B54708")
    red = colors.HexColor("#B42318")

    title = ParagraphStyle("P4Title", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=25, leading=29, textColor=navy, spaceAfter=4)
    subtitle = ParagraphStyle("P4Subtitle", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=10, leading=14, textColor=muted, spaceAfter=11)
    section = ParagraphStyle("P4Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
                             fontSize=15, leading=19, textColor=navy, spaceBefore=12, spaceAfter=7)
    sub = ParagraphStyle("P4Sub", parent=styles["Heading3"], fontName="Helvetica-Bold",
                         fontSize=10.5, leading=14, textColor=navy, spaceBefore=7, spaceAfter=4)
    body = ParagraphStyle("P4Body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=8.8, leading=13, textColor=slate, spaceAfter=4)
    small = ParagraphStyle("P4Small", parent=body, fontSize=7.7, leading=10.5, textColor=muted)
    mono = ParagraphStyle("P4Mono", parent=body, fontName="Courier", fontSize=7.2,
                          leading=9.5, textColor=slate)
    verdict_style = ParagraphStyle("P4Verdict", parent=styles["Heading2"], fontName="Helvetica-Bold",
                                   fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.white)

    def txt(v):
        if v is None:
            return "Not available"
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, (dict, list)):
            return json.dumps(_jsonable(v), indent=2, default=str)
        return str(v)

    def para(v, style=body):
        return Paragraph(html.escape(txt(v)).replace("\n", "<br/>"), style)

    def kv_table(rows):
        if not rows:
            return
        t = Table(rows, colWidths=[50*mm, 126*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(0,-1),light), ("GRID",(0,0),(-1,-1),0.35,line),
            ("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),6),
            ("RIGHTPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5)
        ]))
        story.append(t)

    def section_table(headers, rows, widths):
        data = [[para(h, small) for h in headers]]
        for row in rows:
            data.append([para(v, body if i < len(row)-1 else small) for i,v in enumerate(row)])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAECF0")),
            ("GRID",(0,0),(-1,-1),0.35,line), ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)
        ]))
        story.append(t)

    disposition = str(integrity.get("disposition") or "").lower()
    status_value = str(integrity.get("status") or "").lower()
    if disposition in {"quarantine","blocked","reject","rejected","fail","failed"}:
        verdict_text, verdict_bg = "MODEL QUARANTINED", red
    elif disposition == "accept":
        verdict_text, verdict_bg = "PHASE 4 ACCEPTED", teal
    else:
        verdict_text, verdict_bg = "PHASE 4 REVIEW REQUIRED", amber

    def header_footer(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(navy)
        canvas.rect(0, h-7*mm, w, 7*mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(muted)
        canvas.drawString(17*mm, 9*mm, "TrustCV • Phase 4 • Model Integrity")
        canvas.drawRightString(w-17*mm, 9*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            rightMargin=17*mm, leftMargin=17*mm,
                            topMargin=19*mm, bottomMargin=16*mm,
                            title="TrustCV Phase 4 Model Integrity Report", author="TrustCV")
    story = [Spacer(1,3*mm), Paragraph("TrustCV",title),
             Paragraph("Phase 4 — Model Integrity Report",subtitle)]
    vt=Table([[Paragraph(verdict_text,verdict_style)]],colWidths=[176*mm],rowHeights=[17*mm])
    vt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),verdict_bg),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [vt,Spacer(1,6*mm)]

    rows=[]
    for k,v in [("Model file",filename),("Detected model type",model_type.upper()),
                ("SHA-256 digest",digest),("Access level",access_level),
                ("Execution status",integrity.get("status")),("Disposition",integrity.get("disposition")),
                ("Report generated",generated_at)]:
        rows.append([para(k,small),para(v,body)])
    kv_table(rows)

    story.append(Paragraph("Phase 4 Assessment",section))
    if model_type.lower()=="yolo" and status_value=="placeholder":
        intro=("YOLO-specific trigger/backdoor detection is a declared placeholder in this Round-1 integration. "
               "This report records that limitation and does not present the placeholder as a completed detector.")
    else:
        intro=("Phase 4 evaluates model integrity using the configured model-specific checks. "
               "The disposition below is determined by the executed evidence; it is independent of Phase 3 identity.")
    story.append(Paragraph(intro,body))

    story.append(Paragraph("Integrity Findings",section))
    if flags:
        for idx,flag in enumerate(flags,1):
            if not isinstance(flag,dict):
                continue
            check=flag.get("check","Integrity check")
            disp=str(flag.get("disposition","review")).upper()
            reason=flag.get("reason",flag.get("detail","No reason supplied."))
            meta=[
                [para("Check",small),para(check,body)],
                [para("Disposition",small),para(disp,body)]
            ]
            if flag.get("confidence") is not None:
                meta.append([para("Confidence / severity",small),para(flag.get("confidence"),body)])
            t=Table(meta,colWidths=[50*mm,126*mm])
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),light),("GRID",(0,0),(-1,-1),0.35,line),
                                   ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),
                                   ("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
            story.append(Paragraph(f"Finding {idx} — {html.escape(txt(check))}",sub))
            story.append(t)
            story.append(Paragraph("Reason",small))
            story.append(Paragraph(html.escape(txt(reason)).replace("\n","<br/>"),body))
            evidence=flag.get("evidence") or {}
            if evidence:
                story.append(Paragraph("Technical evidence",small))
                story.append(Paragraph(html.escape(txt(evidence)).replace("\n","<br/>"),mono))
    else:
        story.append(Paragraph("No individual integrity findings were returned.",body))

    # Extract and summarize Neural Cleanse evidence without requiring a fixed schema.
    nc = integrity.get("neural_cleanse") if isinstance(integrity,dict) else None
    if nc is None and isinstance(integrity.get("details"),dict):
        nc=integrity["details"].get("neural_cleanse")
    if nc is None:
        for flag in flags:
            if isinstance(flag,dict) and "neural_cleanse" in str(flag.get("check","")).lower():
                ev=flag.get("evidence")
                if isinstance(ev,dict):
                    nc=ev
                break

    if isinstance(nc,dict):
        story.append(Paragraph("Neural Cleanse / Trigger Reconstruction",section))
        recovered=nc.get("recovered_trigger") or {}
        rows=[]
        key_map=[
            ("Suspect class",nc.get("suspect_class",recovered.get("target_class"))),
            ("Anomaly index / score",nc.get("anomaly_score",recovered.get("anomaly_index"))),
            ("Threshold",nc.get("threshold",recovered.get("threshold"))),
            ("Held-out ASR",nc.get("heldout_asr",nc.get("verification_asr_on_holdout",recovered.get("verification_asr_on_holdout")))),
            ("Mask size",nc.get("mask_l1",nc.get("mask_size",recovered.get("mask_size")))),
            ("Bounding box",nc.get("bbox",recovered.get("bbox_xyxy"))),
            ("Active pixels",nc.get("active_pixels",recovered.get("active_pixels"))),
            ("Calibration images",nc.get("n_calib")),
            ("Holdout images",nc.get("n_holdout")),
            ("Split method",nc.get("split_method")),
        ]
        for k,v in key_map:
            if v is not None:
                rows.append([para(k,small),para(v,body)])
        kv_table(rows)

        per_class=nc.get("per_class")
        if not per_class and isinstance(nc.get("per_class_flags"),list):
            per_class=nc["per_class_flags"]
        if isinstance(per_class,list) and per_class:
            pc=[]
            for item in per_class:
                if isinstance(item,dict):
                    pc.append((item.get("target_class",item.get("class",item.get("class_id",""))),
                                item.get("mask_size",item.get("mask_l1","")),
                                item.get("optimization_asr",item.get("heldout_asr",item.get("asr",""))),
                                item.get("anomaly_index",item.get("anomaly_score",""))))
            if pc:
                story.append(Paragraph("Per-class trigger assessment",sub))
                section_table(["Class","Mask size","ASR","Anomaly index"],pc,
                              [30*mm,48*mm,48*mm,50*mm])

    # Dataset evidence belongs in Phase 4 because it is used by the detector.
    reference=integrity.get("reference_data") if isinstance(integrity,dict) else None
    if reference:
        story.append(Paragraph("Reference Data Used by Phase 4",section))
        rows=[]
        if isinstance(reference,dict):
            for k in ("source","filename","total_images","calibration_images","holdout_images",
                      "split_method","calibration_seed","holdout_seed","channels","height","width","num_classes"):
                if k in reference:
                    rows.append([para(k.replace("_"," ").title(),small),para(reference[k],body)])
        kv_table(rows)

    limitations=integrity.get("limitations",[]) if isinstance(integrity,dict) else []
    if limitations:
        story.append(Paragraph("Phase 4 Limitations",section))
        for x in limitations:
            story.append(Paragraph("• "+html.escape(txt(x)),body))

    story.append(Paragraph("Phase 4 Decision",section))
    if disposition in {"quarantine","blocked","reject","rejected","fail","failed"}:
        decision=("The model is quarantined by Phase 4. The integrity evidence produced by the executed checks "
                  "requires the model to stop at this gate.")
    elif disposition=="accept":
        decision="Phase 4 accepted the model based on the executed integrity checks."
    elif model_type.lower()=="yolo" and status_value=="placeholder":
        decision=("Phase 4 is a declared placeholder for YOLO in this integration. The model remains in review "
                  "for this phase; any downstream demo continuation is explicitly separate from Phase 4 clearance.")
    else:
        decision="Phase 4 requires review based on the returned integrity evidence."
    story.append(Paragraph(decision,body))

    doc.build(story,onFirstPage=header_footer,onLaterPages=header_footer)


def _build_trust_report_pdf(result: dict[str, Any], output_path: Path) -> None:
    """Dispatch PDF generation to the exact requested report scope."""
    scope = str(result.get("report_scope") or "").lower()
    report_type = str(result.get("report_type") or "").lower()

    if scope == "phase3" or "phase 3" in report_type:
        return _build_phase3_report_pdf(result, output_path)
    if scope == "phase4" or "phase 4" in report_type:
        return _build_phase4_report_pdf(result, output_path)
    return _build_complete_report_pdf(result, output_path)


def _build_complete_report_pdf(result: dict[str, Any], output_path: Path) -> None:
    """Build a human-readable, PS-aligned TrustCV assurance report.

    The frontend sends either a phase report or the complete report.  The
    complete report keeps model metadata under ``model`` and phase evidence
    under ``phase3``/``phase4``/``phase6``...; older reports may use top-level
    fields.  Normalize both shapes before rendering so the cover never shows
    "Unknown" when the evidence is present.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Normalize report shape ----------
    model = result.get("model") or {}
    phase3 = result.get("phase3") or {}
    phase4_report = result.get("phase4") or {}
    phase4 = phase4_report.get("integrity") if isinstance(phase4_report, dict) else None
    if not isinstance(phase4, dict):
        phase4 = result.get("integrity") or {}
    phase3_data = phase3.get("mirad") if isinstance(phase3, dict) else None
    if not isinstance(phase3_data, dict):
        phase3_data = phase3.get("phase3") if isinstance(phase3, dict) else None
    if not isinstance(phase3_data, dict):
        phase3_data = result.get("mirad") or {}

    filename = model.get("filename") or result.get("filename") or "Unknown"
    model_type = model.get("model_type") or result.get("model_type") or "Unknown"
    digest = model.get("sha256") or result.get("sha256") or "Unknown"
    access_level = (
        phase4.get("access_level")
        or result.get("access_level")
        or "white_box"
    )
    generated_at = result.get("generated_at") or utc_now()

    checks = []
    if isinstance(phase3, dict):
        checks = phase3.get("checks") or []
    if not checks and isinstance(phase3_data, dict):
        checks = phase3_data.get("checks") or []
    if not checks and isinstance(result.get("checks"), list):
        checks = result["checks"]

    reference_data = (
        result.get("reference_data")
        or (phase4.get("reference_data") if isinstance(phase4, dict) else None)
        or (phase3.get("reference_data") if isinstance(phase3, dict) else None)
    )

    phase6 = result.get("phase6")
    phase7 = result.get("phase7")
    phase8 = result.get("phase8")
    phase9 = result.get("phase9")
    shift = result.get("shift") or result.get("distribution_shift") or {}
    inference = result.get("inference") or {}

    # ---------- Styles ----------
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#172033")
    slate = colors.HexColor("#344054")
    muted = colors.HexColor("#667085")
    line = colors.HexColor("#D0D5DD")
    light = colors.HexColor("#F2F4F7")
    teal = colors.HexColor("#13795B")
    amber = colors.HexColor("#B54708")
    red = colors.HexColor("#B42318")
    blue = colors.HexColor("#175CD3")

    title = ParagraphStyle(
        "TrustTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=25, leading=29, textColor=navy, spaceAfter=4
    )
    subtitle = ParagraphStyle(
        "TrustSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=10, leading=14, textColor=muted, spaceAfter=11
    )
    section = ParagraphStyle(
        "TrustSection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=navy, spaceBefore=12, spaceAfter=7
    )
    sub = ParagraphStyle(
        "TrustSub", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=14, textColor=navy, spaceBefore=7, spaceAfter=4
    )
    body = ParagraphStyle(
        "TrustBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.8, leading=13, textColor=slate, spaceAfter=4
    )
    small = ParagraphStyle(
        "TrustSmall", parent=body, fontSize=7.7, leading=10.5, textColor=muted
    )
    mono = ParagraphStyle(
        "TrustMono", parent=body, fontName="Courier", fontSize=7.2,
        leading=9.5, textColor=slate
    )
    verdict_style = ParagraphStyle(
        "TrustVerdict", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.white
    )

    import html

    def txt(v):
        if v is None:
            return "Not available"
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, (dict, list)):
            return json.dumps(_jsonable(v), indent=2, default=str)
        return str(v)

    def para(v, style=body):
        return Paragraph(html.escape(txt(v)).replace("\n", "<br/>"), style)

    def status(v):
        if v is True:
            return "PASS"
        if v is False:
            return "FAIL"
        if v is None:
            return "NOT AVAILABLE"
        return str(v).replace("_", " ").upper()

    def add_kv(rows, key, value):
        rows.append([para(key, small), para(value, body)])

    def add_kv_table(rows):
        if not rows:
            return
        t = Table(rows, colWidths=[50*mm, 126*mm], repeatRows=0)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), light),
            ("GRID", (0,0), (-1,-1), 0.35, line),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)

    def add_bullets(items, style=body):
        for item in items:
            story.append(Paragraph("• " + html.escape(txt(item)), style))
        if items:
            story.append(Spacer(1, 1.5*mm))

    def add_section_table(headers, rows, widths):
        data = [[para(h, small) for h in headers]]
        for row in rows:
            data.append([para(v, body if i < len(row)-1 else small) for i, v in enumerate(row)])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAECF0")),
            ("TEXTCOLOR", (0,0), (-1,0), slate),
            ("GRID", (0,0), (-1,-1), 0.35, line),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)

    # ---------- Overall Verdict ----------
    # Phase 4 acceptance is only a model-integrity gate. The complete report
    # must reflect downstream inference/OOD/integrity failures as well.
    p4_disposition = str(phase4.get("disposition") or "").lower()
    p6 = result.get("phase6") or {}
    p8 = result.get("phase8") or {}
    p8_integrity = p8.get("integrity") if isinstance(p8, dict) else {}
    if not isinstance(p8_integrity, dict):
        p8_integrity = {}
    p8_disposition = str(
        p8_integrity.get("disposition")
        or p8.get("disposition")
        or p8.get("status")
        or ""
    ).lower()
    explicit_overall = str(result.get("overall_disposition") or "").lower()
    p6_out_of_distribution = p6.get("inDistribution") is False

    if explicit_overall in {"quarantine", "blocked", "reject", "rejected", "fail", "failed"}:
        verdict_text, verdict_bg = "MODEL QUARANTINED", red
    elif p4_disposition in {"quarantine", "blocked", "reject", "rejected", "fail", "failed"}:
        verdict_text, verdict_bg = "MODEL QUARANTINED", red
    elif p8_disposition in {"quarantine", "blocked", "reject", "rejected", "fail", "failed", "flagged"}:
        verdict_text, verdict_bg = "MODEL QUARANTINED", red
    elif p6_out_of_distribution:
        verdict_text, verdict_bg = "MODEL QUARANTINED", red
    elif explicit_overall == "review":
        verdict_text, verdict_bg = "REVIEW REQUIRED", amber
    elif model_type == "yolo" and str(phase4.get("status")) == "placeholder":
        verdict_text, verdict_bg = "REVIEW REQUIRED", amber
    else:
        verdict_text, verdict_bg = "MODEL ACCEPTED", teal

    def header_footer(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(navy)
        canvas.rect(0, h - 7*mm, w, 7*mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(muted)
        canvas.drawString(17*mm, 9*mm, "TrustCV • Model Assurance & Verification")
        canvas.drawRightString(w - 17*mm, 9*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=17*mm, leftMargin=17*mm,
        topMargin=19*mm, bottomMargin=16*mm,
        title="TrustCV Complete Model Assurance Report",
        author="TrustCV",
    )
    story = []

    # ---------- Cover / executive summary ----------
    story += [Spacer(1, 3*mm), Paragraph("TrustCV", title),
              Paragraph("Complete Model Assurance & Verification Report", subtitle)]

    vtable = Table([[Paragraph(verdict_text, verdict_style)]],
                   colWidths=[176*mm], rowHeights=[17*mm])
    vtable.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), verdict_bg),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(vtable)
    story.append(Spacer(1, 6*mm))

    identity_rows = []
    add_kv(identity_rows, "Model file", filename)
    add_kv(identity_rows, "Detected model type", model_type.upper())
    add_kv(identity_rows, "SHA-256 digest", digest)
    add_kv(identity_rows, "Access level", access_level)
    add_kv(identity_rows, "Report generated", generated_at)
    add_kv(identity_rows, "Report type", result.get("report_type", "TrustCV Complete Model Assurance Report"))
    add_kv_table(identity_rows)

    story.append(Paragraph("Executive Summary", section))
    if model_type.lower() == "yolo":
        summary = (
            f"TrustCV identified <b>{html.escape(filename)}</b> as a YOLO model and verified its "
            "artifact identity through MIRAD. The current Round-1 YOLO Phase 4 backdoor/trigger "
            "assessment is explicitly a placeholder, so the model is reported for review rather "
            "than being presented as fully cleared. The downstream YOLO inference assurance path "
            "can still provide distribution-shift, inference-output, integrity, and audit evidence."
        )
    else:
        summary = (
            f"TrustCV evaluated <b>{html.escape(filename)}</b> using the available identity, "
            "model-integrity, inference, and audit stages. The report distinguishes verified "
            "evidence from unavailable or placeholder checks and records the access assumptions, "
            "supporting evidence, disposition, and limitations."
        )
    story.append(Paragraph(summary, body))

    story.append(Paragraph("Assessment Scope Against SIH26228", section))
    story.append(Paragraph(
        "The SIH26228 problem statement calls for an evidence-based assurance layer covering "
        "training-data integrity, model integrity, inference provenance/output integrity, "
        "distribution-shift/anomaly assessment, and analyst-facing governance. It also calls for "
        "human-readable reasons, evidence, confidence/severity, affected assets, dispositions, "
        "a tamper-evident audit trail, and an explicit coverage/limitations statement. "
        "This report maps those requirements to the evidence available for this run.",
        body
    ))

    coverage_rows = [
        ("Training-data integrity",
         "Reference dataset supplied" if reference_data else "Not demonstrated",
         "Dataset metadata is recorded when supplied; full dataset attack analysis is not claimed unless Phase 4 produced it.",
         "Requires the dataset verification/poisoning module for complete coverage."),
        ("Model integrity",
         "Placeholder for YOLO" if model_type.lower() == "yolo" else status(phase4.get("disposition")),
         "Model type, access level, findings, evidence and limitations are recorded.",
         "YOLO-specific trigger/backdoor analysis is not implemented in this Round-1 path." if model_type.lower() == "yolo" else "See Phase 4 findings below."),
        ("Inference provenance & output integrity",
         status(phase8.get("status") if isinstance(phase8, dict) else None),
         "Input/model/inference event evidence and the generated audit record are reported when Phase 6-9 has run.",
         "Depends on an inference execution for a real Phase 8/9 result."),
        ("Distribution shift & anomaly assessment",
         status(phase6.get("status") if isinstance(phase6, dict) else None),
         "OOD/in-distribution result, confidence and reasons are reported for the executed YOLO path.",
         "No real result exists until an inference image is processed."),
        ("Analyst-facing governance",
         "Implemented",
         "Human-readable findings, disposition, evidence, confidence fields and limitations are included.",
         "Unsupported attack classes remain explicitly marked."),
    ]
    add_section_table(
        ["PS capability", "Run status", "Evidence produced", "Known limitation"],
        coverage_rows,
        [38*mm, 28*mm, 57*mm, 53*mm]
    )

    # ---------- Phase 3 ----------
    story.append(Paragraph("Phase 3 — Trust, Identity & Provenance", section))
    story.append(Paragraph(
        "Phase 3 establishes what artifact was evaluated and whether it matches the MIRAD "
        "reference. Registration/reference status is an identity control, not a statement that "
        "the model is safe. Missing signatures or provenance are reported as unavailable rather "
        "than inferred.",
        body
    ))
    p3_rows = []
    if checks:
        for c in checks:
            p3_rows.append((
                c.get("label", c.get("id", "Check")),
                status(c.get("passed")),
                c.get("detail", "No additional evidence supplied.")
            ))
    elif isinstance(phase3_data, dict):
        p3_rows.append(("Phase 3 result", status(phase3_data.get("passed")), phase3_data.get("reason", "")))
    if p3_rows:
        add_section_table(["Check", "Status", "Evidence / human-readable meaning"],
                           p3_rows, [49*mm, 27*mm, 100*mm])

    artifact = phase3_data.get("artifact_identity") if isinstance(phase3_data, dict) else None
    if not isinstance(artifact, dict) and isinstance(phase3_data, dict):
        evidence = phase3_data.get("evidence", {})
        candidate = evidence.get("candidate", {}) if isinstance(evidence, dict) else {}
        artifact = candidate if candidate else None
    if artifact:
        story.append(Paragraph("MIRAD Artifact Identity", sub))
        rows = []
        for k in ("artifact_type", "artifact_id", "version", "artifact_digest", "format"):
            if k in artifact:
                add_kv(rows, k.replace("_", " ").title(), artifact[k])
        add_kv_table(rows)

    limitations3 = []
    if isinstance(phase3_data, dict):
        limitations3 = phase3_data.get("limitations", []) or []
        unavailable = phase3_data.get("unavailable_checks", []) or []
        if unavailable:
            limitations3 += ["Unavailable check: " + txt(x) for x in unavailable]
    if limitations3:
        story.append(Paragraph("Phase 3 limitations", sub))
        add_bullets(limitations3)

    # ---------- Reference dataset ----------
    if reference_data:
        story.append(Paragraph("Reference Dataset Used for Assessment", section))
        ref_rows = []
        if isinstance(reference_data, dict):
            preferred = (
                "source", "filename", "total_images", "num_images", "channels",
                "height", "width", "num_classes", "calibration_images",
                "holdout_images", "split_method", "calibration_seed", "holdout_seed"
            )
            for k in preferred:
                if k in reference_data:
                    add_kv(ref_rows, k.replace("_", " ").title(), reference_data[k])
        if ref_rows:
            add_kv_table(ref_rows)
        story.append(Paragraph(
            "The report records the reference data actually supplied to TrustCV. TrustCV does "
            "not silently download a dataset for this assessment.",
            small
        ))

    # ---------- Phase 4 ----------
    story.append(Paragraph("Phase 4 — Model Integrity", section))
    p4_rows = []
    add_kv(p4_rows, "Execution status", phase4.get("status"))
    add_kv(p4_rows, "Disposition", phase4.get("disposition"))
    add_kv(p4_rows, "Access level", phase4.get("access_level", access_level))
    if p4_rows:
        add_kv_table(p4_rows)

    flags = phase4.get("flags", []) if isinstance(phase4, dict) else []
    if flags:
        story.append(Paragraph("Integrity Findings", sub))
        # Do not put potentially large machine-generated evidence into a single
        # table row. ReportLab cannot split an oversized table row across pages,
        # which previously caused Phase 4 PDF generation to fail with LayoutError.
        # Each finding is therefore rendered as a compact metadata table followed
        # by normal Paragraph flowables that can safely continue onto the next page.
        for idx, flag in enumerate(flags, 1):
            check_name = flag.get("check", "Finding")
            disposition = str(flag.get("disposition", "review")).upper()
            reason = flag.get("reason", flag.get("detail", "No reason supplied."))
            confidence = flag.get("confidence")
            evidence_text = flag.get("evidence") or {}

            story.append(Paragraph(f"Finding {idx} — {html.escape(txt(check_name))}", sub))
            finding_meta = [
                [para("Affected asset / check", small), para(check_name, body)],
                [para("Disposition", small), para(disposition, body)],
            ]
            if confidence is not None:
                finding_meta.append([para("Confidence / severity", small), para(confidence, body)])
            ft = Table(finding_meta, colWidths=[50*mm, 126*mm])
            ft.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (0,-1), light),
                ("GRID", (0,0), (-1,-1), 0.35, line),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 5),
                ("RIGHTPADDING", (0,0), (-1,-1), 5),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(ft)
            story.append(Paragraph("Reason / human-readable explanation", small))
            story.append(Paragraph(html.escape(txt(reason)).replace("\n", "<br/>"), body))
            if evidence_text:
                story.append(Paragraph("Technical evidence", small))
                evidence_text_str = txt(evidence_text)
                story.append(Paragraph(html.escape(evidence_text_str).replace("\n", "<br/>"), mono))

    # Neural Cleanse details / class-level evidence
    nc = phase4.get("neural_cleanse") if isinstance(phase4, dict) else None
    if nc is None and isinstance(phase4.get("details"), dict):
        nc = phase4["details"].get("neural_cleanse")
    if isinstance(nc, dict):
        story.append(Paragraph("Neural Cleanse / Trigger Reconstruction Evidence", sub))
        rows = []
        for k in (
            "suspect_class", "anomaly_score", "threshold", "heldout_asr",
            "mask_l1", "bbox", "active_pixels", "median_mask_l1", "mad_mask_l1",
            "n_calib", "n_holdout", "split_method"
        ):
            if k in nc:
                add_kv(rows, k.replace("_", " ").title(), nc[k])
        if rows:
            add_kv_table(rows)

        per_class = nc.get("per_class")
        if isinstance(per_class, list) and per_class:
            story.append(Paragraph("Per-class trigger assessment", sub))
            pc_rows = []
            for item in per_class:
                if isinstance(item, dict):
                    pc_rows.append((
                        item.get("class", item.get("class_id", "")),
                        item.get("mask_l1", ""),
                        item.get("heldout_asr", item.get("asr", "")),
                        item.get("anomaly_score", "")
                    ))
            if pc_rows:
                add_section_table(["Class", "Mask L1", "Held-out ASR", "Anomaly score"],
                                   pc_rows, [30*mm, 40*mm, 50*mm, 56*mm])

    p4_limits = phase4.get("limitations", []) if isinstance(phase4, dict) else []
    if p4_limits:
        story.append(Paragraph("Phase 4 limitations", sub))
        add_bullets(p4_limits)

    # ---------- Phase 6 ----------
    story.append(PageBreak())
    story.append(Paragraph("Phase 6 — Distribution-Shift & Anomaly Assessment", section))
    if isinstance(phase6, dict):
        rows = []
        add_kv(rows, "Execution status", phase6.get("status"))
        add_kv(rows, "In distribution", phase6.get("inDistribution"))
        add_kv(rows, "Confidence / score", phase6.get("confidence"))
        add_kv(rows, "Human-readable result", phase6.get("detail"))
        add_kv_table(rows)
        reasons = phase6.get("reasons")
        if reasons:
            story.append(Paragraph("Observed OOD / shift signals", sub))
            add_bullets(reasons if isinstance(reasons, list) else [reasons])
    else:
        story.append(Paragraph(
            "Phase 6 has not produced a real inference result in this report. The report therefore "
            "does not invent an OOD or distribution-shift conclusion.",
            body
        ))

    # ---------- Phase 7 ----------
    story.append(Paragraph("Phase 7 — Inference Output", section))
    if isinstance(phase7, dict):
        rows = []
        add_kv(rows, "Execution status", phase7.get("status"))
        detections = phase7.get("detections") or []
        add_kv(rows, "Detections produced", len(detections) if isinstance(detections, list) else "Not available")
        add_kv_table(rows)

        if isinstance(detections, list) and detections:
            det_rows = []
            for idx, d in enumerate(detections, 1):
                if isinstance(d, dict):
                    det_rows.append((
                        idx,
                        d.get("class", d.get("class_id", "Unknown")),
                        d.get("confidence", d.get("conf", "Unknown")),
                        d.get("bbox", d.get("box", "Unknown"))
                    ))
            if det_rows:
                story.append(Paragraph("Detected objects", sub))
                add_section_table(["#", "Class", "Confidence", "Bounding box"],
                                   det_rows, [12*mm, 42*mm, 40*mm, 82*mm])
        else:
            story.append(Paragraph(
                "No detections were recorded for this inference event, or the inference path returned "
                "no detection list.",
                body
            ))
    else:
        story.append(Paragraph("Phase 7 has not produced a real inference result.", body))

    # ---------- Phase 8 ----------
    story.append(Paragraph("Phase 8 — Inference / Output Integrity", section))
    if isinstance(phase8, dict):
        integrity = phase8.get("integrity")
        rows = []
        add_kv(rows, "Execution status", phase8.get("status"))
        if isinstance(integrity, dict):
            for k in ("passed", "valid", "disposition", "confidence", "reason", "detail",
                      "input_hash", "model_sha256", "output_hash", "record_hash"):
                if k in integrity:
                    add_kv(rows, k.replace("_", " ").title(), integrity[k])
        else:
            add_kv(rows, "Integrity evidence", phase8.get("detail") or integrity)
        add_kv_table(rows)
    else:
        story.append(Paragraph(
            "Phase 8 has not produced a real integrity result for this report.",
            body
        ))

    # ---------- Phase 9 ----------
    story.append(Paragraph("Phase 9 — Audit Trail & Reproducibility", section))
    record_hash = result.get("record_hash")
    if not record_hash and isinstance(phase9, dict):
        record_hash = phase9.get("record_hash")
    if not record_hash:
        record_hash = inference.get("record_hash") if isinstance(inference, dict) else None

    ledger_valid = None
    ledger_message = None
    try:
        # Verify only the audit chain belonging to this report/run.
        # Verifying the entire persistent ledger here can fail because an
        # older/unrelated run may contain a bad or legacy chain; that must
        # not invalidate the current run's verified audit record.
        report_run_id = result.get("run_id")
        if report_run_id:
            ledger_valid, ledger_message = ledger.verify(str(report_run_id))
        else:
            ledger_valid, ledger_message = ledger.verify()
    except Exception as exc:
        ledger_message = f"Ledger verification unavailable: {exc}"

    rows = []
    add_kv(rows, "Execution status", phase9.get("status") if isinstance(phase9, dict) else "Not available")
    add_kv(rows, "Inference record hash", record_hash or "Not available")
    add_kv(rows, "Audit ledger verification at report time",
           "PASS" if ledger_valid is True else ("FAIL" if ledger_valid is False else "NOT AVAILABLE"))
    add_kv(rows, "Ledger message", ledger_message or "No ledger verification message returned.")
    add_kv_table(rows)
    story.append(Paragraph(
        "The record hash is the evidence identifier for the inference event. A valid ledger check "
        "supports tamper-evident auditability of the stored event chain; it does not by itself prove "
        "that the model is safe.",
        small
    ))

    # ---------- Governance ----------
    story.append(Paragraph("Analyst-Facing Assurance & Governance", section))
    governance_rows = [
        ("Affected asset", f"{filename} ({model_type})"),
        ("Model identity", f"SHA-256 {digest}"),
        ("Access assumption", access_level),
        ("Disposition", phase4.get("disposition") or "review"),
        ("Human-readable reason", (
            "The model identity was verified, but the YOLO-specific Phase 4 trigger/backdoor "
            "assessment is currently a declared placeholder."
            if model_type.lower() == "yolo" and phase4.get("status") == "placeholder"
            else "See the Phase 4 findings and Phase 6-9 evidence above."
        )),
        ("Recommended action represented by this report", (
            "Review the model before treating it as fully assured; downstream inference evidence "
            "is reported separately."
            if model_type.lower() == "yolo" and phase4.get("status") == "placeholder"
            else "Use the recorded disposition and evidence together with the stated limitations."
        )),
    ]
    add_section_table(["Governance field", "Recorded information"], governance_rows,
                      [52*mm, 124*mm])

    # ---------- Coverage / limitations ----------
    story.append(Paragraph("Coverage Statement & Known Limitations", section))
    limits = []
    if model_type.lower() == "yolo":
        limits.append("YOLO-specific Phase 4 trigger/backdoor detection is a declared Round-1 placeholder.")
    if not reference_data:
        limits.append("No reference dataset metadata was supplied in the report payload.")
    if not isinstance(phase6, dict) or phase6.get("status") != "real":
        limits.append("Distribution-shift evidence is not real unless a Phase 6 inference event was executed.")
    if not isinstance(phase8, dict) or phase8.get("status") != "real":
        limits.append("Inference-output integrity evidence is not real unless a Phase 8 event was executed.")
    if not isinstance(phase9, dict) or phase9.get("status") != "real":
        limits.append("A real inference audit event is not available unless Phase 9 was executed.")
    if isinstance(phase3_data, dict):
        for x in phase3_data.get("limitations", []) or []:
            if x not in limits:
                limits.append(x)
    for x in phase4.get("limitations", []) or []:
        if x not in limits:
            limits.append(x)
    if limits:
        add_bullets(limits)
    else:
        story.append(Paragraph("No additional limitations were returned by the executed stages.", body))

    story.append(Paragraph("Offline / Air-Gapped Operation", section))
    story.append(Paragraph(
        "The TrustCV design is intended to execute locally without cloud inference dependencies. "
        "This run used the uploaded model/reference artifacts and the local TrustCV backend. "
        "The report does not claim external-service verification that was not actually performed.",
        body
    ))

    # ---------- Evidence appendix ----------
    story.append(PageBreak())
    story.append(Paragraph("Evidence Appendix — Human-Readable Technical Record", section))
    story.append(Paragraph(
        "This appendix preserves the key machine-generated values needed to reproduce or audit the "
        "assessment without dumping raw JSON. Values are grouped by pipeline phase.",
        body
    ))
    appendix = [
        ("Model filename", filename),
        ("Model type", model_type),
        ("SHA-256", digest),
        ("MIRAD artifact ID", artifact.get("artifact_id") if isinstance(artifact, dict) else "Not available"),
        ("MIRAD version", artifact.get("version") if isinstance(artifact, dict) else "Not available"),
        ("Phase 4 status", phase4.get("status")),
        ("Phase 4 disposition", phase4.get("disposition")),
        ("Phase 6 status", phase6.get("status") if isinstance(phase6, dict) else "Not available"),
        ("Phase 7 status", phase7.get("status") if isinstance(phase7, dict) else "Not available"),
        ("Phase 8 status", phase8.get("status") if isinstance(phase8, dict) else "Not available"),
        ("Phase 9 status", phase9.get("status") if isinstance(phase9, dict) else "Not available"),
        ("Inference record hash", record_hash or "Not available"),
    ]
    add_section_table(["Evidence item", "Recorded value"], appendix, [62*mm, 114*mm])

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "Source alignment: SIH26228 — “Trustworthy Computer Vision Integrity Assurance for Data, "
        "Models and Inference Outputs in Multi-Contributor Pipelines”, Ministry of defence (MoD), "
        "Indian Army (DGIS). The report is an implementation-time assurance artifact and should be "
        "read together with the project's source code, architecture, setup notes, audit log, and "
        "coverage statement.",
        small
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)



def _pdf_text(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, (dict, list)):
        return json.dumps(_jsonable(value), indent=2, ensure_ascii=False, default=str)
    return str(value)


def _build_audit_log_pdf(run: dict[str, Any], output_path: Path) -> None:
    """Human-readable Phase 9 audit log generated only after MIRAD chain verification."""
    run_id = run["run_id"]
    valid, message = ledger.verify(run_id)
    if not valid:
        raise RuntimeError(f"Audit chain verification failed: {message}")
    events = ledger.entries(run_id)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("AuditTitle", parent=styles["Title"], fontSize=22, leading=26, textColor=colors.HexColor("#172033"), spaceAfter=6)
    section = ParagraphStyle("AuditSection", parent=styles["Heading2"], fontSize=14, leading=18, textColor=colors.HexColor("#172033"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("AuditBody", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#344054"), spaceAfter=4)
    mono = ParagraphStyle("AuditMono", parent=body, fontName="Courier", fontSize=7, leading=9)
    small = ParagraphStyle("AuditSmall", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#667085"))
    story = [Paragraph("TrustCV Audit Log", title), Paragraph("MIRAD-backed chronological, tamper-evident security event record", small), Spacer(1, 5)]
    meta = [
        [Paragraph("Run ID", small), Paragraph(_pdf_text(run_id), mono)],
        [Paragraph("Model", small), Paragraph(_pdf_text(run.get("filename")), body)],
        [Paragraph("Model SHA-256", small), Paragraph(_pdf_text(run.get("model_sha256")), mono)],
        [Paragraph("MIRAD audit chain", small), Paragraph("PASS", body)],
        [Paragraph("Verification message", small), Paragraph(_pdf_text(message), body)],
        [Paragraph("Event count", small), Paragraph(str(len(events)), body)],
    ]
    t=Table(meta,colWidths=[45*mm,131*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F2F4F7")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D0D5DD")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),5)])); story.append(t)
    story.append(Paragraph("Chronological Event Register", section))
    rows=[["Seq","Timestamp","Phase","Event","Status","Event Hash"]]
    for e in events:
        payload=e.get("payload") or {}
        phase=payload.get("phase") or e.get("event_type")
        status=payload.get("status") or ("PASS" if payload.get("verified") is True else "—")
        rows.append([str(e.get("sequence")),str(e.get("timestamp")),str(phase),str(e.get("event_type")),str(status),str(e.get("current_hash"))[:20]+"…"])
    data=[[Paragraph(str(x),small) for x in rows[0]]]
    for r in rows[1:]: data.append([Paragraph(str(x),mono if i in (0,5) else body) for i,x in enumerate(r)])
    table=Table(data,colWidths=[10*mm,31*mm,24*mm,40*mm,22*mm,49*mm],repeatRows=1); table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAECF0")),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D0D5DD")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),4)])); story.append(table)
    story.append(PageBreak())
    story.append(Paragraph("Security Evidence by Event", section))
    for e in events:
        story.append(Paragraph(f"Event {e.get('sequence')} · {e.get('event_type')}", section))
        story.append(Paragraph(_pdf_text(e.get("payload") or {}), mono))
    doc=SimpleDocTemplate(str(output_path),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=16*mm,bottomMargin=16*mm)
    doc.build(story)


def _build_provenance_pdf(run: dict[str, Any], output_path: Path) -> None:
    """Human-readable lifecycle provenance; intentionally distinct from audit chronology."""
    styles=getSampleStyleSheet()
    title=ParagraphStyle("ProvTitle",parent=styles["Title"],fontSize=22,leading=26,textColor=colors.HexColor("#172033"),spaceAfter=6)
    section=ParagraphStyle("ProvSection",parent=styles["Heading2"],fontSize=14,leading=18,textColor=colors.HexColor("#172033"),spaceBefore=10,spaceAfter=6)
    body=ParagraphStyle("ProvBody",parent=styles["BodyText"],fontSize=8.5,leading=12,textColor=colors.HexColor("#344054"),spaceAfter=4)
    mono=ParagraphStyle("ProvMono",parent=body,fontName="Courier",fontSize=7,leading=9)
    small=ParagraphStyle("ProvSmall",parent=body,fontSize=7.5,leading=10,textColor=colors.HexColor("#667085"))
    story=[Paragraph("TrustCV Model Provenance",title),Paragraph("Lifecycle lineage from model identity and reference data through inference outputs",small)]
    meta=[
        [Paragraph("Run ID",small),Paragraph(_pdf_text(run.get("run_id")),mono)],
        [Paragraph("Model",small),Paragraph(_pdf_text(run.get("filename")),body)],
        [Paragraph("Model type",small),Paragraph(_pdf_text(run.get("model_type")),body)],
        [Paragraph("Model SHA-256",small),Paragraph(_pdf_text(run.get("model_sha256")),mono)],
        [Paragraph("Audit verification",small),Paragraph(_pdf_text((run.get("audit_verification") or {}).get("valid")),body)],
        [Paragraph("Digital signature",small),Paragraph("NOT AVAILABLE — signing stage intentionally not enabled",body)],
    ]
    t=Table(meta,colWidths=[45*mm,131*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F2F4F7")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D0D5DD")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),5)])); story.append(t)
    story.append(Paragraph("Lifecycle Checkpoints",section))
    cp=run.get("checkpoints") or {}
    cp_rows=[["Phase","Actual SHA-256","Expected SHA-256","Result","Audit event"]]
    for phase in ["phase3","phase4_pre","phase4","phase6_pre","phase6","phase7","phase8"]:
        c=cp.get(phase)
        if not c: continue
        cp_rows.append([phase,c.get("actual_model_sha256"),c.get("expected_model_sha256"),"PASS" if c.get("verified") else "FAIL",c.get("audit_event_id")])
    data=[[Paragraph(x,small) for x in cp_rows[0]]]+[[Paragraph(_pdf_text(x),mono if i in (1,2,4) else body) for i,x in enumerate(r)] for r in cp_rows[1:]]
    t=Table(data,colWidths=[23*mm,43*mm,43*mm,18*mm,49*mm],repeatRows=1); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAECF0")),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D0D5DD")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),4)])); story.append(t)
    story.append(PageBreak())
    story.append(Paragraph("Input → Output Traceability",section))
    for prov in run.get("provenance_records") or []:
        m=prov.get("metadata") or {}
        story.append(Paragraph(f"{prov.get('event_id')} · Frame {m.get('frame_index')}",section))
        lines={"Input digest":prov.get("input_digest"),"Dataset":f"{prov.get('dataset_identity')} · {prov.get('dataset_digest')}","Model":f"{prov.get('model_identity')} · {prov.get('model_digest')}","Preprocessing digest":prov.get("preprocessing_digest"),"Inference config digest":prov.get("inference_config_digest"),"Output digest":prov.get("output_digest"),"Audit event":m.get("audit_event_id"),"Replay check":prov.get("replay_verification")}
        rows=[[Paragraph(k,small),Paragraph(_pdf_text(v),mono if "digest" in k.lower() or k=="Audit event" else body)] for k,v in lines.items()]
        t=Table(rows,colWidths=[45*mm,131*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F2F4F7")),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D0D5DD")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),4)])); story.append(t); story.append(Spacer(1,4))
    story.append(Paragraph("Anomalous / Malicious-output Candidates",section))
    findings=run.get("findings") or []
    if not findings: story.append(Paragraph("No evidence-based anomaly candidates were recorded for this run.",body))
    for f in findings:
        story.append(Paragraph(f"{f.get('finding_id')} · {f.get('finding_type')} · {f.get('recommended_disposition')}",body)); story.append(Paragraph(_pdf_text(f),mono))
    story.append(Paragraph("Coverage & Limitations",section))
    limits=[
        "Hash checkpoints establish model artifact continuity; they do not prove that a model was benign before upload.",
        "MIRAD registration establishes a reference-known identity; registration is not a model-safety verdict.",
        "Audit chronology and provenance lineage are separate records with different purposes.",
        "Camera inference has no ground-truth labels, so measured accuracy is not fabricated.",
        "OOD evaluates input distribution signals; it does not mean that the expected object must be present.",
        "TRACE is used as a declared TRACE-inspired YOLO behavioral adaptation with demo calibration limits.",
        "Digital signatures are not enabled in this stage; no issuer or signature is fabricated.",
    ]
    for x in limits: story.append(Paragraph("• "+x,body))
    doc=SimpleDocTemplate(str(output_path),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=16*mm,bottomMargin=16*mm)
    doc.build(story)


@app.post("/api/audit/pdf")
async def generate_audit_pdf(run_id: str = Form(...)):
    run=RUNS.get(run_id)
    if not run: return {"error":"Run not found in the current TrustCV server session."}
    try:
        valid,message=ledger.verify(run_id)
        if not valid: return {"error":f"Audit chain verification failed: {message}"}
        name=f"TrustCV_Audit_Log_{run_id}.pdf"; path=RUNTIME_DIR/name
        _build_audit_log_pdf(run,path)
        return {"success":True,"filename":name,"run_id":run_id}
    except Exception as exc:
        return {"error":f"Audit PDF generation failed: {type(exc).__name__}: {exc}"}


@app.post("/api/provenance/pdf")
async def generate_provenance_pdf(run_id: str = Form(...)):
    run=RUNS.get(run_id)
    if not run: return {"error":"Run not found in the current TrustCV server session."}
    try:
        name=f"TrustCV_Model_Provenance_{run_id}.pdf"; path=RUNTIME_DIR/name
        _build_provenance_pdf(run,path)
        return {"success":True,"filename":name,"run_id":run_id}
    except Exception as exc:
        return {"error":f"Provenance PDF generation failed: {type(exc).__name__}: {exc}"}

@app.post("/api/report/pdf")
async def generate_report_pdf(report: str = Form(...)):
    """Generate a human-readable PDF for exactly the requested phase/report scope."""
    try:
        result = json.loads(report)
        if not isinstance(result, dict):
            return {"error": "Report payload must be a JSON object."}
    except Exception as exc:
        return {"error": f"Invalid report JSON: {exc}"}

    try:
        report_id = sha256_bytes(canonical_json(result))
        output_path = RUNTIME_DIR / f"trustcv_report_{report_id}.pdf"
        _build_trust_report_pdf(result, output_path)
        if not output_path.is_file():
            return {"error": "PDF generation completed without creating the report file."}
        return {
            "success": True,
            "filename": output_path.name,
            "model_ref": result.get("model_ref") or result.get("sha256"),
            "path": str(output_path),
        }
    except Exception as exc:
        return {"error": f"PDF generation failed: {type(exc).__name__}: {exc}"}


@app.post("/api/verify/model")
async def verify_model(
    file: UploadFile = File(...),
    reference_dataset: UploadFile | None = File(None),
    access_level: str = Form("white_box"),
):
    data = await file.read()
    digest = sha256_bytes(data)
    filename = file.filename or "uploaded_model.pt"
    path = RUNTIME_DIR / f"{digest}{Path(filename).suffix.lower() or '.pt'}"
    path.write_bytes(data)

    model_type, type_evidence = identify_model_type(path)
    if model_type == "unknown":
        path.unlink(missing_ok=True)
        return {
            "passed": False,
            "model_type": "unknown",
            "sha256": digest,
            "checks": [{"id": "model_type", "label": "Model Type Detection", "passed": False, "detail": type_evidence}],
            "error": "Unsupported or unrecognized model format.",
        }

    mirad = _mirad_model_check(path, filename, model_type)

    reference_images = None
    reference_info = None

    if reference_dataset is not None:
        dataset_bytes = await reference_dataset.read()
        if not dataset_bytes:
            path.unlink(missing_ok=True)
            return {
                "passed": False,
                "model_type": model_type,
                "sha256": digest,
                "error": "The uploaded reference dataset is empty.",
            }

        suffix = Path(reference_dataset.filename or "reference.zip").suffix.lower() or ".zip"
        with tempfile.NamedTemporaryFile(
            prefix="trustcv_reference_",
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(dataset_bytes)
            dataset_path = Path(tmp.name)

        try:
            if model_type == "yolo":
                # Do not materialize an entire COCO archive into RAM just to test compatibility.
                # The compatibility inspector reads metadata and validates a bounded image sample.
                yolo_dataset_info = inspect_reference_zip(dataset_path)
                yolo_model_info = inspect_yolo_model(path)
                yolo_compatibility = validate_yolo_reference(yolo_model_info, yolo_dataset_info)
                reference_images = None
                reference_info = yolo_dataset_info
                reference_info["compatibility"] = yolo_compatibility
            else:
                dataset_result = load_dataset_from_zip(
                    dataset_path,
                    max_samples=None,
                )
                reference_images = dataset_result.images
                reference_info = dataset_summary(dataset_result)
        except Exception as exc:
            dataset_path.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            return {
                "passed": False,
                "model_type": model_type,
                "sha256": digest,
                "error": f"Reference dataset could not be loaded: {exc}",
            }
        finally:
            dataset_path.unlink(missing_ok=True)

    if model_type == "smallcnn":
        if reference_images is None:
            return {
                "passed": False,
                "model_type": model_type,
                "sha256": digest,
                "model_ref": digest,
                "mirad": mirad,
                "phase3": mirad,
                "checks": mirad["checks"],
                "error": "SmallCNN Phase 4 requires a clean reference dataset. Upload a CIFAR-10 binary ZIP or another compatible clean reference dataset.",
            }

        compatibility = validate_dataset_for_model(
            dataset_result,
            expected_channels=3,
            expected_height=32,
            expected_width=32,
            expected_classes=10,
        )
        reference_info["compatibility"] = compatibility

        if not compatibility["compatible"]:
            return _jsonable({
                "passed": False,
                "model_type": model_type,
                "model_type_evidence": type_evidence,
                "filename": filename,
                "sha256": digest,
                "model_ref": digest,
                "mirad": mirad,
                "phase3": mirad,
                "reference_data": reference_info,
                "phase4": {
                    "phase": "phase4_model_integrity",
                    "status": "blocked",
                    "disposition": "review",
                    "access_level": access_level,
                    "flags": [],
                    "reference_data": reference_info,
                    "limitations": [
                        "Phase 4 was not executed because the supplied reference dataset is incompatible with the current SmallCNN testbed.",
                        *compatibility["errors"],
                    ],
                },
                "error": "Reference dataset is incompatible with the current SmallCNN testbed.",
            })

        phase4 = _phase4_smallcnn(
            path,
            access_level,
            reference_images,
            reference_info,
        )
        phase6_9 = _smallcnn_phase6_9_mock(digest)
    elif model_type == "yolo":
        compatibility = (reference_info or {}).get("compatibility") if reference_info else None
        if reference_dataset is None:
            phase4 = {
                "phase": "phase4_model_integrity",
                "status": "blocked",
                "disposition": "review",
                "access_level": access_level,
                "flags": [],
                "limitations": ["A clean reference dataset is required before YOLO Phase 4 can run."],
            }
        elif not compatibility or not compatibility.get("compatible"):
            errors = (compatibility or {}).get("errors", ["Reference dataset is incompatible with the uploaded YOLO object detector."])
            phase4 = {
                "phase": "phase4_model_integrity",
                "status": "blocked",
                "disposition": "review",
                "access_level": access_level,
                "flags": [{"check": "yolo_dataset_compatibility", "disposition": "blocked", "confidence": None, "reason": errors[0] if errors else "Reference dataset is incompatible.", "evidence": {"compatibility": compatibility}}],
                "reference_data": reference_info,
                "limitations": ["Phase 4 was not executed because the supplied reference dataset is incompatible with the uploaded YOLO detector.", *errors],
            }
        else:
            phase4 = {
                "phase": "phase4_model_integrity",
                "status": "placeholder",
                "disposition": "review",
                "access_level": access_level,
                "flags": [{"check": "yolo_dataset_compatibility", "disposition": "compatible", "confidence": None, "reason": "Reference dataset passed the compatibility gate. YOLO trigger analysis is the next implementation stage.", "evidence": {"compatibility": compatibility}}],
                "reference_data": reference_info,
                "limitations": ["YOLO-specific trigger/backdoor analysis remains to be implemented."],
            }
        phase6_9 = {
            "phase6": {"status": "ready", "detail": "Existing YOLO OOD gate will run after an input is supplied."},
            "phase7": {"status": "ready", "detail": "Existing YOLO inference/live-CV path will run after an input is supplied."},
            "phase8": {"status": "ready", "detail": "Existing YOLO inference-integrity path will run after an input is supplied."},
            "phase9": {"status": "ready", "detail": "Existing audit-ledger path will record the inference event."},
        }

    _log_event({
        "schema_version": "4.0",
        "timestamp_utc": utc_now(),
        "type": "model_assurance",
        "filename": filename,
        "sha256": digest,
        "model_type": model_type,
        "mirad": mirad["mirad"],
        "phase4": phase4,
        "reference_data": reference_info,
    })

    overall_passed = phase4.get("disposition") == "accept"

    return _jsonable({
        "passed": overall_passed,
        "model_type": model_type,
        "model_type_evidence": type_evidence,
        "filename": filename,
        "sha256": digest,
        "model_ref": digest,
        "mirad": mirad,
        "phase3": mirad,
        "phase4": phase4,
        "phase6_9": phase6_9,
    })


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    model_ref: str = Form(...),
    run_id: str | None = Form(None),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    """Run YOLO Phase 6-9 with MIRAD checkpoints, evidence, provenance and audit."""
    candidates = list(RUNTIME_DIR.glob(f"{model_ref}.*"))
    if not candidates:
        return {"error": "Model session not found. Upload the model again."}
    weights_path = candidates[0]
    model_type, _ = identify_model_type(weights_path)
    if model_type == "smallcnn":
        return _smallcnn_phase6_9_mock(model_ref)
    if model_type != "yolo":
        return {"error": f"Phase 6-9 inference adapter unavailable for model type: {model_type}"}

    model_sha = sha256_file(weights_path)
    if not run_id or run_id not in RUNS:
        run_id = _new_run(weights_path.name, model_type, model_sha, weights_path)
    RUNS[run_id]["model_path"] = str(weights_path)

    p6_pre = _checkpoint_model(run_id, "phase6_pre", weights_path, RUNS[run_id]["model_sha256"])
    if not p6_pre["verified"]:
        return {"error": "Model hash verification failed before Phase 6.", "run_id": run_id, "phase6": {"status": "blocked", "disposition": "quarantine"}, "hash_checkpoint": p6_pre}

    detector, gate, integrity = _get_yolo_pipeline(weights_path)
    data = await file.read()
    input_digest = f"sha256:{sha256_bytes(data)}"
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image", "run_id": run_id}

    g = gate.check(frame, conf, iou)
    dets = detector.predict(frame, conf=conf, iou=iou) if not g.ood else []
    phase6 = {"status": "real", "inDistribution": not g.ood, "confidence": round(float(g.mls), 4), "detail": ", ".join(g.reasons) if g.reasons else "No OOD signals triggered"}
    phase7 = {"status": "real", "detections": dets, "conf_threshold": conf, "iou_threshold": iou}
    p6 = _checkpoint_model(run_id, "phase6", weights_path, RUNS[run_id]["model_sha256"])
    if not p6["verified"]:
        return {"error": "Model hash verification failed after Phase 6.", "run_id": run_id, "phase6": phase6, "hash_checkpoint": p6}

    p7 = _checkpoint_model(run_id, "phase7", weights_path, RUNS[run_id]["model_sha256"])
    if not p7["verified"]:
        return {"error": "Model hash verification failed after Phase 7.", "run_id": run_id, "phase6": phase6, "phase7": phase7, "hash_checkpoint": p7}

    integrity_result = integrity.evaluate_frame(frame, dets, g, conf, iou, True)
    phase8 = {"status": "real", "integrity": integrity_result}
    p8 = _checkpoint_model(run_id, "phase8", weights_path, RUNS[run_id]["model_sha256"])
    if not p8["verified"]:
        return {"error": "Model hash verification failed after Phase 8.", "run_id": run_id, "phase6": phase6, "phase7": phase7, "phase8": phase8, "hash_checkpoint": p8}

    output_obj = {"phase6": phase6, "phase7": phase7, "phase8": phase8, "conf_threshold": conf, "iou_threshold": iou}
    run = RUNS[run_id]
    dataset_digest = None
    dataset_identity = "reference-data:not-provided"
    p4 = run.get("phase4") or {}
    ref = p4.get("reference_data") if isinstance(p4, dict) else None
    if isinstance(ref, dict):
        dataset_digest = ref.get("supplied_digest")
        dataset_identity = ref.get("supplied_filename") or ref.get("format") or dataset_identity
    provenance = _build_provenance_for_output(run_id, frame_index=1, input_digest=input_digest, output_obj=output_obj, dataset_digest=dataset_digest, dataset_identity=dataset_identity, phase6=phase6, phase7=phase7, phase8=phase8, model_path=weights_path)

    output_event = _log_event({
        "type": "inference.output", "phase": "phase8", "run_id": run_id, "frame_index": 1,
        "source": file.filename, "input_sha256": input_digest.replace("sha256:", ""),
        "model_sha256": model_sha, "output_sha256": provenance["output_digest"].replace("sha256:", ""),
        "phase6": phase6, "phase7": phase7, "phase8": phase8, "provenance_id": provenance["event_id"],
    }, run_id=run_id, event_type="inference.output")
    provenance.setdefault("metadata", {})["audit_event_id"] = output_event["audit_id"]
    evidence, finding = _make_evidence_and_finding(run_id, frame_index=1, input_digest=input_digest, phase6=phase6, phase7=phase7, phase8=phase8, provenance_id=provenance["event_id"], audit_event_id=output_event["audit_id"])
    if finding:
        _log_event({"type": "security.finding", "phase": "phase8", "run_id": run_id, "finding": finding, "evidence": evidence}, run_id=run_id, event_type="security.finding")

    RUNS[run_id]["phase6"] = phase6
    RUNS[run_id]["phase7"] = phase7
    RUNS[run_id]["phase8"] = phase8
    phase9_event = _log_event({
        "type": "phase9.audit_summary", "phase": "phase9", "run_id": run_id,
        "model_sha256": model_sha, "provenance_id": provenance["event_id"],
        "finding_ids": [finding["finding_id"]] if finding else [],
    }, run_id=run_id, event_type="phase9.audit_summary")
    audit_valid, audit_message = ledger.verify(run_id)
    phase9 = {"status": "real", "record_hash": phase9_event["current_hash"], "audit_event_hash": phase9_event["current_hash"], "audit_event_id": phase9_event["audit_id"], "events_recorded": len(ledger.entries(run_id)), "ledger_verification": audit_valid, "ledger_message": audit_message, "run_id": run_id, "model_sha256": model_sha, "timestamp_utc": utc_now()}
    RUNS[run_id]["phase9"] = phase9
    RUNS[run_id]["audit_verification"] = {"valid": audit_valid, "message": audit_message}

    return _jsonable({"run_id": run_id, "phase6": {**phase6, "hash_checkpoint": p6}, "phase7": {**phase7, "hash_checkpoint": p7}, "phase8": {**phase8, "hash_checkpoint": p8}, "phase9": phase9, "record_hash": phase9_event["current_hash"], "provenance": provenance, "findings": RUNS[run_id]["findings"], "evidence": RUNS[run_id]["evidence"]})


@app.post("/api/analyze/batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    model_ref: str = Form(...),
    run_id: str | None = Form(None),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    """Run YOLO Phase 6-9 over camera frames with one auditable run lineage."""
    candidates = list(RUNTIME_DIR.glob(f"{model_ref}.*"))
    if not candidates:
        return {"error": "Model session not found. Upload the model again."}
    weights_path = candidates[0]
    model_type, _ = identify_model_type(weights_path)
    if model_type != "yolo":
        return {"error": "Live camera batch inference is currently available for YOLO models only."}
    if not 6 <= len(files) <= 9:
        return {"error": "Live camera inference expects 6 to 9 captured frames."}
    model_sha = sha256_file(weights_path)
    if not run_id or run_id not in RUNS:
        run_id = _new_run(weights_path.name, model_type, model_sha, weights_path)
    RUNS[run_id]["model_path"] = str(weights_path)
    p6_pre = _checkpoint_model(run_id, "phase6_pre", weights_path, RUNS[run_id]["model_sha256"])
    if not p6_pre["verified"]:
        return {"error": "Model hash verification failed before camera analysis.", "run_id": run_id, "hash_checkpoint": p6_pre}

    detector, gate, integrity = _get_yolo_pipeline(weights_path)
    frame_results = []
    input_digests = []
    for index, upload in enumerate(files, 1):
        data = await upload.read()
        input_digest = f"sha256:{sha256_bytes(data)}"
        input_digests.append(input_digest)
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            frame_results.append({"frame_index": index, "source": upload.filename, "error": "Could not decode image", "phase6": {"status": "error", "inDistribution": None}, "phase7": {"status": "error", "detections": []}, "phase8": {"status": "error"}, "phase9": {"status": "not_recorded"}})
            continue
        g = gate.check(frame, conf, iou)
        dets = detector.predict(frame, conf=conf, iou=iou) if not g.ood else []
        phase6 = {"status": "real", "inDistribution": not g.ood, "confidence": round(float(g.mls), 4), "detail": ", ".join(g.reasons) if g.reasons else "No OOD signals triggered"}
        phase7 = {"status": "real", "detections": dets, "conf_threshold": conf, "iou_threshold": iou}
        integrity_result = integrity.evaluate_frame(frame, dets, g, conf, iou, True)
        phase8 = {"status": "real", "integrity": integrity_result}
        frame_results.append({"frame_index": index, "source": upload.filename, "phase6": phase6, "phase7": phase7, "phase8": phase8, "input_digest": input_digest})

    p6 = _checkpoint_model(run_id, "phase6", weights_path, RUNS[run_id]["model_sha256"])
    if not p6["verified"]:
        return {"error": "Model hash verification failed after Phase 6.", "run_id": run_id, "frames": frame_results, "hash_checkpoint": p6}
    p7 = _checkpoint_model(run_id, "phase7", weights_path, RUNS[run_id]["model_sha256"])
    if not p7["verified"]:
        return {"error": "Model hash verification failed after Phase 7.", "run_id": run_id, "frames": frame_results, "hash_checkpoint": p7}
    p8 = _checkpoint_model(run_id, "phase8", weights_path, RUNS[run_id]["model_sha256"])
    if not p8["verified"]:
        return {"error": "Model hash verification failed after Phase 8.", "run_id": run_id, "frames": frame_results, "hash_checkpoint": p8}

    run = RUNS[run_id]
    p4 = run.get("phase4") or {}
    ref = p4.get("reference_data") if isinstance(p4, dict) else None
    dataset_digest = ref.get("supplied_digest") if isinstance(ref, dict) else None
    dataset_identity = (ref.get("supplied_filename") or ref.get("format") or "reference-data:not-provided") if isinstance(ref, dict) else "reference-data:not-provided"
    for frame_result in frame_results:
        if "error" in frame_result:
            continue
        output_obj = {"phase6": frame_result["phase6"], "phase7": frame_result["phase7"], "phase8": frame_result["phase8"], "conf_threshold": conf, "iou_threshold": iou}
        provenance = _build_provenance_for_output(run_id, frame_index=frame_result["frame_index"], input_digest=frame_result["input_digest"], output_obj=output_obj, dataset_digest=dataset_digest, dataset_identity=dataset_identity, phase6=frame_result["phase6"], phase7=frame_result["phase7"], phase8=frame_result["phase8"], model_path=weights_path)
        output_event = _log_event({"type": "inference.output", "phase": "phase8", "run_id": run_id, "frame_index": frame_result["frame_index"], "source": frame_result["source"], "input_sha256": frame_result["input_digest"].replace("sha256:", ""), "model_sha256": model_sha, "output_sha256": provenance["output_digest"].replace("sha256:", ""), "phase6": frame_result["phase6"], "phase7": frame_result["phase7"], "phase8": frame_result["phase8"], "provenance_id": provenance["event_id"]}, run_id=run_id, event_type="inference.output")
        provenance.setdefault("metadata", {})["audit_event_id"] = output_event["audit_id"]
        evidence, finding = _make_evidence_and_finding(run_id, frame_index=frame_result["frame_index"], input_digest=frame_result["input_digest"], phase6=frame_result["phase6"], phase7=frame_result["phase7"], phase8=frame_result["phase8"], provenance_id=provenance["event_id"], audit_event_id=output_event["audit_id"])
        if finding:
            _log_event({"type": "security.finding", "phase": "phase8", "run_id": run_id, "frame_index": frame_result["frame_index"], "finding": finding, "evidence": evidence}, run_id=run_id, event_type="security.finding")
        frame_result["phase9"] = {"status": "real", "provenance_id": provenance["event_id"], "output_event_id": output_event["audit_id"], "output_event_hash": output_event["current_hash"]}

    valid = [r for r in frame_results if "error" not in r]
    ood_count = sum(r["phase6"].get("inDistribution") is False for r in valid)
    detections = [d for r in valid for d in r["phase7"].get("detections", [])]
    dispositions = [str(r.get("phase8", {}).get("integrity", {}).get("disposition", "")).lower() for r in valid]
    integrity_disposition = "quarantine" if any(d in {"quarantine", "blocked", "reject", "rejected", "fail", "failed", "flagged"} for d in dispositions) else ("review" if any(d in {"review", "warning", "warn"} for d in dispositions) else "accept")
    hashes = [r["phase9"]["output_event_hash"] for r in valid if r.get("phase9", {}).get("output_event_hash")]
    RUNS[run_id]["phase6"] = {"status": "real", "inDistribution": ood_count == 0, "confidence": round(float(np.mean([r["phase6"]["confidence"] for r in valid])), 4) if valid else None, "detail": f"{ood_count} of {len(valid)} frames triggered OOD/shift signals." if valid else "No frames could be analyzed.", "frames_in_distribution": sum(r["phase6"].get("inDistribution") is True for r in valid), "frames_out_of_distribution": ood_count}
    RUNS[run_id]["phase7"] = {"status": "real", "detections": detections, "frame_count": len(frame_results)}
    RUNS[run_id]["phase8"] = {"status": "real", "disposition": integrity_disposition, "frame_count": len(frame_results), "flagged_frames": sum(d != "accept" for d in dispositions)}
    phase9_event = _log_event({"type": "phase9.audit_summary", "phase": "phase9", "run_id": run_id, "model_sha256": model_sha, "frame_count": len(frame_results), "output_event_hashes": hashes, "provenance_ids": [r.get("phase9", {}).get("provenance_id") for r in valid]}, run_id=run_id, event_type="phase9.audit_summary")
    audit_valid, audit_message = ledger.verify(run_id)
    phase9 = {"status": "real", "record_hash": phase9_event["current_hash"], "audit_event_hash": phase9_event["current_hash"], "audit_event_id": phase9_event["audit_id"], "record_hashes": [phase9_event["current_hash"], *hashes], "events_recorded": len(ledger.entries(run_id)), "ledger_verification": audit_valid, "ledger_message": audit_message, "run_id": run_id, "model_sha256": model_sha, "timestamp_utc": utc_now()}
    RUNS[run_id]["phase9"] = phase9
    RUNS[run_id]["audit_verification"] = {"valid": audit_valid, "message": audit_message}

    return _jsonable({"mode": "camera_batch", "run_id": run_id, "frame_count": len(frame_results), "phase6": {**RUNS[run_id]["phase6"], "hash_checkpoint": p6}, "phase7": {**RUNS[run_id]["phase7"], "hash_checkpoint": p7}, "phase8": {**RUNS[run_id]["phase8"], "hash_checkpoint": p8}, "phase9": phase9, "frames": frame_results, "record_hash": phase9_event["current_hash"], "provenance": RUNS[run_id]["provenance_records"], "findings": RUNS[run_id]["findings"], "evidence": RUNS[run_id]["evidence"]})

@app.post("/api/verify/dataset")
async def verify_dataset(file: UploadFile = File(...)):
    # Dataset work is intentionally left for the next integration stage.
    data = await file.read()
    digest = sha256_bytes(data)
    return {
        "passed": False,
        "sha256": digest,
        "checks": [
            {"id": "hash", "label": "Hash Verification", "passed": True, "detail": f"sha256:{digest[:12]}..."},
            {"id": "mirad", "label": "MIRAD Dataset Verification", "passed": None, "detail": "Dataset MIRAD integration is not wired in this stage."},
        ],
    }


@app.get("/api/ledger/verify")
def verify_ledger():
    ok, msg = ledger.verify()
    return {"valid": ok, "message": msg}


@app.get("/api/health")
def health():
    return {"status": "ok", "mirad": "enabled", "phase4": _PHASE4_AVAILABLE}
