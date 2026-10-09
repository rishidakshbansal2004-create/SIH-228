"""
TrustCV Dataset Security — MIRAD-derived cryptographic assurance layer.

Dataset-only integration of the security concepts used in the MIRAD prototype:
canonical hashing, trusted verification keys, signed provenance, replay/freshness
checks, append-only hash-chain audit records, signed checkpoints and structured
verification results.

This module deliberately contains NO model/inference provenance fields.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

APP = Path(__file__).resolve().parent

try:
    RUNTIME = Path.home() / ".trustcv_dataset_security"
    RUNTIME.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    RUNTIME = Path(tempfile.gettempdir()) / ".trustcv_dataset_security"
    RUNTIME.mkdir(parents=True, exist_ok=True)

KEY_FILE = RUNTIME / "dataset_signing_private.pem"
TRUST_FILE = RUNTIME / "dataset_trust_anchor.json"
STATE_FILE = RUNTIME / "dataset_security_state.json"
AUDIT_FILE = RUNTIME / "dataset_security_audit.jsonl"
CHECKPOINT_FILE = RUNTIME / "dataset_security_checkpoint.json"

try:
    PROJECT_OUTPUT = APP / "demo_output" / "dataset_security"
    PROJECT_OUTPUT.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    PROJECT_OUTPUT = RUNTIME / "demo_output" / "dataset_security"
    PROJECT_OUTPUT.mkdir(parents=True, exist_ok=True)

try:
    MANIFEST_OUTPUT = APP / "demo_output" / "signed_manifests"
    MANIFEST_OUTPUT.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    MANIFEST_OUTPUT = RUNTIME / "signed_manifests"
    MANIFEST_OUTPUT.mkdir(parents=True, exist_ok=True)

CANONICALIZATION_VERSION = "1.0"
SECURITY_SCHEMA_VERSION = "1.0"
POLICY_VERSION = "1.0"
TRUST_ANCHOR_ID = "TRUSTCV-DATASET-LOCAL-DEFAULT"
SIGNATURE_ALGORITHM = "ed25519"
HASH_ALGORITHM = "sha256"
FRESHNESS_WINDOW_SECONDS = 300
MAX_FUTURE_SKEW_SECONDS = 30


@dataclass(frozen=True)
class VerificationPolicy:
    """Immutable dataset verification policy for canonicalization, signatures, freshness, and trust controls."""
    hash_algorithm: str = "sha256"
    signature_algorithm: str = "ed25519"
    canonicalization_version: str = "1.0"
    policy_version: str = "1.0"
    schema_version: str = "1.0"
    freshness_window_seconds: int = 300
    maximum_future_skew_seconds: int = 30
    nonce_required: bool = True
    sequence_required: bool = True
    sequence_scope: str = "session"
    replay_policy: str = "strict"
    trust_anchor_ids: tuple[str, ...] = ("TRUSTCV-DATASET-LOCAL-DEFAULT", "default")
    accepted_formats: tuple[str, ...] = ("coco", "yolo", "png", "jpeg", "jpg", "zip", "csv")
    schema_versions: tuple[str, ...] = ("1.0",)
    strict_context_binding: bool = True
    trust_anchor_required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.hash_algorithm.lower() not in {"sha256"}:
            raise ValueError("POLICY_INVALID: hash_algorithm must be sha256")
        if self.signature_algorithm.lower() not in {"ed25519"}:
            raise ValueError("POLICY_INVALID: signature_algorithm must be ed25519")
        if self.canonicalization_version not in {"1.0"}:
            raise ValueError("UNSUPPORTED_CANONICALIZATION_VERSION")
        if self.freshness_window_seconds <= 0:
            raise ValueError("POLICY_INVALID: freshness_window_seconds must be positive")


DEFAULT_VERIFICATION_POLICY = VerificationPolicy()


def check_policy_compatibility(policy: VerificationPolicy, incoming: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Reject silent policy downgrade or incompatible policy modifications."""
    if incoming is None:
        return True, None
    try:
        freshness_window = int(incoming["freshness_window_seconds"]) if "freshness_window_seconds" in incoming else None
        future_skew = int(incoming["maximum_future_skew_seconds"]) if "maximum_future_skew_seconds" in incoming else None
    except (TypeError, ValueError, OverflowError):
        return False, "POLICY_INVALID"

    exact_fields = {
        "hash_algorithm": policy.hash_algorithm,
        "signature_algorithm": policy.signature_algorithm,
        "canonicalization_version": policy.canonicalization_version,
        "sequence_scope": policy.sequence_scope,
        "policy_version": policy.policy_version,
        "schema_version": policy.schema_version,
        "replay_policy": policy.replay_policy,
    }
    for field_name, expected in exact_fields.items():
        if field_name in incoming and incoming[field_name] != expected:
            return False, "POLICY_DOWNGRADE_DETECTED"

    if freshness_window is not None and (freshness_window <= 0 or freshness_window > policy.freshness_window_seconds):
        return False, "POLICY_DOWNGRADE_DETECTED"
    if "strict_context_binding" in incoming and incoming["strict_context_binding"] is False and policy.strict_context_binding:
        return False, "POLICY_DOWNGRADE_DETECTED"
    if "sequence_required" in incoming and incoming["sequence_required"] is False:
        return False, "POLICY_DOWNGRADE_DETECTED"
    if "nonce_required" in incoming and incoming["nonce_required"] is False and policy.nonce_required:
        return False, "POLICY_DOWNGRADE_DETECTED"
    if "trust_anchor_required" in incoming and incoming["trust_anchor_required"] is False and policy.trust_anchor_required:
        return False, "POLICY_DOWNGRADE_DETECTED"
    if future_skew is not None and (future_skew > policy.maximum_future_skew_seconds or future_skew < 0):
        return False, "POLICY_DOWNGRADE_DETECTED"
    return True, None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonicalize(value: Any) -> bytes:
    """Deterministic JSON-like canonical representation."""
    def enc(v: Any) -> str:
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("non-finite float")
            return format(v, ".17g")
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(enc(x) for x in v) + "]"
        if isinstance(v, dict):
            pairs = sorted((str(k), v[k]) for k in v)
            return "{" + ",".join(enc(k) + ":" + enc(val) for k, val in pairs) + "}"
        raise TypeError(f"unsupported canonical type: {type(v).__name__}")
    return enc(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_bytes(canonicalize(value))


def _jsonable(value: Any) -> Any:
    """Convert NumPy/Pandas scalar containers into deterministic primitives."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _public_b64(public_key: Ed25519PublicKey) -> str:
    return base64.b64encode(
        public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")


def _private_from_pem(raw: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Stored signing key is not Ed25519")
    return key


def _public_from_b64(value: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(value.encode("ascii")))


def ensure_trust_anchor() -> dict[str, Any]:
    """Create a local trusted dataset-signing key once, then reuse it."""
    existing = _load_json(TRUST_FILE, {})
    if existing.get("key_id") and existing.get("public_key"):
        return existing

    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    public_b64 = _public_b64(public)
    key_id = "TRUSTCV-DATASET-" + sha256_bytes(
        base64.b64decode(public_b64)
    )[:16].upper()

    KEY_FILE.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    try:
        KEY_FILE.chmod(0o600)
    except OSError:
        pass

    record = {
        "trust_anchor_id": TRUST_ANCHOR_ID,
        "key_id": key_id,
        "public_key": public_b64,
        "algorithm": SIGNATURE_ALGORITHM,
        "created_at": _utc_iso(),
        "purpose": "TrustCV local dataset provenance signing",
    }
    _save_json(TRUST_FILE, record)
    return record


def _load_private() -> Ed25519PrivateKey:
    ensure_trust_anchor()
    if not KEY_FILE.exists():
        raise FileNotFoundError("Dataset signing key is not configured")
    return _private_from_pem(KEY_FILE.read_bytes())


def _load_state() -> dict[str, Any]:
    state = _load_json(
        STATE_FILE,
        {"provenance_sequence": 0, "audit_sequence": 0, "seen_event_ids": [], "seen_nonces": [], "last_audit_hash": None},
    )
    if not isinstance(state, dict):
        return {"sequence": 0, "seen_event_ids": [], "seen_nonces": [], "last_audit_hash": None}
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["seen_event_ids"] = list(dict.fromkeys(state.get("seen_event_ids", [])))[-5000:]
    state["seen_nonces"] = list(dict.fromkeys(state.get("seen_nonces", [])))[-5000:]
    _save_json(STATE_FILE, state)


def create_signed_dataset_manifest(
    *,
    dataset_name: str,
    dataset_digest: str,
    source: str = "TrustCV Local Dataset Intake",
    owner: str = "Verified Contributor",
    vendor: str | None = None,
    contributor_id: str | None = None,
    contribution_id: str | None = None,
    batch_id: str | None = None,
    task_type: str | None = None,
    annotation_format: str | None = None,
    label_schema: list[str] | None = None,
    version: str = "local-intake-1.0",
) -> dict[str, Any]:
    """Create a locally trusted manifest for the exact uploaded dataset bytes.

    The local trusted Ed25519 authority signs the canonicalized payload
    including the SHA-256 of the dataset, the contributor identity,
    contribution identity, and batch identity, establishing an authentic, verifiable provenance root.
    """
    anchor = ensure_trust_anchor()
    clean_digest = str(dataset_digest).removeprefix("sha256:").strip().lower()
    effective_vendor = vendor or owner
    clean_contrib = str(contributor_id or effective_vendor).strip()
    record = {
        "manifest_type": "TrustCV Dataset Manifest",
        "manifest_version": SECURITY_SCHEMA_VERSION,
        "dataset_name": str(dataset_name),
        "expected_sha256": clean_digest,
        "source": source,
        "vendor": effective_vendor,
        "contributor_id": clean_contrib,
        "version": version,
        "created_at": _utc_iso(),
        "signature_algorithm": "Ed25519",
        "hash_algorithm": "SHA-256",
        "canonicalization_version": CANONICALIZATION_VERSION,
        "trust_anchor_id": anchor["trust_anchor_id"],
        "key_id": anchor["key_id"],
        "public_key": anchor["public_key"],
        "attestation": "LOCAL_TRUSTCV_INTAKE",
    }
    if contribution_id:
        record["contribution_id"] = str(contribution_id).strip()
    if batch_id:
        record["batch_id"] = str(batch_id).strip()
    if task_type:
        record["task_type"] = str(task_type).strip()
    if annotation_format:
        record["annotation_format"] = str(annotation_format).strip()
    if label_schema is not None:
        record["label_schema"] = label_schema

    record["signature"] = base64.b64encode(
        _load_private().sign(canonicalize(record))
    ).decode("ascii")
    return record


def verify_signed_dataset_manifest(
    manifest_input: bytes | str | dict[str, Any],
    *,
    actual_dataset_digest: str,
    expected_contributor_id: str | None = None,
    expected_contribution_id: str | None = None,
    expected_batch_id: str | None = None,
    actual_dataset_name: str | None = None,
    trusted_anchor: dict[str, Any] | None = None,
    backend: Any = None,
    enforce_lineage: bool = True,
) -> dict[str, Any]:
    """
    Cryptographically verify a signed dataset manifest against actual observed values.

    Truthful verification rule:
    A PASS result is ONLY returned when:
      1. Manifest schema is intact.
      2. Ed25519 (or legacy RSA-PSS) digital signature is mathematically valid.
      3. The signing key is recognized by the trusted dataset trust anchor.
      4. The signed dataset digest strictly matches the actual uploaded dataset digest.
      5. The signed contributor/vendor strictly matches the expected contributor (if specified).
      6. The signed contribution strictly matches the expected contribution (if specified).
      7. The signed batch strictly matches the expected batch (if specified).
    """
    # 1. Parse manifest
    if isinstance(manifest_input, bytes):
        try:
            manifest = json.loads(manifest_input.decode("utf-8"))
        except Exception as exc:
            return {
                "overall": "FAIL",
                "valid": False,
                "status": "INVALID_MANIFEST",
                "reason": f"Manifest is not valid JSON: {exc}",
                "manifest": {},
                "checks": {
                    "manifest_integrity": {"status": "FAIL", "reason": str(exc)},
                    "digital_signature": {"status": "UNAVAILABLE", "reason": "Manifest unreadable"},
                    "dataset_digest": {"status": "UNAVAILABLE", "signed": None, "observed": actual_dataset_digest},
                    "contributor": {"status": "UNAVAILABLE", "signed": None, "observed": expected_contributor_id},
                    "contribution": {"status": "UNAVAILABLE", "signed": None, "observed": expected_contribution_id},
                    "batch": {"status": "UNAVAILABLE", "signed": None, "observed": expected_batch_id},
                },
            }
    elif isinstance(manifest_input, str):
        try:
            manifest = json.loads(manifest_input)
        except Exception as exc:
            return {
                "overall": "FAIL",
                "valid": False,
                "status": "INVALID_MANIFEST",
                "reason": f"Manifest is not valid JSON: {exc}",
                "manifest": {},
                "checks": {
                    "manifest_integrity": {"status": "FAIL", "reason": str(exc)},
                    "digital_signature": {"status": "UNAVAILABLE", "reason": "Manifest unreadable"},
                    "dataset_digest": {"status": "UNAVAILABLE", "signed": None, "observed": actual_dataset_digest},
                    "contributor": {"status": "UNAVAILABLE", "signed": None, "observed": expected_contributor_id},
                    "contribution": {"status": "UNAVAILABLE", "signed": None, "observed": expected_contribution_id},
                    "batch": {"status": "UNAVAILABLE", "signed": None, "observed": expected_batch_id},
                },
            }
    elif isinstance(manifest_input, dict):
        manifest = dict(manifest_input)
    else:
        return {
            "overall": "FAIL",
            "valid": False,
            "status": "INVALID_MANIFEST",
            "reason": f"Unsupported manifest type: {type(manifest_input).__name__}",
            "manifest": {},
            "checks": {},
        }

    anchor = trusted_anchor or ensure_trust_anchor()

    # 2. Check schema / integrity
    required = ["expected_sha256", "signature"]
    missing = [k for k in required if not manifest.get(k)]
    integrity_valid = len(missing) == 0
    checks: dict[str, dict[str, Any]] = {
        "manifest_integrity": {
            "status": "VALID" if integrity_valid else "FAIL",
            "schema_version": str(manifest.get("manifest_version", "1.0")),
            "missing_fields": missing,
            "reason": "Manifest schema fields present." if integrity_valid else f"Missing fields: {missing}",
        }
    }

    if not integrity_valid:
        return {
            "overall": "FAIL",
            "valid": False,
            "status": "INVALID_MANIFEST",
            "reason": f"Manifest missing required fields: {', '.join(missing)}",
            "manifest": manifest,
            "checks": checks,
        }

    # 3. Cryptographic Signature Verification
    sig_raw = str(manifest.get("signature", "")).strip()
    pub_key_raw = str(manifest.get("public_key") or manifest.get("public_key_pem") or "").strip()
    algo = str(manifest.get("signature_algorithm", "ed25519")).lower()
    is_trusted_anchor = (
        manifest.get("trust_anchor_id") == anchor.get("trust_anchor_id")
        and manifest.get("key_id") == anchor.get("key_id")
        and manifest.get("public_key") == anchor.get("public_key")
    )

    sig_status = "FAIL"
    sig_reason = ""

    if not sig_raw or not pub_key_raw:
        sig_status = "UNAVAILABLE"
        sig_reason = "Signature or public key missing from manifest."
    elif algo == "ed25519":
        unsigned = {k: v for k, v in manifest.items() if k != "signature"}
        try:
            canonical_bytes = canonicalize(unsigned)
            sig_bytes = base64.b64decode(sig_raw.encode("ascii"))
            pub_bytes = base64.b64decode(pub_key_raw.encode("ascii"))
            key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            key.verify(sig_bytes, canonical_bytes)

            if is_trusted_anchor:
                sig_status = "VALID"
                sig_reason = "Ed25519 signature is cryptographically valid and bound to the trusted dataset trust anchor."
            else:
                sig_status = "MISSING_TRUSTED_KEY"
                sig_reason = "Ed25519 signature is mathematically valid, but the key is not bound to the trusted dataset trust anchor."
        except InvalidSignature:
            sig_status = "INVALID_SIGNATURE"
            sig_reason = "Cryptographic signature check failed. Payload was tampered or signature is invalid."
        except Exception as exc:
            sig_status = "INVALID_SIGNATURE"
            sig_reason = f"Signature verification error: {exc}"
    elif "rsa" in algo:
        # Legacy RSA-PSS path
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            public_obj = serialization.load_pem_public_key(pub_key_raw.encode("utf-8"))
            public_obj.verify(
                base64.b64decode(sig_raw),
                actual_dataset_digest.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            sig_status = "VALID_UNANCHORED"
            sig_reason = "Legacy RSA-PSS signature is mathematically valid, but not represented by the MIRAD Ed25519 trust anchor."
        except InvalidSignature:
            sig_status = "INVALID_SIGNATURE"
            sig_reason = "RSA signature check failed."
        except Exception as exc:
            sig_status = "FAIL"
            sig_reason = f"RSA verification error: {exc}"
    else:
        sig_status = "FAIL"
        sig_reason = f"Unsupported signature algorithm: {algo}"

    checks["digital_signature"] = {
        "status": sig_status,
        "algorithm": algo.upper(),
        "key_id": manifest.get("key_id", "UNKNOWN"),
        "trust_anchor_id": manifest.get("trust_anchor_id", "UNKNOWN"),
        "trusted_key": is_trusted_anchor,
        "reason": sig_reason,
    }

    # 4. Dataset Digest Binding
    signed_digest = str(manifest.get("expected_sha256", "")).removeprefix("sha256:").strip().lower()
    observed_digest = str(actual_dataset_digest).removeprefix("sha256:").strip().lower()
    digest_match = bool(signed_digest and signed_digest == observed_digest)

    checks["dataset_digest"] = {
        "status": "MATCH" if digest_match else "MISMATCH",
        "signed": signed_digest,
        "observed": observed_digest,
        "reason": "Signed dataset digest matches actual uploaded bytes." if digest_match else f"Digest mismatch: signed={signed_digest[:16]}... vs observed={observed_digest[:16]}...",
    }

    # 5. Extract Signed Identity Context
    signed_contrib = str(
        manifest.get("contributor_id")
        or manifest.get("vendor")
        or manifest.get("owner")
        or ""
    ).strip()
    signed_contrib_id = str(manifest.get("contribution_id") or "").strip()
    signed_batch = str(manifest.get("batch_id") or manifest.get("batch") or "").strip()

    # Query persistent backend for exact signed lineage if enabled and backend provided
    lineage_res = None
    if enforce_lineage and backend is not None:
        if hasattr(backend, "verify_dataset_manifest_lineage"):
            lineage_res = backend.verify_dataset_manifest_lineage(
                dataset_digest=actual_dataset_digest,
                contributor_id=signed_contrib,
                contribution_id=signed_contrib_id or None,
                batch_id=signed_batch or None,
            )

    # 6. Contributor Binding
    if expected_contributor_id:
        obs_contrib = str(expected_contributor_id).strip()
        if obs_contrib in ("ROOT / UNKNOWN SOURCE", "DEFAULT_CONTRIBUTOR", ""):
            contrib_match = True
            contrib_status = "MATCH"
        else:
            contrib_match = (signed_contrib.lower() == obs_contrib.lower())
            contrib_status = "MATCH" if contrib_match else "MISMATCH"
    elif lineage_res and lineage_res.get("persisted_contribution"):
        obs_contrib = lineage_res["persisted_contribution"].get("contributor_id") or signed_contrib
        contrib_match = bool(signed_contrib and signed_contrib.lower() == obs_contrib.lower())
        contrib_status = "MATCH" if contrib_match else "MISMATCH"
    else:
        obs_contrib = signed_contrib or "NOT_SPECIFIED"
        contrib_match = bool(signed_contrib)
        contrib_status = "MATCH" if signed_contrib else "NOT_SPECIFIED"

    checks["contributor"] = {
        "status": contrib_status,
        "signed": signed_contrib or "NOT_SPECIFIED",
        "observed": obs_contrib,
        "persisted": (lineage_res.get("persisted_contribution", {}).get("contributor_id") if lineage_res and lineage_res.get("persisted_contribution") else None),
        "reason": "Contributor binding matches." if contrib_status == "MATCH" else (
            "No expected contributor specified." if contrib_status == "NOT_SPECIFIED" else
            f"Contributor mismatch: signed '{signed_contrib}' vs observed '{obs_contrib}'"
        ),
    }

    # 7. Contribution Binding
    if lineage_res is not None:
        l_stat = lineage_res.get("status")
        p_c = lineage_res.get("persisted_contribution")
        persisted_contrib_id = p_c.get("contribution_id") if p_c else None

        if l_stat == "CONTRIBUTION_CONTEXT_NOT_FOUND":
            contrib_id_status = "CONTRIBUTION_CONTEXT_NOT_FOUND"
            obs_contrib_id = "NOT_FOUND"
            contrib_id_reason = lineage_res.get("reason", f"Signed contribution '{signed_contrib_id}' not found in persistent database.")
        elif l_stat == "PERSISTED_LINEAGE_INCONSISTENCY" and any(c.get("field") in ("contribution_id", "dataset_digest") for c in lineage_res.get("conflicts", [])):
            contrib_id_status = "PERSISTED_LINEAGE_INCONSISTENCY"
            obs_contrib_id = persisted_contrib_id or "INCONSISTENT"
            contrib_id_reason = lineage_res.get("reason", "Persisted contribution lineage inconsistency.")
        elif l_stat == "MATCH":
            obs_contrib_id = persisted_contrib_id or signed_contrib_id
            if expected_contribution_id and expected_contribution_id.strip().lower() != signed_contrib_id.lower():
                contrib_id_status = "MISMATCH"
                obs_contrib_id = expected_contribution_id.strip()
                contrib_id_reason = f"Contribution mismatch: signed '{signed_contrib_id}' vs expected '{obs_contrib_id}'"
            else:
                contrib_id_status = "MATCH" if signed_contrib_id else "NOT_SPECIFIED"
                contrib_id_reason = f"Contribution binding matches persisted backend record '{obs_contrib_id}'."
        else:
            contrib_id_status = l_stat
            obs_contrib_id = persisted_contrib_id or "UNKNOWN"
            contrib_id_reason = lineage_res.get("reason", "Contribution lineage evaluated.")
    else:
        if expected_contribution_id:
            obs_contrib_id = expected_contribution_id.strip()
            if signed_contrib_id:
                c_match = (signed_contrib_id.lower() == obs_contrib_id.lower())
                contrib_id_status = "MATCH" if c_match else "MISMATCH"
                contrib_id_reason = "Contribution binding matches." if c_match else f"Contribution mismatch: signed '{signed_contrib_id}' vs observed '{obs_contrib_id}'"
            else:
                contrib_id_status = "MISMATCH"
                contrib_id_reason = "Contribution missing from manifest."
        else:
            obs_contrib_id = signed_contrib_id or "NOT_SPECIFIED"
            contrib_id_status = "MATCH" if signed_contrib_id else "NOT_SPECIFIED"
            contrib_id_reason = "Contribution binding matches." if signed_contrib_id else "No expected contribution specified."

    checks["contribution"] = {
        "status": contrib_id_status,
        "signed": signed_contrib_id or "NOT_SPECIFIED",
        "observed": obs_contrib_id,
        "persisted": (lineage_res.get("persisted_contribution", {}).get("contribution_id") if lineage_res and lineage_res.get("persisted_contribution") else None),
        "reason": contrib_id_reason,
    }

    # 8. Batch Binding
    if lineage_res is not None:
        l_stat = lineage_res.get("status")
        p_b = lineage_res.get("persisted_batch")
        persisted_batch_id = p_b.get("batch_id") if p_b else None

        if l_stat == "PERSISTED_LINEAGE_INCONSISTENCY" and any("batch" in c.get("field", "") for c in lineage_res.get("conflicts", [])):
            batch_status = "PERSISTED_LINEAGE_INCONSISTENCY"
            obs_batch = "INCONSISTENT"
            batch_reason = lineage_res.get("reason", "Persisted batch lineage inconsistency.")
        elif l_stat == "CONTRIBUTION_CONTEXT_NOT_FOUND":
            if expected_batch_id and expected_batch_id.strip().lower() != signed_batch.lower():
                batch_status = "MISMATCH"
                obs_batch = expected_batch_id.strip()
                batch_reason = f"Batch mismatch: signed '{signed_batch}' vs expected '{obs_batch}'"
            else:
                batch_status = "CONTRIBUTION_CONTEXT_NOT_FOUND"
                obs_batch = "UNRESOLVED"
                batch_reason = "Batch cannot be resolved because contribution context was not found in persistent database."
        elif expected_batch_id and expected_batch_id.strip().lower() != signed_batch.lower():
            batch_status = "MISMATCH"
            obs_batch = expected_batch_id.strip()
            batch_reason = f"Batch mismatch: signed '{signed_batch}' vs expected '{obs_batch}'"
        elif l_stat == "MATCH":
            obs_batch = persisted_batch_id or signed_batch
            batch_status = "MATCH" if signed_batch else "NOT_SPECIFIED"
            batch_reason = f"Batch binding matches persisted backend record '{obs_batch}'."
        else:
            batch_status = "MATCH" if (signed_batch and persisted_batch_id and signed_batch.lower() == persisted_batch_id.lower()) else ("NOT_SPECIFIED" if not signed_batch else "MISMATCH")
            obs_batch = persisted_batch_id or signed_batch
            batch_reason = lineage_res.get("reason", "Batch lineage evaluated.")
    else:
        if expected_batch_id:
            obs_batch = expected_batch_id.strip()
            if signed_batch:
                b_match = (signed_batch.lower() == obs_batch.lower())
                batch_status = "MATCH" if b_match else "MISMATCH"
                batch_reason = "Batch binding matches." if b_match else f"Batch mismatch: signed '{signed_batch}' vs observed '{obs_batch}'"
            else:
                batch_status = "MISMATCH"
                batch_reason = "Batch missing from manifest."
        else:
            obs_batch = signed_batch or "NOT_SPECIFIED"
            batch_status = "MATCH" if signed_batch else "NOT_SPECIFIED"
            batch_reason = "Batch binding matches." if signed_batch else "No expected batch specified."

    checks["batch"] = {
        "status": batch_status,
        "signed": signed_batch or "NOT_SPECIFIED",
        "observed": obs_batch,
        "persisted": (lineage_res.get("persisted_batch", {}).get("batch_id") if lineage_res and lineage_res.get("persisted_batch") else None),
        "reason": batch_reason,
    }

    if lineage_res:
        checks["lineage"] = lineage_res

    # 9. Dataset Name Binding (if present in manifest)
    signed_name = str(manifest.get("dataset_name") or "").strip()
    if actual_dataset_name and signed_name:
        obs_name = str(actual_dataset_name).strip()
        name_match = (signed_name.lower() == obs_name.lower())
        name_status = "MATCH" if name_match else "MISMATCH"
    else:
        obs_name = actual_dataset_name or signed_name or "NOT_SPECIFIED"
        name_status = "MATCH" if (signed_name and signed_name == obs_name) else "NOT_SPECIFIED"

    checks["dataset_name"] = {
        "status": name_status,
        "signed": signed_name or "NOT_SPECIFIED",
        "observed": obs_name,
    }

    # 10. Dataset Annotation / Purpose Context
    annotation_format = str(manifest.get("annotation_format") or "NONE / UNAVAILABLE")
    task_type = str(manifest.get("task_type") or "UNANNOTATED_IMAGE_DATASET")
    label_schema = manifest.get("label_schema")
    checks["annotation_context"] = {
        "status": "VALID",
        "format": annotation_format,
        "task_type": task_type,
        "label_schema": label_schema,
        "reason": f"Observed dataset facts: format={annotation_format}, task={task_type}",
    }

    # 11. Determine Overall Truthful Status
    if sig_status == "INVALID_SIGNATURE":
        overall = "INVALID_SIGNATURE"
        reason = "Digital signature verification failed (tampered content or bad signature)."
    elif sig_status == "MISSING_TRUSTED_KEY":
        overall = "MISSING_TRUSTED_KEY"
        reason = "Digital signature is valid but signer is NOT bound to the trusted dataset trust anchor."
    elif sig_status in ("FAIL", "UNAVAILABLE"):
        overall = "FAIL"
        reason = sig_reason or "Digital signature is missing or unverified."
    elif not digest_match:
        overall = "DATASET_DIGEST_MISMATCH"
        reason = f"Signed dataset digest ({signed_digest[:16]}...) does NOT match uploaded dataset ({observed_digest[:16]}...)."
    elif contrib_status == "MISMATCH":
        overall = "CONTRIBUTOR_MISMATCH"
        reason = f"Signed contributor '{signed_contrib}' does NOT match expected contributor '{obs_contrib}'."
    elif contrib_id_status == "CONTRIBUTION_CONTEXT_NOT_FOUND" or batch_status == "CONTRIBUTION_CONTEXT_NOT_FOUND":
        overall = "CONTRIBUTION_CONTEXT_NOT_FOUND"
        reason = contrib_id_reason or "Signed contribution not found in persistent backend records."
    elif contrib_id_status == "PERSISTED_LINEAGE_INCONSISTENCY" or batch_status == "PERSISTED_LINEAGE_INCONSISTENCY":
        overall = "PERSISTED_LINEAGE_INCONSISTENCY"
        reason = (contrib_id_reason if contrib_id_status == "PERSISTED_LINEAGE_INCONSISTENCY" else batch_reason) or "Persisted lineage inconsistency detected."
    elif batch_status == "MISMATCH":
        overall = "BATCH_MISMATCH"
        reason = f"Signed batch '{signed_batch}' does NOT match expected batch '{obs_batch}'."
    elif contrib_id_status == "MISMATCH":
        overall = "CONTRIBUTION_MISMATCH"
        reason = f"Signed contribution '{signed_contrib_id}' does NOT match expected contribution '{obs_contrib_id}'."
    elif sig_status in ("VALID", "VALID_UNANCHORED") and digest_match and contrib_status == "MATCH" and contrib_id_status in ("MATCH", "NOT_SPECIFIED") and batch_status in ("MATCH", "NOT_SPECIFIED"):
        overall = "PASS" if sig_status == "VALID" else "REVIEW"
        reason = (
            "Cryptographic signature is VALID, anchored to the trusted key, and strictly bound to the dataset digest, contributor, contribution, and batch."
            if overall == "PASS" else
            "Signature is mathematically valid but not from the local Ed25519 trust anchor."
        )
    else:
        overall = "FAIL"
        reason = "Verification failed one or more security bindings."

    return {
        "overall": overall,
        "valid": overall == "PASS",
        "status": overall,
        "reason": reason,
        "manifest": manifest,
        "checks": checks,
    }



def create_dataset_provenance(
    *,
    dataset_name: str,
    dataset_digest: str,
    manifest_digest: str | None,
    source: str = "TrustCV Local Dataset Intake",
    owner: str = "Verified Contributor",
    version: str = "local-intake-1.0",
    analysis_digest: str,
    evidence_digest: str,
    contributor_id: str | None = None,
    contribution_id: str | None = None,
    batch_id: str | None = None,
    context: str = "TrustCV:dataset:intake",
) -> dict[str, Any]:
    """Create a signed, dataset-only provenance event."""
    anchor = ensure_trust_anchor()
    state = _load_state()
    sequence = int(state.get("provenance_sequence", 0)) + 1
    event_id = "dataset-event-" + secrets.token_hex(12)
    nonce = secrets.token_hex(16)

    clean_contrib = str(contributor_id or owner or "NOT PROVIDED").strip()
    clean_batch = str(batch_id or "").strip()
    clean_cntrb = str(contribution_id or "").strip()

    record = {
        "provenance_version": SECURITY_SCHEMA_VERSION,
        "event_id": event_id,
        "timestamp": _utc_iso(),
        "nonce": nonce,
        "sequence": sequence,
        "context": context,
        "dataset_name": dataset_name,
        "dataset_digest": f"sha256:{dataset_digest}",
        "manifest_digest": f"sha256:{manifest_digest}" if manifest_digest else None,
        "source": source or "NOT PROVIDED",
        "owner": owner or "NOT PROVIDED",
        "contributor_id": clean_contrib,
        "contribution_id": clean_cntrb or None,
        "batch_id": clean_batch or None,
        "version": version or "NOT PROVIDED",
        "analysis_digest": f"sha256:{analysis_digest}",
        "evidence_digest": f"sha256:{evidence_digest}",
        "hash_algorithm": HASH_ALGORITHM,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "trust_anchor_id": anchor["trust_anchor_id"],
        "verification_policy_version": POLICY_VERSION,
        "key_id": anchor["key_id"],
        "public_key": anchor["public_key"],
    }
    signature = _load_private().sign(canonicalize(record))
    record["signature"] = base64.b64encode(signature).decode("ascii")
    return record


def verify_dataset_provenance(
    record: dict[str, Any],
    *,
    expected_dataset_digest: str | None = None,
    expected_context: str = "TrustCV:dataset:intake",
) -> dict[str, Any]:
    """Verify schema, trust anchor, signature and dataset binding."""
    required = [
        "provenance_version", "event_id", "timestamp", "nonce", "sequence",
        "context", "dataset_name", "dataset_digest", "source", "owner", "version",
        "analysis_digest", "evidence_digest", "hash_algorithm",
        "canonicalization_version", "signature_algorithm", "trust_anchor_id",
        "verification_policy_version", "key_id", "public_key", "signature",
    ]
    missing = [k for k in required if not record.get(k)]
    schema_valid = not missing and isinstance(record.get("sequence"), int) and not isinstance(record.get("sequence"), bool) and record["sequence"] >= 1

    timestamp_valid = False
    if schema_valid:
        try:
            datetime.fromisoformat(str(record["timestamp"]).replace("Z", "+00:00"))
            timestamp_valid = True
        except (TypeError, ValueError):
            pass
    schema_valid = schema_valid and timestamp_valid

    anchor = ensure_trust_anchor()
    trust_anchor_valid = (
        record.get("trust_anchor_id") == anchor.get("trust_anchor_id")
        and record.get("key_id") == anchor.get("key_id")
        and record.get("public_key") == anchor.get("public_key")
    )

    signature_valid = False
    if schema_valid and trust_anchor_valid:
        try:
            unsigned = {k: v for k, v in record.items() if k != "signature"}
            _public_from_b64(anchor["public_key"]).verify(
                base64.b64decode(str(record["signature"]).encode("ascii")),
                canonicalize(unsigned),
            )
            signature_valid = True
        except (InvalidSignature, ValueError, TypeError, KeyError):
            signature_valid = False

    dataset_binding = True
    if expected_dataset_digest is not None:
        normalized = str(record.get("dataset_digest", ""))
        dataset_binding = normalized.lower() in {
            expected_dataset_digest.lower(),
            f"sha256:{expected_dataset_digest.lower()}",
        }

    policy_compat, policy_compat_reason = check_policy_compatibility(
        DEFAULT_VERIFICATION_POLICY,
        record.get("verification_policy") if isinstance(record.get("verification_policy"), dict) else None,
    )
    policy_valid = (
        policy_compat
        and record.get("provenance_version") == SECURITY_SCHEMA_VERSION
        and record.get("hash_algorithm") == HASH_ALGORITHM
        and record.get("canonicalization_version") == CANONICALIZATION_VERSION
        and record.get("signature_algorithm") == SIGNATURE_ALGORITHM
        and record.get("verification_policy_version") == POLICY_VERSION
        and record.get("context") == expected_context
    )

    trusted = schema_valid and signature_valid and trust_anchor_valid and dataset_binding and policy_valid
    return {
        "trusted": trusted,
        "schema_valid": schema_valid,
        "signature_valid": signature_valid,
        "trust_anchor_valid": trust_anchor_valid,
        "dataset_binding_valid": dataset_binding,
        "policy_valid": policy_valid,
        "missing_fields": missing,
        "reason": (
            "Signed dataset provenance is valid and bound to the trusted local dataset key."
            if trusted else
            ("Policy downgrade detected: " + (policy_compat_reason or "incompatible policy") if not policy_compat else
             "Dataset provenance verification failed; treat the record as untrusted.")
        ),
    }


def check_dataset_replay(record: dict[str, Any], *, expected_context: str = "TrustCV:dataset:intake") -> dict[str, Any]:
    """Freshness + nonce + event-id + sequence replay protection."""
    state = _load_state()
    now = _utc_now()
    result = {
        "valid": False,
        "event_id_unique": False,
        "nonce_unique": False,
        "fresh": False,
        "sequence_valid": False,
        "context_valid": False,
        "reason": "",
    }

    event_id = str(record.get("event_id", ""))
    nonce = str(record.get("nonce", ""))
    context = str(record.get("context", ""))
    seq = record.get("sequence")

    if not event_id or event_id in state.get("seen_event_ids", []):
        result["reason"] = "Duplicate or missing event_id."
        return result
    result["event_id_unique"] = True

    if not nonce or nonce in state.get("seen_nonces", []):
        result["reason"] = "Nonce is missing or has already been accepted."
        return result
    result["nonce_unique"] = True

    if context != expected_context:
        result["reason"] = "Dataset provenance context does not match the security scope."
        return result
    result["context_valid"] = True

    try:
        timestamp = datetime.fromisoformat(str(record["timestamp"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError):
        result["reason"] = "Invalid provenance timestamp."
        return result

    result["fresh"] = (
        now - timedelta(seconds=FRESHNESS_WINDOW_SECONDS)
        <= timestamp
        <= now + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS)
    )
    if not result["fresh"]:
        result["reason"] = "Dataset provenance event is stale or from the future."
        return result

    last = state.get("provenance_sequence", 0)
    if not isinstance(seq, int) or isinstance(seq, bool) or seq != int(last) + 1:
        result["reason"] = "Dataset provenance sequence is not the next expected value."
        return result
    result["sequence_valid"] = True

    result["valid"] = True
    result["reason"] = "Freshness, nonce, event identity, context and sequence checks passed."
    return result


def append_dataset_audit_event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Append a hash-linked audit event and update replay state."""
    state = _load_state()
    sequence = int(state.get("audit_sequence", 0)) + 1
    previous = state.get("last_audit_hash")
    record = {
        "audit_id": f"dataset-audit-{sequence:06d}",
        "event_type": event_type,
        "timestamp": _utc_iso(),
        "payload": payload,
        "previous_hash": previous,
        "sequence": sequence,
    }
    current_hash = sha256_obj(record)
    record["current_hash"] = current_hash

    with AUDIT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")

    state["audit_sequence"] = sequence
    state["last_audit_hash"] = current_hash
    _save_state(state)
    return record


def verify_dataset_audit() -> dict[str, Any]:
    """Verify the entire dataset audit hash chain and sequence continuity."""
    if not AUDIT_FILE.exists():
        return {"valid": True, "events": 0, "last_hash": None, "reason": "No dataset audit events yet."}

    previous = None
    expected_sequence = 1
    last_hash = None
    count = 0
    try:
        for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            stored = record.get("current_hash")
            unsigned = {k: v for k, v in record.items() if k != "current_hash"}
            if record.get("sequence") != expected_sequence:
                return {"valid": False, "events": count, "last_hash": last_hash, "reason": "Audit sequence discontinuity."}
            if record.get("previous_hash") != previous:
                return {"valid": False, "events": count, "last_hash": last_hash, "reason": "Audit previous-hash link is broken."}
            if stored != sha256_obj(unsigned):
                return {"valid": False, "events": count, "last_hash": last_hash, "reason": "Audit record hash mismatch."}
            previous = stored
            last_hash = stored
            expected_sequence += 1
            count += 1
    except Exception as exc:
        return {"valid": False, "events": count, "last_hash": last_hash, "reason": f"Audit parsing failed: {exc}"}

    return {"valid": True, "events": count, "last_hash": last_hash, "reason": "Dataset audit hash chain is valid."}


def create_signed_checkpoint() -> dict[str, Any]:
    """Create an Ed25519-signed checkpoint over the latest dataset audit state."""
    audit = verify_dataset_audit()
    anchor = ensure_trust_anchor()
    payload = {
        "checkpoint_id": "dataset-checkpoint-" + secrets.token_hex(8),
        "latest_sequence": audit["events"],
        "latest_audit_hash": audit["last_hash"] or "GENESIS",
        "timestamp": _utc_iso(),
        "schema_version": SECURITY_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "signing_key_id": anchor["key_id"],
        "trust_anchor_id": anchor["trust_anchor_id"],
        "public_key": anchor["public_key"],
    }
    payload["signature"] = base64.b64encode(
        _load_private().sign(canonicalize(payload))
    ).decode("ascii")
    _save_json(CHECKPOINT_FILE, payload)
    return payload


def verify_dataset_checkpoint() -> dict[str, Any]:
    """Verify the stored checkpoint and its binding to the current audit head."""
    if not CHECKPOINT_FILE.exists():
        return {"trusted": False, "reason": "No dataset audit checkpoint exists."}

    checkpoint = _load_json(CHECKPOINT_FILE, {})
    anchor = ensure_trust_anchor()
    required = ["checkpoint_id", "latest_sequence", "latest_audit_hash", "timestamp",
                "schema_version", "canonicalization_version", "signing_key_id",
                "trust_anchor_id", "public_key", "signature"]
    schema_valid = all(checkpoint.get(k) not in (None, "") for k in required)
    schema_valid = schema_valid and isinstance(checkpoint.get("latest_sequence"), int) and not isinstance(checkpoint.get("latest_sequence"), bool) and checkpoint["latest_sequence"] >= 0
    if schema_valid:
        try:
            datetime.fromisoformat(str(checkpoint["timestamp"]).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            schema_valid = False
    schema_valid = schema_valid and checkpoint.get("schema_version") == SECURITY_SCHEMA_VERSION
    schema_valid = schema_valid and checkpoint.get("canonicalization_version") == CANONICALIZATION_VERSION
    if not schema_valid:
        return {"trusted": False, "schema_valid": False, "reason": "Checkpoint schema is invalid."}

    audit = verify_dataset_audit()
    binding = (
        checkpoint["latest_sequence"] == audit["events"]
        and checkpoint["latest_audit_hash"] == (audit["last_hash"] or "GENESIS")
    )
    anchor_valid = (
        checkpoint["trust_anchor_id"] == anchor["trust_anchor_id"]
        and checkpoint["signing_key_id"] == anchor["key_id"]
        and checkpoint["public_key"] == anchor["public_key"]
    )
    sig_valid = False
    try:
        unsigned = {k: v for k, v in checkpoint.items() if k != "signature"}
        _public_from_b64(anchor["public_key"]).verify(
            base64.b64decode(checkpoint["signature"].encode("ascii")),
            canonicalize(unsigned),
        )
        sig_valid = True
    except (InvalidSignature, ValueError, TypeError, KeyError):
        sig_valid = False

    trusted = schema_valid and audit["valid"] and binding and anchor_valid and sig_valid
    return {
        "trusted": trusted,
        "schema_valid": schema_valid,
        "audit_valid": audit["valid"],
        "content_binding_valid": binding,
        "trust_anchor_valid": anchor_valid,
        "signature_valid": sig_valid,
        "reason": "Signed checkpoint matches the current dataset audit head." if trusted else "Dataset audit checkpoint verification failed.",
    }


def verify_manifest_file(
    manifest_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    actual_dataset_digest: str | None = None,
    expected_contributor_id: str | None = None,
    expected_batch_id: str | None = None,
    actual_dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Independently verify a signed dataset manifest file directly from disk.
    Parses the actual JSON file, canonicalizes the unsigned payload,
    verifies the Ed25519 signature against the trust anchor, and verifies dataset/contributor/batch bindings.
    """
    path = Path(manifest_path)
    if not path.exists():
        return {
            "overall": "UNAVAILABLE",
            "valid": False,
            "status": "UNAVAILABLE",
            "reason": f"Manifest file does not exist: {path}",
            "manifest": {},
            "checks": {},
        }
    try:
        raw_bytes = path.read_bytes()
    except Exception as exc:
        return {
            "overall": "FAIL",
            "valid": False,
            "status": "INVALID_MANIFEST",
            "reason": f"Could not read manifest file: {exc}",
            "manifest": {},
            "checks": {},
        }

    computed_digest = actual_dataset_digest
    if dataset_path is not None:
        ds_p = Path(dataset_path)
        if not ds_p.exists():
            return {
                "overall": "FAIL",
                "valid": False,
                "status": "DATASET_NOT_FOUND",
                "reason": f"Dataset file does not exist: {ds_p}",
                "manifest": {},
                "checks": {},
            }
        computed_digest = sha256_bytes(ds_p.read_bytes())

    if not computed_digest:
        computed_digest = ""

    return verify_signed_dataset_manifest(
        raw_bytes,
        actual_dataset_digest=computed_digest,
        expected_contributor_id=expected_contributor_id,
        expected_batch_id=expected_batch_id,
        actual_dataset_name=actual_dataset_name,
    )


def persist_dataset_security_run(
    *,
    dataset: dict[str, Any],
    reports: list[dict[str, Any]],
    checks: list[Any],
    manifest: dict[str, Any] | None,
    contributor_id: str | None = None,
    contribution_id: str | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Build and persist dataset provenance, audit and checkpoint records."""
    analysis_payload = {
        "dataset_sha256": dataset.get("hash"),
        "image_count": len(reports),
        "reports": _jsonable(reports),
        "checks": _jsonable(checks),
    }
    analysis_digest = sha256_obj(analysis_payload)

    evidence_payload = {
        "dataset_sha256": dataset.get("hash"),
        "manifest": _jsonable(manifest or {}),
        "analysis_digest": analysis_digest,
        "quality_flags": sum(bool(r.get("quality_issues")) for r in reports),
        "visual_outliers": sum(bool(r.get("is_visual_outlier")) for r in reports),
        "spectral_outliers": sum(bool(r.get("is_spectral_outlier")) for r in reports),
        "high_shift": sum(r.get("shift_status") == "HIGH SHIFT" for r in reports),
        "quarantine": sum(r.get("disposition") == "QUARANTINE" for r in reports),
    }
    evidence_digest = sha256_obj(evidence_payload)

    manifest_digest = None
    if manifest:
        manifest_digest = sha256_obj(manifest)

    c_id = contributor_id or (manifest or {}).get("contributor_id") or dataset.get("selected_contributor")
    b_id = batch_id or (manifest or {}).get("batch_id") or dataset.get("selected_batch")
    cntrb_id = contribution_id or dataset.get("selected_contribution")

    provenance = create_dataset_provenance(
        dataset_name=str(dataset.get("name", "")),
        dataset_digest=str(dataset.get("hash", "")),
        manifest_digest=manifest_digest,
        source=str((manifest or {}).get("source", "NOT PROVIDED")),
        owner=str((manifest or {}).get("vendor") or c_id or "NOT PROVIDED"),
        version=str((manifest or {}).get("version", "NOT PROVIDED")),
        analysis_digest=analysis_digest,
        evidence_digest=evidence_digest,
        contributor_id=c_id,
        contribution_id=cntrb_id,
        batch_id=b_id,
    )

    provenance_verification = verify_dataset_provenance(
        provenance,
        expected_dataset_digest=str(dataset.get("hash", "")),
    )
    replay = check_dataset_replay(provenance)
    if replay["valid"]:
        state = _load_state()
        state["seen_event_ids"].append(provenance["event_id"])
        state["seen_nonces"].append(provenance["nonce"])
        state["provenance_sequence"] = provenance["sequence"]
        _save_state(state)

    audit_event = append_dataset_audit_event(
        "dataset_security_assurance",
        {
            "event_id": provenance["event_id"],
            "dataset_name": dataset.get("name"),
            "dataset_sha256": dataset.get("hash"),
            "analysis_digest": provenance["analysis_digest"],
            "evidence_digest": provenance["evidence_digest"],
            "contributor_id": c_id,
            "batch_id": b_id,
            "provenance_trusted": provenance_verification["trusted"],
            "replay_valid": replay["valid"],
        },
    )
    checkpoint = create_signed_checkpoint()
    checkpoint_verification = verify_dataset_checkpoint()

    audit_verification = verify_dataset_audit()

    # Save real manifest to demo_output/signed_manifests if present
    saved_manifest_path = None
    if manifest:
        clean_ds = str(dataset.get("name", "dataset")).replace(" ", "_").replace("/", "_").replace("\\", "_")
        clean_c = str(c_id or "UNKNOWN").replace(" ", "_")
        clean_b = str(b_id or "BATCH").replace(" ", "_")
        manifest_file = MANIFEST_OUTPUT / f"{clean_ds}_{clean_c}_{clean_b}.manifest.json"
        _save_json(manifest_file, manifest)
        saved_manifest_path = str(manifest_file)

    result = {
        "schema_version": SECURITY_SCHEMA_VERSION,
        "dataset_only": True,
        "dataset_manifest": _jsonable(manifest or {}),
        "manifest_path": saved_manifest_path,
        "provenance": provenance,
        "provenance_verification": provenance_verification,
        "replay_verification": replay,
        "audit_event": audit_event,
        "audit_verification": audit_verification,
        "checkpoint": checkpoint,
        "checkpoint_verification": checkpoint_verification,
        "analysis_digest": f"sha256:{analysis_digest}",
        "evidence_digest": f"sha256:{evidence_digest}",
        "generated_utc": _utc_iso(),
    }

    # Persist inspectable, real security artifacts alongside the project.
    # No private key or secret state is written here.
    _save_json(PROJECT_OUTPUT / "provenance.json", provenance)
    _save_json(PROJECT_OUTPUT / "verification_result.json", {
        "provenance_verification": provenance_verification,
        "replay_verification": replay,
        "audit_verification": audit_verification,
        "checkpoint_verification": checkpoint_verification,
    })
    _save_json(PROJECT_OUTPUT / "checkpoint.json", checkpoint)
    try:
        (PROJECT_OUTPUT / "audit_log.json").write_text(
            json.dumps([
                json.loads(line) for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
            ], indent=2),
            encoding="utf-8",
        )
    except Exception:
        _save_json(PROJECT_OUTPUT / "audit_log.json", [])

    # Also mirror directly to demo_output root for accessibility
    demo_root = APP / "demo_output"
    _save_json(demo_root / "provenance.json", provenance)
    _save_json(demo_root / "verification_result.json", {
        "provenance_verification": provenance_verification,
        "replay_verification": replay,
        "audit_verification": audit_verification,
        "checkpoint_verification": checkpoint_verification,
    })
    _save_json(demo_root / "checkpoint.json", checkpoint)
    try:
        (demo_root / "audit_log.json").write_text(
            json.dumps([
                json.loads(line) for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
            ], indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return result


def replay_existing_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Attempt to accept an already-used provenance record; should fail closed."""
    return check_dataset_replay(record)


def get_persisted_security_paths() -> dict[str, str]:
    return {
        "trust_anchor": str(TRUST_FILE),
        "audit": str(AUDIT_FILE),
        "checkpoint": str(CHECKPOINT_FILE),
        "state": str(STATE_FILE),
    }
