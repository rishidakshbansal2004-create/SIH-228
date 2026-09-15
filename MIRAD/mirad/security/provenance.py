"""Cryptographic provenance for ML and dataset lineage records."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from mirad.security.canonical import canonicalize
from mirad.security.constants import CANONICALIZATION_VERSION
from mirad.security.errors import SecurityErrorCode
from mirad.security.keys import TrustedKeyRecord, verify_signature_with_key
from mirad.security.trust_anchor import TrustAnchorStore
from mirad.security.hashing import parse_digest


@dataclass(frozen=True)
class ProvenanceVerificationResult:
    signature_valid: bool
    key_known: bool
    key_trusted: bool
    trust_anchor_valid: bool
    policy_valid: bool
    provenance_trusted: bool
    failure_code: str | None = None
    schema_valid: bool = True

    def __bool__(self) -> bool:
        return self.provenance_trusted


@dataclass
class ProvenanceRecord:
    """Signed provenance record, with canonical fingerprint protection."""

    provenance_version: str
    event_id: str
    timestamp: str
    nonce: str
    sequence: int
    context: str
    input_digest: str
    dataset_identity: str
    dataset_digest: str
    model_identity: str
    model_digest: str
    preprocessing_digest: str
    inference_config_digest: str
    output_digest: str
    hash_algorithm: str = "sha256"
    canonicalization_version: str = CANONICALIZATION_VERSION
    signature_algorithm: str = "ed25519"
    trust_anchor_id: str = "default"
    verification_policy_version: str = "1.0"
    key_id: str | None = None
    public_key: str | None = None
    signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provenance_version": self.provenance_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "sequence": self.sequence,
            "context": self.context,
            "input_digest": self.input_digest,
            "dataset_identity": self.dataset_identity,
            "dataset_digest": self.dataset_digest,
            "model_identity": self.model_identity,
            "model_digest": self.model_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "inference_config_digest": self.inference_config_digest,
            "output_digest": self.output_digest,
            "hash_algorithm": self.hash_algorithm,
            "canonicalization_version": self.canonicalization_version,
            "signature_algorithm": self.signature_algorithm,
            "trust_anchor_id": self.trust_anchor_id,
            "verification_policy_version": self.verification_policy_version,
        }
        if self.key_id is not None:
            payload["key_id"] = self.key_id
        if self.public_key is not None:
            payload["public_key"] = self.public_key
        if self.signature is not None:
            payload["signature"] = self.signature
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_provenance(
    *,
    event_id: str,
    nonce: str,
    sequence: int,
    context: str,
    input_digest: str,
    dataset_identity: str,
    dataset_digest: str,
    model_identity: str,
    model_digest: str,
    preprocessing_digest: str,
    inference_config_digest: str,
    output_digest: str,
    provenance_version: str = "1.0",
    trust_anchor_id: str = "default",
    verification_policy_version: str = "1.0",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an unsigned provenance record."""
    return {
        "provenance_version": provenance_version,
        "event_id": event_id,
        "timestamp": _utc_now_iso(),
        "nonce": nonce,
        "sequence": sequence,
        "context": context,
        "input_digest": input_digest,
        "dataset_identity": dataset_identity,
        "dataset_digest": dataset_digest,
        "model_identity": model_identity,
        "model_digest": model_digest,
        "preprocessing_digest": preprocessing_digest,
        "inference_config_digest": inference_config_digest,
        "output_digest": output_digest,
        "hash_algorithm": "sha256",
        "canonicalization_version": CANONICALIZATION_VERSION,
        "signature_algorithm": "ed25519",
        "trust_anchor_id": trust_anchor_id,
        "verification_policy_version": verification_policy_version,
        "metadata": metadata or {},
    }


def sign_provenance(
    provenance: dict[str, Any],
    *,
    private_key: Ed25519PrivateKey,
    key_id: str,
    public_key: str,
) -> dict[str, Any]:
    """Sign a provenance record with an Ed25519 private key."""
    signed = dict(provenance)
    signed["key_id"] = key_id
    signed["public_key"] = public_key
    canonical_bytes = canonicalize({k: v for k, v in signed.items() if k != "signature"})
    signed["signature"] = b64encode(private_key.sign(canonical_bytes)).decode("ascii")
    return signed


def verify_provenance(
    provenance: dict[str, Any],
    *,
    trust_store: TrustAnchorStore | None = None,
    public_key: str | Ed25519PublicKey | None = None,
) -> ProvenanceVerificationResult:
    """Verify provenance, keeping raw signature validity separate from trust."""
    required_strings = (
        "provenance_version", "event_id", "timestamp", "nonce", "context",
        "input_digest", "dataset_identity", "dataset_digest", "model_identity",
        "model_digest", "preprocessing_digest", "inference_config_digest",
        "output_digest", "hash_algorithm", "canonicalization_version",
        "signature_algorithm", "trust_anchor_id", "verification_policy_version",
        "key_id", "public_key", "signature",
    )
    schema_valid = all(isinstance(provenance.get(name), str) and provenance[name] for name in required_strings)
    schema_valid = schema_valid and isinstance(provenance.get("sequence"), int) and not isinstance(provenance.get("sequence"), bool) and provenance["sequence"] >= 1
    for digest_name in ("input_digest", "dataset_digest", "model_digest", "preprocessing_digest", "inference_config_digest", "output_digest"):
        try:
            parse_digest(provenance.get(digest_name, ""))
        except (TypeError, ValueError):
            schema_valid = False
    timestamp = provenance.get("timestamp")
    if schema_valid:
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            schema_valid = False
    if not schema_valid:
        return ProvenanceVerificationResult(False, False, False, False, False, False, SecurityErrorCode.INVALID_SCHEMA.value, False)
    if not provenance.get("signature"):
        return ProvenanceVerificationResult(False, False, False, False, False, False, SecurityErrorCode.INVALID_SIGNATURE.value, True)
    key_id = provenance.get("key_id")
    record = trust_store.get_trusted_key(key_id) if trust_store is not None and key_id else None
    key_known = record is not None
    key_trusted = bool(record and record.status.value == "ACTIVE")
    trust_anchor_valid = bool(record and provenance.get("trust_anchor_id", "default") == record.trust_anchor_id)
    policy_valid = provenance.get("signature_algorithm", "ed25519").lower() == "ed25519"
    policy_valid = policy_valid and provenance.get("provenance_version", "1.0") == "1.0"
    policy_valid = policy_valid and provenance.get("canonicalization_version", CANONICALIZATION_VERSION) == CANONICALIZATION_VERSION
    policy_valid = policy_valid and provenance.get("verification_policy_version", "1.0") == "1.0"
    failure_code = None if policy_valid else SecurityErrorCode.POLICY_INVALID.value
    candidate_public_key = public_key
    if trust_store is not None:
        candidate_public_key = record.public_key if record is not None else None
    if candidate_public_key is None:
        return ProvenanceVerificationResult(False, key_known, key_trusted, trust_anchor_valid, policy_valid, False, failure_code, True)
    payload = {k: v for k, v in provenance.items() if k != "signature"}
    signature_valid = verify_signature_with_key(payload=canonicalize(payload), signature=provenance["signature"], public_key=candidate_public_key)
    trusted = signature_valid and trust_store is not None and key_known and key_trusted and trust_anchor_valid and policy_valid
    if not signature_valid and failure_code is None:
        failure_code = SecurityErrorCode.INVALID_SIGNATURE.value
    return ProvenanceVerificationResult(signature_valid, key_known, key_trusted, trust_anchor_valid, policy_valid, trusted, failure_code, True)


def provenance_identity(provenance: dict[str, Any]) -> str:
    return provenance.get("event_id") or provenance.get("nonce") or "unknown"
