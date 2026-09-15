"""Signed checkpoints bound to the latest audit chain state."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from mirad.security.canonical import canonicalize
from mirad.security.keys import TrustedKeyRecord, verify_signature_with_key
from mirad.security.trust_anchor import TrustAnchorStore


@dataclass(frozen=True)
class CheckpointVerificationResult:
    signature_valid: bool
    key_known: bool
    key_trusted: bool
    trust_anchor_valid: bool
    content_valid: bool
    checkpoint_trusted: bool
    schema_valid: bool = True
    failure_code: str | None = None

    def __bool__(self) -> bool:
        return self.checkpoint_trusted


@dataclass
class CheckpointRecord:
    """Checkpoint summary for the audit history."""

    checkpoint_id: str
    latest_sequence: int
    latest_audit_hash: str
    timestamp: str
    schema_version: str
    canonicalization_version: str
    signing_key_id: str
    signature: str
    trust_anchor_id: str = "default"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "latest_sequence": self.latest_sequence,
            "latest_audit_hash": self.latest_audit_hash,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "signing_key_id": self.signing_key_id,
            "signature": self.signature,
            "trust_anchor_id": self.trust_anchor_id,
            "metadata": self.metadata or {},
        }


class CheckpointStore:
    """Process-local store of checkpoint records."""

    def __init__(self) -> None:
        self._items: list[CheckpointRecord] = []

    def append(self, checkpoint: CheckpointRecord) -> None:
        self._items.append(checkpoint)

    def list(self) -> list[CheckpointRecord]:
        return list(self._items)

    def last(self) -> CheckpointRecord | None:
        return self._items[-1] if self._items else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_checkpoint(
    *,
    checkpoint_id: str,
    latest_sequence: int,
    latest_audit_hash: str,
    signing_key_id: str,
    private_key: Ed25519PrivateKey,
    public_key: str,
    schema_version: str = "1.0",
    canonicalization_version: str = "1.0",
    trust_anchor_id: str = "default",
) -> dict[str, Any]:
    """Create a signed checkpoint for the latest audit record state."""
    payload = {
        "checkpoint_id": checkpoint_id,
        "latest_sequence": latest_sequence,
        "latest_audit_hash": latest_audit_hash,
        "timestamp": _utc_now_iso(),
        "schema_version": schema_version,
        "canonicalization_version": canonicalization_version,
        "signing_key_id": signing_key_id,
        "trust_anchor_id": trust_anchor_id,
        "public_key": public_key,
    }
    signature = b64encode(private_key.sign(canonicalize(payload))).decode("ascii")
    payload["signature"] = signature
    return payload


def verify_checkpoint(
    checkpoint: dict[str, Any],
    *,
    trust_store: TrustAnchorStore | None = None,
    public_key: str | Ed25519PublicKey | None = None,
    expected_latest_audit_hash: str | None = None,
    expected_latest_sequence: int | None = None,
) -> CheckpointVerificationResult:
    """Verify a signed checkpoint, trust-anchor status, and content binding."""
    required_strings = ("checkpoint_id", "latest_audit_hash", "timestamp", "schema_version", "canonicalization_version", "signing_key_id", "signature", "trust_anchor_id")
    schema_valid = all(isinstance(checkpoint.get(name), str) and checkpoint[name] for name in required_strings)
    schema_valid = schema_valid and isinstance(checkpoint.get("latest_sequence"), int) and not isinstance(checkpoint.get("latest_sequence"), bool) and checkpoint["latest_sequence"] >= 1
    if schema_valid:
        try:
            datetime.fromisoformat(checkpoint["timestamp"].replace("Z", "+00:00"))
        except (TypeError, ValueError):
            schema_valid = False
    if not schema_valid:
        return CheckpointVerificationResult(False, False, False, False, False, False, False, "CHECKPOINT_INVALID")
    if not checkpoint.get("signature"):
        return CheckpointVerificationResult(False, False, False, False, False, False, True, "CHECKPOINT_INVALID")
    content_valid = True
    if expected_latest_audit_hash is not None and checkpoint.get("latest_audit_hash") != expected_latest_audit_hash:
        content_valid = False
    if expected_latest_sequence is not None and checkpoint.get("latest_sequence") != expected_latest_sequence:
        content_valid = False
    key_id = checkpoint.get("signing_key_id")
    trusted = trust_store.get_trusted_key(key_id) if trust_store is not None and key_id else None
    key_known = trusted is not None
    key_trusted = bool(trusted and trusted.status.value == "ACTIVE")
    trust_anchor_valid = bool(trusted and checkpoint.get("trust_anchor_id", "default") == trusted.trust_anchor_id)
    candidate_key = trusted.public_key if trust_store is not None and trusted is not None else public_key
    if candidate_key is None:
        return CheckpointVerificationResult(False, key_known, key_trusted, trust_anchor_valid, content_valid, False, True, "CHECKPOINT_UNTRUSTED")
    payload = {k: v for k, v in checkpoint.items() if k != "signature"}
    signature_valid = verify_signature_with_key(payload=canonicalize(payload), signature=checkpoint["signature"], public_key=candidate_key)
    trusted_result = signature_valid and content_valid and trust_store is not None and key_known and key_trusted and trust_anchor_valid
    return CheckpointVerificationResult(signature_valid, key_known, key_trusted, trust_anchor_valid, content_valid, trusted_result, True, None if trusted_result else "CHECKPOINT_INVALID")
