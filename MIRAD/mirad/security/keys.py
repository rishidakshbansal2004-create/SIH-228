"""Ed25519 key management and trusted verification-key lifecycle."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from mirad.security.canonical import canonicalize
from mirad.security.trust_anchor import TrustAnchorStore


class KeyStatus(StrEnum):
    """Lifecycle states for trusted verification keys."""

    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    REVOKED = "REVOKED"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrustedKeyRecord:
    """A trusted public verification key bound to a trust anchor identity."""

    key_id: str
    public_key: str
    status: KeyStatus = KeyStatus.ACTIVE
    trust_anchor_id: str = "default"
    created_at: str = field(default_factory=lambda: _utc_now_iso())
    revoked_at: str | None = None
    retired_at: str | None = None
    development_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key_id": self.key_id,
            "public_key": self.public_key,
            "status": self.status.value,
            "trust_anchor_id": self.trust_anchor_id,
            "created_at": self.created_at,
            "development_only": self.development_only,
        }
        if self.revoked_at is not None:
            payload["revoked_at"] = self.revoked_at
        if self.retired_at is not None:
            payload["retired_at"] = self.retired_at
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _public_key_to_bytes(public_key: str | Ed25519PublicKey) -> bytes:
    if isinstance(public_key, Ed25519PublicKey):
        return public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    if isinstance(public_key, str):
        return b64decode(public_key.encode("ascii"))
    raise TypeError("public_key must be a base64 string or Ed25519 public key object")


def _public_key_to_base64(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return b64encode(raw).decode("ascii")


def generate_ed25519_keypair(*, key_id: str | None = None, development_only: bool = False) -> tuple[Ed25519PrivateKey, str, str]:
    """Generate an Ed25519 keypair and return (private_key, public_key_b64, key_id)."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    generated_key_id = key_id or f"key-{_utc_now_iso().replace(':', '').replace('-', '').replace('T', '')}"
    if development_only:
        generated_key_id = f"DEVELOPMENT_ONLY::{generated_key_id}"
    return private_key, _public_key_to_base64(public_key), generated_key_id


def register_trusted_key(trust_store: TrustAnchorStore, record: TrustedKeyRecord) -> TrustedKeyRecord:
    """Register a trusted verification key with the existing trust-anchor store."""
    trust_store.register_trusted_key(record)
    return record


def verify_signature_with_key(*, payload: bytes, signature: str, public_key: str | Ed25519PublicKey) -> bool:
    """Verify an Ed25519 signature over canonicalized payload bytes."""
    try:
        key = public_key if isinstance(public_key, Ed25519PublicKey) else Ed25519PublicKey.from_public_bytes(b64decode(public_key.encode("ascii")))
        key.verify(b64decode(signature.encode("ascii")), payload)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def key_digest(public_key: str | Ed25519PublicKey) -> str:
    return b64encode(_public_key_to_bytes(public_key)).decode("ascii")


def key_status_for_record(record: TrustedKeyRecord) -> KeyStatus:
    if record.status == KeyStatus.REVOKED:
        return KeyStatus.REVOKED
    if record.status == KeyStatus.RETIRED:
        return KeyStatus.RETIRED
    if record.status == KeyStatus.ACTIVE:
        return KeyStatus.ACTIVE
    return KeyStatus.UNKNOWN


def canonical_signature_payload(payload: Any) -> bytes:
    return canonicalize(payload)
