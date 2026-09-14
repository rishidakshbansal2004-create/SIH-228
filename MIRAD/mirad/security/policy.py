"""Verification policy model and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mirad.security.errors import SecurityErrorCode


@dataclass(frozen=True)
class VerificationPolicy:
    """Immutable policy for canonicalization, signature, freshness, and trust controls."""

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
    trust_anchor_ids: tuple[str, ...] = ("default",)
    accepted_formats: tuple[str, ...] = ("onnx", "pytorch", "torchscript", "coco", "yolo")
    schema_versions: tuple[str, ...] = ("1.0",)
    strict_context_binding: bool = True
    trust_anchor_required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.hash_algorithm.lower() not in {"sha256"}:
            raise ValueError(SecurityErrorCode.POLICY_INVALID.value)
        if self.signature_algorithm.lower() not in {"ed25519"}:
            raise ValueError(SecurityErrorCode.POLICY_INVALID.value)
        if self.canonicalization_version not in {"1.0"}:
            raise ValueError(SecurityErrorCode.UNSUPPORTED_CANONICALIZATION_VERSION.value)
        if self.freshness_window_seconds <= 0:
            raise ValueError(SecurityErrorCode.POLICY_INVALID.value)
        if self.sequence_scope not in {"source", "session", "pipeline", "stream", "security context", "security_context"}:
            raise ValueError(SecurityErrorCode.POLICY_INVALID.value)


def check_policy_compatibility(policy: VerificationPolicy, incoming: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Reject silent policy downgrade or incompatible policy changes."""
    if incoming is None:
        return True, None
    try:
        freshness_window = int(incoming["freshness_window_seconds"]) if "freshness_window_seconds" in incoming else None
        future_skew = int(incoming["maximum_future_skew_seconds"]) if "maximum_future_skew_seconds" in incoming else None
    except (TypeError, ValueError, OverflowError):
        return False, SecurityErrorCode.POLICY_INVALID.value
    exact_fields = {
        "hash_algorithm": policy.hash_algorithm,
        "signature_algorithm": policy.signature_algorithm,
        "canonicalization_version": policy.canonicalization_version,
        "sequence_scope": policy.sequence_scope,
        "policy_version": policy.policy_version,
        "schema_version": policy.schema_version,
        "replay_policy": policy.replay_policy,
        "accepted_formats": policy.accepted_formats,
        "trust_anchor_ids": policy.trust_anchor_ids,
    }
    for field_name, expected in exact_fields.items():
        if field_name in incoming and incoming[field_name] != expected:
            return False, SecurityErrorCode.POLICY_DOWNGRADE_DETECTED.value
    if freshness_window is not None and (freshness_window <= 0 or freshness_window > policy.freshness_window_seconds):
        return False, SecurityErrorCode.POLICY_DOWNGRADE_DETECTED.value
    if "strict_context_binding" in incoming and incoming["strict_context_binding"] is False and policy.strict_context_binding:
        return False, SecurityErrorCode.POLICY_DOWNGRADE_DETECTED.value
    if "sequence_required" in incoming and incoming["sequence_required"] is False:
        return False, SecurityErrorCode.POLICY_DOWNGRADE_DETECTED.value
    if "nonce_required" in incoming and incoming["nonce_required"] is False and policy.nonce_required:
        return False, SecurityErrorCode.POLICY_DOWNGRADE_DETECTED.value
    if "trust_anchor_required" in incoming and incoming["trust_anchor_required"] is False and policy.trust_anchor_required:
        return False, SecurityErrorCode.POLICY_DOWNGRADE_DETECTED.value
    if future_skew is not None and (future_skew > policy.maximum_future_skew_seconds or future_skew < 0):
        return False, SecurityErrorCode.POLICY_DOWNGRADE_DETECTED.value
    return True, None
