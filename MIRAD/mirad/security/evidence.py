"""Evidence envelopes for other MIRAD components and external evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hmac import compare_digest
from typing import Any

from mirad.security.canonical import canonicalize
from mirad.security.hashing import hash_bytes


@dataclass
class EvidenceEnvelope:
    """Structured evidence record consumed by security findings and workflows.

    The envelope is evidence-bound to a specific asset identity and producer, but is
    never treated as trusted merely because another component produced it.
    """

    evidence_id: str
    evidence_type: str
    producer: str
    producer_version: str
    method: str
    affected_asset: str
    asset_type: str
    asset_version: str
    asset_digest: str
    evidence_payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: _utc_now_iso())
    confidence: float = 0.0
    severity: str | None = None
    context: str | None = None
    schema_version: str = "1.0"
    integrity_digest: str | None = None
    signature: str | None = None
    provenance_ref: str | None = None
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "method": self.method,
            "affected_asset": self.affected_asset,
            "asset_type": self.asset_type,
            "asset_version": self.asset_version,
            "asset_digest": self.asset_digest,
            "evidence_payload": self.evidence_payload,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "schema_version": self.schema_version,
            "limitations": self.limitations,
        }
        if self.severity is not None:
            payload["severity"] = self.severity
        if self.context is not None:
            payload["context"] = self.context
        if self.integrity_digest is not None:
            payload["integrity_digest"] = self.integrity_digest
        if self.signature is not None:
            payload["signature"] = self.signature
        if self.provenance_ref is not None:
            payload["provenance_ref"] = self.provenance_ref
        return payload

    def validate(self) -> tuple[bool, str | None]:
        required = [
            self.evidence_id,
            self.evidence_type,
            self.producer,
            self.producer_version,
            self.method,
            self.affected_asset,
            self.asset_type,
            self.asset_version,
            self.asset_digest,
            self.schema_version,
        ]
        if any(v in (None, "") for v in required):
            return False, "Missing required evidence fields"
        if not isinstance(self.evidence_payload, dict):
            return False, "Evidence payload must be a dict"
        supplied_digest = self.integrity_digest
        payload = {k: v for k, v in self.to_dict().items() if k != "integrity_digest"}
        recalculated = hash_bytes(canonicalize(payload))
        if supplied_digest is None:
            self.integrity_digest = recalculated
        elif not compare_digest(supplied_digest, recalculated):
            return False, "Evidence integrity digest mismatch"
        return True, None

    def trust_state(
        self,
        *,
        producer_known: bool = False,
        producer_trusted: bool = False,
        signature_valid: bool = False,
    ) -> dict[str, bool]:
        """Expose structural, integrity, signature, and producer trust separately."""
        valid, _ = self.validate()
        return {
            "evidence_structurally_valid": valid,
            "evidence_integrity_valid": valid,
            "evidence_signature_valid": signature_valid,
            "producer_known": producer_known,
            "producer_trusted": producer_trusted,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def submit_external_evidence(
    *,
    evidence_id: str,
    evidence_type: str,
    producer: str,
    producer_version: str,
    method: str,
    affected_asset: str,
    asset_type: str,
    asset_version: str,
    asset_digest: str,
    evidence_payload: dict[str, Any],
    timestamp: str | None = None,
    confidence: float = 0.0,
    severity: str | None = None,
    context: str | None = None,
    schema_version: str = "1.0",
    signature: str | None = None,
    provenance_ref: str | None = None,
    limitations: list[str] | None = None,
) -> EvidenceEnvelope:
    """Create an evidence envelope and validate the minimum schema.

    The envelope is structured and integrity-bound, but evidence remains separate
    from trusted authenticity decisions.
    """
    envelope = EvidenceEnvelope(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        producer=producer,
        producer_version=producer_version,
        method=method,
        affected_asset=affected_asset,
        asset_type=asset_type,
        asset_version=asset_version,
        asset_digest=asset_digest,
        evidence_payload=evidence_payload,
        timestamp=timestamp or _utc_now_iso(),
        confidence=confidence,
        severity=severity,
        context=context,
        schema_version=schema_version,
        signature=signature,
        provenance_ref=provenance_ref,
        limitations=list(limitations or []),
    )
    ok, reason = envelope.validate()
    if not ok:
        raise ValueError(f"Invalid evidence envelope: {reason}")
    return envelope
