"""Evidence-based security findings and recommended disposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class FindingSeverity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendedDisposition(StrEnum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    QUARANTINE = "QUARANTINE"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNKNOWN = "UNKNOWN"


@dataclass
class SecurityFinding:
    """Structured security finding bound to an asset, provenance, and audit record."""

    finding_id: str
    finding_type: str
    affected_asset: str
    affected_asset_type: str
    asset_version: str
    asset_digest: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    severity: FindingSeverity = FindingSeverity.MEDIUM
    recommended_disposition: RecommendedDisposition = RecommendedDisposition.REVIEW
    limitations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: _utc_now_iso())
    provenance_id: str | None = None
    audit_event_id: str | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "affected_asset": self.affected_asset,
            "affected_asset_type": self.affected_asset_type,
            "asset_version": self.asset_version,
            "asset_digest": self.asset_digest,
            "reason": self.reason,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "recommended_disposition": self.recommended_disposition.value,
            "limitations": self.limitations,
            "timestamp": self.timestamp,
            "provenance_id": self.provenance_id,
            "audit_event_id": self.audit_event_id,
            "schema_version": self.schema_version,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_finding(
    *,
    finding_id: str,
    finding_type: str,
    affected_asset: str,
    affected_asset_type: str,
    asset_version: str,
    asset_digest: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    confidence: float = 0.0,
    severity: FindingSeverity = FindingSeverity.MEDIUM,
    recommended_disposition: RecommendedDisposition = RecommendedDisposition.REVIEW,
    limitations: list[str] | None = None,
    provenance_id: str | None = None,
    audit_event_id: str | None = None,
) -> SecurityFinding:
    return SecurityFinding(
        finding_id=finding_id,
        finding_type=finding_type,
        affected_asset=affected_asset,
        affected_asset_type=affected_asset_type,
        asset_version=asset_version,
        asset_digest=asset_digest,
        reason=reason,
        evidence=evidence or {},
        confidence=confidence,
        severity=severity,
        recommended_disposition=recommended_disposition,
        limitations=limitations or [],
        provenance_id=provenance_id,
        audit_event_id=audit_event_id,
    )


def secure_finding(finding: SecurityFinding) -> SecurityFinding:
    """Return a finding unchanged but ensures the response is explicitly evidence-based."""
    if not finding.reason:
        finding.reason = "Evidence-based finding; no implied malicious intent."
    if not finding.limitations:
        finding.limitations = ["Finding is evidence-based and does not infer malicious intent."]
    return finding
