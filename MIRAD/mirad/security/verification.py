"""Structured artifact verification results — never bare booleans."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mirad.security.errors import SecurityErrorCode, SecurityStatus
from mirad.security.hashing import hash_file
from mirad.security.identity import ArtifactIdentity, ArtifactRegistration, ArtifactType, RegistrationStatus
from mirad.security.manifest import verify_manifest_files
from mirad.security.trust_anchor import TrustAnchorStore


@dataclass
class VerificationResult:
    """Structured verification output for integration layers."""

    valid: bool
    status: SecurityStatus
    checks: dict[str, bool | None] = field(default_factory=dict)
    failure_codes: list[str] = field(default_factory=list)
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    unavailable_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status.value,
            "checks": self.checks,
            "failure_codes": self.failure_codes,
            "reason": self.reason,
            "evidence": self.evidence,
            "limitations": self.limitations,
            "unavailable_checks": self.unavailable_checks,
        }


def _digest_mismatch_code(artifact_type: ArtifactType) -> SecurityErrorCode:
    if artifact_type == ArtifactType.MODEL:
        return SecurityErrorCode.MODEL_DIGEST_MISMATCH
    return SecurityErrorCode.DATASET_DIGEST_MISMATCH


def _substitution_code(artifact_type: ArtifactType) -> SecurityErrorCode:
    if artifact_type == ArtifactType.MODEL:
        return SecurityErrorCode.MODEL_SUBSTITUTION_DETECTED
    return SecurityErrorCode.DATASET_SUBSTITUTION_DETECTED


def verify_artifact(
    *,
    candidate: ArtifactIdentity,
    trust_store: TrustAnchorStore,
    artifact_path: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> VerificationResult:
    """
    Verify a candidate artifact against trusted references.

    When *artifact_path* is supplied, digests are computed from disk.
    When *manifest* is supplied, multi-file integrity is also checked.
    """
    limitations = [
        "Digest mismatch establishes artifact difference but does not alone establish malicious intent.",
        "Registration as APPROVED means reference-known identity, not guaranteed model/dataset safety.",
    ]
    checks: dict[str, bool | None] = {
        "artifact_identity": None,
        "artifact_digest": None,
        "manifest_integrity": None,
        "trust_anchor": None,
        "registration_status": None,
    }
    failure_codes: list[str] = []
    evidence: dict[str, Any] = {
        "candidate": candidate.to_dict(),
    }

    reference = trust_store.get_artifact_reference(candidate.artifact_id, candidate.version)

    if reference is None:
        checks["trust_anchor"] = False
        checks["artifact_identity"] = False
        failure_codes.append(SecurityErrorCode.TRUST_ANCHOR_FAILURE.value)
        return VerificationResult(
            valid=False,
            status=SecurityStatus.UNKNOWN,
            checks=checks,
            failure_codes=sorted(set(failure_codes)),
            reason="No trusted reference registered for the supplied artifact id and version.",
            evidence=evidence,
            limitations=limitations,
        )

    checks["trust_anchor"] = True
    evidence["reference"] = reference.to_dict()
    checks["registration_status"] = reference.status == RegistrationStatus.APPROVED

    if reference.status == RegistrationStatus.REVOKED:
        failure_codes.append(SecurityErrorCode.TRUST_ANCHOR_FAILURE.value)
        return VerificationResult(
            valid=False,
            status=SecurityStatus.UNAUTHORIZED,
            checks=checks,
            failure_codes=sorted(set(failure_codes)),
            reason="Trusted reference exists but registration status is REVOKED.",
            evidence=evidence,
            limitations=limitations,
        )

    if reference.status != RegistrationStatus.APPROVED:
        failure_codes.append(SecurityErrorCode.TRUST_ANCHOR_FAILURE.value)
        return VerificationResult(
            valid=False,
            status=SecurityStatus.INDETERMINATE,
            checks=checks,
            failure_codes=sorted(set(failure_codes)),
            reason=f"Registration status is {reference.status.value}, not APPROVED.",
            evidence=evidence,
            limitations=limitations,
        )

    ref_identity = reference.identity
    digest_match = ref_identity.artifact_digest == candidate.artifact_digest
    checks["artifact_digest"] = digest_match
    checks["artifact_identity"] = (
        ref_identity.artifact_type == candidate.artifact_type
        and ref_identity.artifact_id == candidate.artifact_id
        and ref_identity.version == candidate.version
        and digest_match
        and ref_identity.manifest_digest == candidate.manifest_digest
    )

    if artifact_path is not None:
        path = Path(artifact_path)
        if path.is_file():
            actual_digest = hash_file(path)
            checks["artifact_digest"] = actual_digest == ref_identity.artifact_digest
            evidence["computed_artifact_digest"] = actual_digest
            if actual_digest != candidate.artifact_digest:
                failure_codes.append(_digest_mismatch_code(candidate.artifact_type).value)
            if actual_digest != ref_identity.artifact_digest:
                failure_codes.append(_substitution_code(candidate.artifact_type).value)
        elif path.is_dir() and manifest is not None:
            manifest_ok, manifest_failures = verify_manifest_files(manifest, path)
            checks["manifest_integrity"] = manifest_ok
            failure_codes.extend(manifest_failures)
            evidence["manifest_verification"] = {"valid": manifest_ok, "failures": manifest_failures}
            if manifest.get("manifest_digest") != ref_identity.manifest_digest:
                failure_codes.append(SecurityErrorCode.MANIFEST_INVALID.value)
        else:
            checks["artifact_digest"] = None
            failure_codes.append(SecurityErrorCode.ARTIFACT_NOT_FOUND.value)

    if not digest_match:
        failure_codes.append(_digest_mismatch_code(candidate.artifact_type).value)
        # Distinguish legitimate registered update (different version) from unauthorized substitution.
        same_id_other_version = trust_store.artifact_store.find_by_digest(candidate.artifact_digest)
        if same_id_other_version:
            status = SecurityStatus.DIFFERENT
            reason = "Candidate digest matches a different registered version; may be a legitimate update."
        else:
            status = SecurityStatus.SUBSTITUTED
            reason = "Candidate artifact digest does not match the trusted reference."
            failure_codes.append(_substitution_code(candidate.artifact_type).value)
        return VerificationResult(
            valid=False,
            status=status,
            checks=checks,
            failure_codes=sorted(set(failure_codes)),
            reason=reason,
            evidence=evidence,
            limitations=limitations,
        )

    if failure_codes:
        return VerificationResult(
            valid=False,
            status=SecurityStatus.MODIFIED,
            checks=checks,
            failure_codes=sorted(set(failure_codes)),
            reason="Artifact verification failed one or more integrity checks.",
            evidence=evidence,
            limitations=limitations,
        )

    return VerificationResult(
        valid=True,
        status=SecurityStatus.VERIFIED,
        checks=checks,
        failure_codes=[],
        reason="Candidate artifact matches the trusted registered reference.",
        evidence=evidence,
        limitations=limitations,
    )
