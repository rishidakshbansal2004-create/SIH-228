"""Artifact identity and registration records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ArtifactType(StrEnum):
    MODEL = "MODEL"
    DATASET = "DATASET"


class RegistrationStatus(StrEnum):
    APPROVED = "APPROVED"
    PENDING_REVIEW = "PENDING_REVIEW"
    REVOKED = "REVOKED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ArtifactIdentity:
    """Cryptographic identity for a model or dataset artifact."""

    artifact_type: ArtifactType
    artifact_id: str
    version: str
    artifact_digest: str
    manifest_digest: str | None = None
    format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type.value,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "artifact_digest": self.artifact_digest,
        }
        if self.manifest_digest is not None:
            payload["manifest_digest"] = self.manifest_digest
        if self.format is not None:
            payload["format"] = self.format
        return payload


@dataclass
class ArtifactRegistration:
    """Trusted reference registration for an approved artifact identity."""

    identity: ArtifactIdentity
    status: RegistrationStatus
    registration_timestamp: str
    parent_version: str | None = None
    supersedes: str | None = None
    trust_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "status": self.status.value,
            "registration_timestamp": self.registration_timestamp,
            "parent_version": self.parent_version,
            "supersedes": self.supersedes,
            "trust_metadata": self.trust_metadata,
        }
