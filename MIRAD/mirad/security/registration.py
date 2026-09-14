"""Explicit artifact registration against trust anchors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mirad.security.hashing import hash_canonical_object, hash_file
from mirad.security.identity import (
    ArtifactIdentity,
    ArtifactRegistration,
    ArtifactType,
    RegistrationStatus,
)
from mirad.security.manifest import create_manifest
from mirad.security.trust_anchor import TrustAnchorStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def register_artifact(
    *,
    trust_store: TrustAnchorStore,
    artifact_type: ArtifactType,
    artifact_id: str,
    version: str,
    format: str,
    artifact_path: str | Path,
    status: RegistrationStatus = RegistrationStatus.APPROVED,
    parent_version: str | None = None,
    supersedes: str | None = None,
    trust_metadata: dict[str, Any] | None = None,
    registration_timestamp: str | None = None,
) -> ArtifactRegistration:
    """
    Register an approved artifact reference.

    Registration means reference-known identity, not guaranteed safety.
    """
    path = Path(artifact_path)
    manifest: dict[str, Any] | None = None
    artifact_digest: str

    if path.is_dir():
        manifest = create_manifest(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            version=version,
            format=format,
            root_path=path,
        )
        artifact_digest = hash_canonical_object(
            {
                "manifest_digest": manifest["manifest_digest"],
                "artifact_id": artifact_id,
                "version": version,
            }
        )
        manifest_digest = manifest["manifest_digest"]
    elif path.is_file():
        artifact_digest = hash_file(path)
        manifest_digest = None
    else:
        raise FileNotFoundError(f"Artifact path not found: {path}")

    identity = ArtifactIdentity(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        version=version,
        artifact_digest=artifact_digest,
        manifest_digest=manifest_digest,
        format=format,
    )

    registration = ArtifactRegistration(
        identity=identity,
        status=status,
        registration_timestamp=registration_timestamp or _utc_now_iso(),
        parent_version=parent_version,
        supersedes=supersedes,
        trust_metadata=trust_metadata or {},
    )
    trust_store.register_artifact_reference(registration)
    return registration
