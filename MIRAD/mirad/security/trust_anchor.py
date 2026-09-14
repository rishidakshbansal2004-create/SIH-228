"""Trust anchor store for approved artifact references and trusted verification keys."""

from __future__ import annotations

from typing import Any

from mirad.security.identity import ArtifactIdentity, ArtifactRegistration, RegistrationStatus
from mirad.security.stores.base import TrustAnchorStoreBase
from mirad.security.stores.memory import InMemoryTrustedArtifactStore


class TrustAnchorStore(TrustAnchorStoreBase):
    """
    Trusted roots for artifact references and verification keys.

    Incoming digests or metadata never become trusted without explicit registration.
    """

    def __init__(self, artifact_store: InMemoryTrustedArtifactStore | None = None) -> None:
        self._artifacts = artifact_store or InMemoryTrustedArtifactStore()
        self._trusted_keys: dict[str, Any] = {}

    @property
    def artifact_store(self) -> InMemoryTrustedArtifactStore:
        return self._artifacts

    def register_artifact_reference(self, registration: ArtifactRegistration) -> None:
        self._artifacts.put(registration)

    def get_artifact_reference(self, artifact_id: str, version: str) -> ArtifactRegistration | None:
        return self._artifacts.get(artifact_id, version)

    def register_trusted_key(self, key_record: Any) -> None:
        key_id = str(key_record.key_id)
        existing = self._trusted_keys.get(key_id)
        if existing is not None and existing.public_key != key_record.public_key:
            raise ValueError(f"Trusted key substitution rejected for key_id={key_id}")
        self._trusted_keys[key_id] = key_record

    def get_trusted_key(self, key_id: str | None) -> Any | None:
        if key_id is None:
            return None
        return self._trusted_keys.get(str(key_id))

    def list_trusted_keys(self) -> list[Any]:
        return list(self._trusted_keys.values())

    def is_trusted_identity(self, identity: ArtifactIdentity) -> bool:
        reference = self._artifacts.get(identity.artifact_id, identity.version)
        if reference is None:
            return False
        if reference.status != RegistrationStatus.APPROVED:
            return False
        ref = reference.identity
        return (
            ref.artifact_type == identity.artifact_type
            and ref.artifact_id == identity.artifact_id
            and ref.version == identity.version
            and ref.artifact_digest == identity.artifact_digest
            and ref.manifest_digest == identity.manifest_digest
        )
