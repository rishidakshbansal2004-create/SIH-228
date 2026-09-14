"""In-memory storage implementations for development and testing."""

from __future__ import annotations

from mirad.security.identity import ArtifactIdentity, ArtifactRegistration
from mirad.security.stores.base import TrustedArtifactStore


class InMemoryTrustedArtifactStore(TrustedArtifactStore):
    """Process-local trusted artifact store."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], ArtifactRegistration] = {}

    def put(self, registration: ArtifactRegistration) -> None:
        key = (registration.identity.artifact_id, registration.identity.version)
        self._by_key[key] = registration

    def get(self, artifact_id: str, version: str) -> ArtifactRegistration | None:
        return self._by_key.get((artifact_id, version))

    def list_versions(self, artifact_id: str) -> list[str]:
        return sorted(version for aid, version in self._by_key if aid == artifact_id)

    def find_by_digest(self, artifact_digest: str) -> list[ArtifactRegistration]:
        return [
            registration
            for registration in self._by_key.values()
            if registration.identity.artifact_digest == artifact_digest
        ]
