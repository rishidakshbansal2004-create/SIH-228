"""Abstract storage interfaces decoupled from cryptographic logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mirad.security.identity import ArtifactIdentity, ArtifactRegistration


class TrustedArtifactStore(ABC):
    """Store for approved artifact reference registrations."""

    @abstractmethod
    def put(self, registration: ArtifactRegistration) -> None:
        """Persist a trusted artifact registration."""

    @abstractmethod
    def get(self, artifact_id: str, version: str) -> ArtifactRegistration | None:
        """Return a registration by artifact id and version."""

    @abstractmethod
    def list_versions(self, artifact_id: str) -> list[str]:
        """List known versions for an artifact id."""

    @abstractmethod
    def find_by_digest(self, artifact_digest: str) -> list[ArtifactRegistration]:
        """Find registrations matching an artifact digest."""


class TrustAnchorStoreBase(ABC):
    """Base interface for trust anchor lookups."""

    @abstractmethod
    def register_artifact_reference(self, registration: ArtifactRegistration) -> None:
        """Register an approved artifact reference as a trust anchor."""

    @abstractmethod
    def get_artifact_reference(self, artifact_id: str, version: str) -> ArtifactRegistration | None:
        """Return a trusted artifact reference if registered."""

    @abstractmethod
    def is_trusted_identity(self, identity: ArtifactIdentity) -> bool:
        """Return whether the exact identity is an approved trusted reference."""
