"""Essential trust anchor smoke tests."""

from mirad.security.identity import ArtifactIdentity, ArtifactRegistration, ArtifactType, RegistrationStatus
from mirad.security.trust_anchor import TrustAnchorStore


def test_unregistered_identity_not_trusted():
    store = TrustAnchorStore()
    identity = ArtifactIdentity(
        artifact_type=ArtifactType.MODEL,
        artifact_id="unknown",
        version="1.0.0",
        artifact_digest="sha256:" + "ab" * 32,
    )
    assert store.is_trusted_identity(identity) is False


def test_registered_approved_identity_trusted():
    store = TrustAnchorStore()
    identity = ArtifactIdentity(
        artifact_type=ArtifactType.DATASET,
        artifact_id="ds-1",
        version="1.0.0",
        artifact_digest="sha256:" + "cd" * 32,
    )
    store.register_artifact_reference(
        ArtifactRegistration(
            identity=identity,
            status=RegistrationStatus.APPROVED,
            registration_timestamp="2026-01-01T00:00:00Z",
        )
    )
    assert store.is_trusted_identity(identity) is True


def test_digest_mismatch_not_trusted():
    store = TrustAnchorStore()
    registered = ArtifactIdentity(
        artifact_type=ArtifactType.MODEL,
        artifact_id="m-1",
        version="1.0.0",
        artifact_digest="sha256:" + "11" * 32,
    )
    store.register_artifact_reference(
        ArtifactRegistration(
            identity=registered,
            status=RegistrationStatus.APPROVED,
            registration_timestamp="2026-01-01T00:00:00Z",
        )
    )
    candidate = ArtifactIdentity(
        artifact_type=ArtifactType.MODEL,
        artifact_id="m-1",
        version="1.0.0",
        artifact_digest="sha256:" + "22" * 32,
    )
    assert store.is_trusted_identity(candidate) is False
