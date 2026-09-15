"""Security / Trust subsystem — artifact identity, integrity, and trust anchors."""

from mirad.security.constants import (
    CANONICALIZATION_VERSION,
    DIGEST_PREFIX,
    HASH_ALGORITHM,
    MANIFEST_SCHEMA_VERSION,
)
from mirad.security.errors import SecurityErrorCode, SecurityStatus
from mirad.security.hashing import (
    hash_bytes,
    hash_canonical_object,
    hash_file,
    hash_stream,
    hash_text,
    parse_digest,
)
from mirad.security.canonical import canonicalize, canonicalize_to_text
from mirad.security.identity import (
    ArtifactIdentity,
    ArtifactRegistration,
    ArtifactType,
    RegistrationStatus,
)
from mirad.security.manifest import create_manifest, verify_manifest_files
from mirad.security.registration import register_artifact
from mirad.security.verification import VerificationResult, verify_artifact
from mirad.security.trust_anchor import TrustAnchorStore
from mirad.security.stores.memory import InMemoryTrustedArtifactStore
from mirad.security.keys import (
    KeyStatus,
    TrustedKeyRecord,
    generate_ed25519_keypair,
    register_trusted_key,
    verify_signature_with_key,
)
from mirad.security.provenance import ProvenanceRecord, ProvenanceVerificationResult, create_provenance, sign_provenance, verify_provenance
from mirad.security.replay import ReplayCheckResult, ReplayStateStore, check_replay
from mirad.security.audit import AuditEvent, AuditStore, append_audit_event, reconstruct_audit, verify_audit_chain
from mirad.security.checkpoint import CheckpointRecord, CheckpointStore, CheckpointVerificationResult, create_checkpoint, verify_checkpoint
from mirad.security.evidence import EvidenceEnvelope, submit_external_evidence
from mirad.security.findings import FindingSeverity, RecommendedDisposition, SecurityFinding, create_finding, secure_finding
from mirad.security.policy import VerificationPolicy, check_policy_compatibility
from mirad.security.api import (
    append_audit_event_to_store,
    create_signed_checkpoint,
    reconstruct_audit_store,
    verify_manifest,
)

__all__ = [
    "CANONICALIZATION_VERSION",
    "DIGEST_PREFIX",
    "HASH_ALGORITHM",
    "MANIFEST_SCHEMA_VERSION",
    "SecurityErrorCode",
    "SecurityStatus",
    "hash_bytes",
    "hash_canonical_object",
    "hash_file",
    "hash_stream",
    "hash_text",
    "parse_digest",
    "canonicalize",
    "canonicalize_to_text",
    "ArtifactIdentity",
    "ArtifactRegistration",
    "ArtifactType",
    "RegistrationStatus",
    "create_manifest",
    "verify_manifest_files",
    "register_artifact",
    "VerificationResult",
    "verify_artifact",
    "TrustAnchorStore",
    "InMemoryTrustedArtifactStore",
    "TrustedKeyRecord",
    "KeyStatus",
    "generate_ed25519_keypair",
    "register_trusted_key",
    "verify_signature_with_key",
    "ProvenanceRecord",
    "ProvenanceVerificationResult",
    "create_provenance",
    "sign_provenance",
    "verify_provenance",
    "ReplayCheckResult",
    "ReplayStateStore",
    "check_replay",
    "AuditEvent",
    "AuditStore",
    "append_audit_event",
    "reconstruct_audit",
    "verify_audit_chain",
    "CheckpointRecord",
    "CheckpointStore",
    "CheckpointVerificationResult",
    "create_checkpoint",
    "verify_checkpoint",
    "EvidenceEnvelope",
    "submit_external_evidence",
    "FindingSeverity",
    "RecommendedDisposition",
    "SecurityFinding",
    "create_finding",
    "secure_finding",
    "VerificationPolicy",
    "check_policy_compatibility",
    "append_audit_event_to_store",
    "create_signed_checkpoint",
    "reconstruct_audit_store",
    "verify_manifest",
]
