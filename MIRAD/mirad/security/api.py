"""High-level integration API for the security/trust subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirad.security.audit import AuditStore, append_audit_event, reconstruct_audit, verify_audit_chain
from mirad.security.checkpoint import CheckpointStore, create_checkpoint, verify_checkpoint
from mirad.security.evidence import EvidenceEnvelope, submit_external_evidence
from mirad.security.findings import RecommendedDisposition, SecurityFinding, create_finding, secure_finding
from mirad.security.manifest import create_manifest, verify_manifest_files
from mirad.security.provenance import create_provenance, sign_provenance, verify_provenance
from mirad.security.replay import ReplayStateStore, check_replay
from mirad.security.registration import register_artifact
from mirad.security.trust_anchor import TrustAnchorStore
from mirad.security.verification import VerificationResult, verify_artifact


def verify_manifest(manifest: dict[str, Any], root_path: str | Path) -> bool:
    ok, _ = verify_manifest_files(manifest, root_path)
    return ok


def append_audit_event_to_store(store: AuditStore, **kwargs: Any) -> Any:
    return append_audit_event(store, **kwargs)


def create_signed_checkpoint(*, checkpoint_id: str, latest_sequence: int, latest_audit_hash: str, signing_key_id: str, private_key: Any, public_key: str, **kwargs: Any) -> dict[str, Any]:
    return create_checkpoint(
        checkpoint_id=checkpoint_id,
        latest_sequence=latest_sequence,
        latest_audit_hash=latest_audit_hash,
        signing_key_id=signing_key_id,
        private_key=private_key,
        public_key=public_key,
        **kwargs,
    )


def reconstruct_audit_store(store: AuditStore) -> list[Any]:
    return reconstruct_audit(store)


__all__ = [
    "TrustAnchorStore",
    "ReplayStateStore",
    "AuditStore",
    "CheckpointStore",
    "VerificationResult",
    "SecurityFinding",
    "EvidenceEnvelope",
    "register_artifact",
    "verify_artifact",
    "create_manifest",
    "verify_manifest",
    "create_provenance",
    "sign_provenance",
    "verify_provenance",
    "check_replay",
    "create_finding",
    "secure_finding",
    "submit_external_evidence",
    "append_audit_event_to_store",
    "verify_audit_chain",
    "create_signed_checkpoint",
    "verify_checkpoint",
    "reconstruct_audit_store",
    "RecommendedDisposition",
]
