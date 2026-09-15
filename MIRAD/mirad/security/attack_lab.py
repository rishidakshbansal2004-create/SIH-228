"""Offline attack-lab scenarios used to exercise known MIRAD checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mirad.security.audit import AuditStore, append_audit_event, verify_audit_chain
from mirad.security.checkpoint import create_checkpoint, verify_checkpoint
from mirad.security.evidence import submit_external_evidence
from mirad.security.identity import ArtifactType
from mirad.security.keys import KeyStatus, TrustedKeyRecord, generate_ed25519_keypair
from mirad.security.policy import VerificationPolicy, check_policy_compatibility
from mirad.security.provenance import sign_provenance, verify_provenance
from mirad.security.registration import register_artifact
from mirad.security.replay import ReplayStateStore, check_replay
from mirad.security.trust_anchor import TrustAnchorStore
from mirad.security.verification import verify_artifact


def _result(attack_id: str, attack_name: str, asset: str, mutation: str, expected: bool, actual: bool, reason: str) -> dict[str, Any]:
    return {"attack_id": attack_id, "attack_name": attack_name, "asset": asset, "mutation": mutation, "expected_result": expected, "actual_result": actual, "passed": actual == expected, "reason": reason}


def run_attack_lab() -> list[dict[str, Any]]:
    """Execute deterministic attacks and return structured outcomes."""
    scenarios: list[dict[str, Any]] = []
    store = TrustAnchorStore()
    private, public, key_id = generate_ed25519_keypair(key_id="lab-key")
    store.register_trusted_key(TrustedKeyRecord(key_id=key_id, public_key=public, status=KeyStatus.ACTIVE, trust_anchor_id="anchor-A"))
    other_private, other_public, other_key_id = generate_ed25519_keypair(key_id="other-key")

    def provenance(anchor="anchor-A", signer_private=private, signer_public=public, signer_id=key_id, event_id="lab-provenance"):
        payload = {"event_id": event_id, "nonce": event_id + "-nonce", "sequence": 1, "context": "pipeline-A", "trust_anchor_id": anchor, "input_digest": "sha256:input", "model_digest": "sha256:model", "preprocessing_digest": "sha256:pre", "inference_config_digest": "sha256:config", "output_digest": "sha256:output"}
        return sign_provenance(payload, private_key=signer_private, key_id=signer_id, public_key=signer_public)

    def trusted(record):
        return verify_provenance(record, trust_store=store).provenance_trusted

    with TemporaryDirectory(prefix="mirad-attack-lab-") as tmp:
        root = Path(tmp)
        model = root / "model.bin"
        model.write_bytes(b"trusted-model")
        registration = register_artifact(trust_store=store, artifact_type=ArtifactType.MODEL, artifact_id="model", version="v1", format="bin", artifact_path=model)
        model.write_bytes(b"modified-model")
        modified = verify_artifact(candidate=registration.identity, trust_store=store, artifact_path=model)
        scenarios.append(_result("AL-001", "artifact modification", "model", "change model bytes", False, modified.valid, "Changed bytes must fail digest verification."))
        model.write_bytes(b"trusted-model")
        other = root / "other.bin"
        other.write_bytes(b"different-model")
        substitution = verify_artifact(candidate=registration.identity, trust_store=store, artifact_path=other)
        scenarios.append(_result("AL-002", "artifact substitution", "model", "supply another file", False, substitution.valid, "A different file must not verify as the registered model."))

    original = provenance()
    for attack_id, name, field in (("AL-003", "input tampering", "input_digest"), ("AL-004", "output tampering", "output_digest"), ("AL-005", "provenance payload modification", "model_digest")):
        mutated = dict(original)
        mutated[field] = "sha256:tampered"
        scenarios.append(_result(attack_id, name, "provenance", field, False, trusted(mutated), "Signed provenance binding must reject payload mutation."))
    scenarios.append(_result("AL-006", "provenance signature modification", "provenance", "replace signature", False, trusted(dict(original, signature="sha256:invalid")), "Modified signature must fail."))
    forged = dict(original, signature=original["signature"][:-2] + "AA")
    scenarios.append(_result("AL-007", "forged signature", "provenance", "alter signature bytes", False, trusted(forged), "Forged signature must fail cryptographic verification."))
    wrong_key = provenance(signer_private=other_private, signer_public=other_public, signer_id=other_key_id)
    scenarios.append(_result("AL-008", "wrong signing key", "provenance", "sign with another key", False, trusted(wrong_key), "The registered key identity must match the signature."))
    unknown = provenance(signer_private=other_private, signer_public=other_public, signer_id="unknown-key")
    scenarios.append(_result("AL-009", "unknown signing key", "provenance", "use unregistered key id", False, trusted(unknown), "Unknown keys are not trusted."))
    revoked_store = TrustAnchorStore()
    revoked_store.register_trusted_key(TrustedKeyRecord(key_id=key_id, public_key=public, status=KeyStatus.REVOKED, trust_anchor_id="anchor-A"))
    scenarios.append(_result("AL-010", "revoked key", "provenance", "revoke signer", False, verify_provenance(original, trust_store=revoked_store).provenance_trusted, "Revoked keys are rejected."))
    retired_store = TrustAnchorStore()
    retired_store.register_trusted_key(TrustedKeyRecord(key_id=key_id, public_key=public, status=KeyStatus.RETIRED, trust_anchor_id="anchor-A"))
    scenarios.append(_result("AL-011", "retired key", "provenance", "retire signer", False, verify_provenance(original, trust_store=retired_store).provenance_trusted, "Retired keys are rejected by the strict lab policy."))
    scenarios.append(_result("AL-012", "trust-anchor mismatch", "provenance", "claim anchor B for anchor A key", False, trusted(provenance(anchor="anchor-B")), "Key and record trust anchors must match."))

    def event(event_id, sequence, context="pipeline-A", offset=-10, nonce=None):
        return {"event_id": event_id, "sequence": sequence, "context": context, "nonce": nonce or event_id + "-nonce", "timestamp": (datetime.now(timezone.utc) + timedelta(seconds=offset)).replace(microsecond=0).isoformat().replace("+00:00", "Z")}

    replay = ReplayStateStore()
    check_replay(event("replay", 1), store=replay, expected_context="pipeline-A")
    scenarios.append(_result("AL-013", "valid event replay", "replay", "replay accepted event", False, check_replay(event("replay", 1), store=replay, expected_context="pipeline-A").valid, "Accepted event identity cannot be replayed."))
    scenarios.append(_result("AL-014", "nonce reuse", "replay", "reuse accepted nonce", False, check_replay(event("nonce", 2, nonce="replay-nonce"), store=replay, expected_context="pipeline-A").valid, "Nonces are scoped replay state."))
    scenarios.append(_result("AL-015", "duplicate event", "replay", "duplicate event id", False, check_replay(event("replay", 2), store=replay, expected_context="pipeline-A").valid, "Duplicate event ids are rejected."))
    scenarios.append(_result("AL-016", "sequence rollback", "replay", "sequence 0 after 1", False, check_replay(event("rollback", 0), store=replay, expected_context="pipeline-A").valid, "Sequence must be strictly increasing."))
    scenarios.append(_result("AL-017", "sequence duplication", "replay", "repeat sequence 1", False, check_replay(event("duplicate-sequence", 1), store=replay, expected_context="pipeline-A").valid, "Duplicate sequence values are rejected."))
    scenarios.append(_result("AL-018", "wrong context", "replay", "pipeline B in pipeline A scope", False, check_replay(event("context", 3, context="pipeline-B"), store=replay, expected_context="pipeline-A").valid, "Expected context is explicit and bound."))
    scenarios.append(_result("AL-019", "stale timestamp", "replay", "event older than freshness window", False, check_replay(event("stale", 1, offset=-301), store=ReplayStateStore()).valid, "UTC freshness window rejects stale events."))
    scenarios.append(_result("AL-020", "future timestamp", "replay", "event beyond skew allowance", False, check_replay(event("future", 1, offset=31), store=ReplayStateStore()).valid, "Future skew is bounded."))
    scenarios.append(_result("AL-031", "missing expected context", "replay", "strict binding without expected context", False, check_replay(event("missing-expected", 1), store=ReplayStateStore()).valid, "Strict context binding requires an explicit expected context."))
    scenarios.append(_result("AL-032", "negative sequence", "replay", "sequence -1", False, check_replay(event("negative-sequence", -1), store=ReplayStateStore(), expected_context="pipeline-A").valid, "Strict sequence protection requires sequence >= 1."))
    malformed = event("malformed-timestamp", 1)
    malformed["timestamp"] = "garbage"
    malformed_result = check_replay(malformed, store=ReplayStateStore(), expected_context="pipeline-A")
    scenarios.append(_result("AL-033", "malformed timestamp", "replay", "timestamp garbage", False, malformed_result.valid, "Malformed timestamps return structured rejection."))

    audit_store = AuditStore()
    first = append_audit_event(audit_store, event_type="one", payload={"value": 1})
    second = append_audit_event(audit_store, event_type="two", payload={"value": 2})
    second.payload["value"] = 99
    scenarios.append(_result("AL-021", "audit event modification", "audit", "change payload", False, verify_audit_chain(audit_store.list()), "Audit verification detects event mutation."))
    second.payload["value"] = 2
    truncated = audit_store.list()[:-1]
    scenarios.append(_result("AL-022", "audit deletion/truncation", "audit", "remove latest event", False, verify_checkpoint(create_checkpoint(checkpoint_id="cp-trunc", latest_sequence=2, latest_audit_hash=second.current_hash, signing_key_id=key_id, private_key=private, public_key=public, trust_anchor_id="anchor-A"), trust_store=store, expected_latest_sequence=len(truncated), expected_latest_audit_hash=truncated[-1].current_hash).checkpoint_trusted, "Signed checkpoint detects truncation."))
    audit_store._events.reverse()
    scenarios.append(_result("AL-023", "audit reordering", "audit", "swap events", False, verify_audit_chain(audit_store.list()), "Audit order is part of the chain."))
    audit_store._events.reverse()
    second.previous_hash = "sha256:broken"
    scenarios.append(_result("AL-024", "audit-chain break", "audit", "change previous hash", False, verify_audit_chain(audit_store.list()), "Previous-hash continuity is required."))
    second.previous_hash = first.current_hash
    second.sequence = 5
    scenarios.append(_result("AL-025", "audit sequence gap", "audit", "sequence 1, 5", False, verify_audit_chain(audit_store.list()), "Audit sequence must be contiguous."))
    second.sequence = 2

    checkpoint = create_checkpoint(checkpoint_id="cp-1", latest_sequence=2, latest_audit_hash=second.current_hash, signing_key_id=key_id, private_key=private, public_key=public, trust_anchor_id="anchor-A")
    scenarios.append(_result("AL-026", "checkpoint modification", "checkpoint", "change latest hash", False, verify_checkpoint(dict(checkpoint, latest_audit_hash="sha256:changed"), trust_store=store).checkpoint_trusted, "Signed checkpoint contents are bound."))
    scenarios.append(_result("AL-027", "checkpoint trust-anchor mismatch", "checkpoint", "claim anchor B", False, verify_checkpoint(dict(checkpoint, trust_anchor_id="anchor-B"), trust_store=store).checkpoint_trusted, "Checkpoint signer anchor is enforced."))

    evidence = submit_external_evidence(evidence_id="e-1", evidence_type="check", producer="lab", producer_version="1", method="offline", affected_asset="model", asset_type="MODEL", asset_version="v1", asset_digest="sha256:" + "aa" * 32, evidence_payload={"issue": "none"})
    evidence.validate()
    evidence.evidence_payload["issue"] = "changed"
    scenarios.append(_result("AL-028", "evidence modification", "evidence", "change payload", False, evidence.validate()[0], "Evidence digest is recomputed during validation."))
    evidence = submit_external_evidence(evidence_id="e-2", evidence_type="check", producer="lab", producer_version="1", method="offline", affected_asset="model", asset_type="MODEL", asset_version="v1", asset_digest="sha256:" + "aa" * 32, evidence_payload={"issue": "none"})
    evidence.validate()
    evidence.integrity_digest = "sha256:" + "00" * 32
    scenarios.append(_result("AL-029", "evidence digest modification", "evidence", "replace digest", False, evidence.validate()[0], "Stored evidence digest is compared, never trusted."))
    policy_ok, reason = check_policy_compatibility(VerificationPolicy(), {"freshness_window_seconds": 0, "sequence_required": False, "strict_context_binding": False})
    scenarios.append(_result("AL-030", "policy downgrade", "policy", "remove freshness, sequence, context requirements", False, policy_ok, reason or "Policy downgrade rejected."))
    return scenarios


if __name__ == "__main__":
    results = run_attack_lab()
    failed = [result for result in results if not result["passed"]]
    print(f"MIRAD attack lab: {len(results) - len(failed)}/{len(results)} passed")
    raise SystemExit(1 if failed else 0)
