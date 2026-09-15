from datetime import datetime, timedelta, timezone

import pytest

from mirad.security.audit import AuditStore, append_audit_event, verify_audit_chain
from mirad.security.checkpoint import create_checkpoint, verify_checkpoint
from mirad.security.evidence import submit_external_evidence
from mirad.security.hashing import hash_canonical_object
from mirad.security.keys import KeyStatus, TrustedKeyRecord, generate_ed25519_keypair
from mirad.security.manifest import create_manifest, verify_manifest_files
from mirad.security.policy import VerificationPolicy, check_policy_compatibility
from mirad.security.provenance import sign_provenance, verify_provenance
from mirad.security.replay import ReplayStateStore, check_replay
from mirad.security.trust_anchor import TrustAnchorStore
from mirad.security.identity import ArtifactType


def _key_store(status=KeyStatus.ACTIVE, anchor="anchor-A"):
    store = TrustAnchorStore()
    private, public, key_id = generate_ed25519_keypair(key_id="hardening-key")
    store.register_trusted_key(TrustedKeyRecord(key_id=key_id, public_key=public, status=status, trust_anchor_id=anchor))
    return store, private, public, key_id


def _provenance(private, public, key_id, anchor="anchor-A"):
    return sign_provenance(
        {
            "provenance_version": "1.0", "event_id": "p-1",
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "nonce": "n-1", "sequence": 1, "context": "pipeline-A",
            "input_digest": "sha256:" + "aa" * 32,
            "dataset_identity": "dataset-1", "dataset_digest": "sha256:" + "bb" * 32,
            "model_identity": "model-1", "model_digest": "sha256:" + "cc" * 32,
            "preprocessing_digest": "sha256:" + "dd" * 32,
            "inference_config_digest": "sha256:" + "ee" * 32,
            "output_digest": "sha256:" + "ff" * 32,
            "hash_algorithm": "sha256", "canonicalization_version": "1.0",
            "signature_algorithm": "ed25519", "trust_anchor_id": anchor,
            "verification_policy_version": "1.0",
        },
        private_key=private, key_id=key_id, public_key=public,
    )


def test_incomplete_signed_provenance_fails_closed():
    store, private, public, key_id = _key_store()
    incomplete = sign_provenance(
        {"event_id": "incomplete", "nonce": "n", "sequence": 1, "context": "pipeline-A"},
        private_key=private, key_id=key_id, public_key=public,
    )
    result = verify_provenance(incomplete, trust_store=store)
    assert result.provenance_trusted is False
    assert result.failure_code == "INVALID_SCHEMA"


def test_trusted_key_substitution_is_rejected():
    store, _, public, key_id = _key_store()
    _, replacement_public, _ = generate_ed25519_keypair(key_id="replacement")
    with pytest.raises(ValueError):
        store.register_trusted_key(TrustedKeyRecord(key_id=key_id, public_key=replacement_public))


def test_trusted_provenance_rejects_untrusted_key_and_anchor_mismatch():
    store, private, public, key_id = _key_store()
    signed = _provenance(private, public, key_id)
    assert verify_provenance(signed, trust_store=store).provenance_trusted
    assert verify_provenance(signed, trust_store=TrustAnchorStore(), public_key=public).provenance_trusted is False
    wrong_anchor = dict(signed)
    wrong_anchor["trust_anchor_id"] = "anchor-B"
    assert verify_provenance(wrong_anchor, trust_store=store).trust_anchor_valid is False
    assert verify_provenance(wrong_anchor, trust_store=store).provenance_trusted is False


def test_raw_signature_verification_is_distinct_from_trust():
    store, private, public, key_id = _key_store()
    result = verify_provenance(_provenance(private, public, key_id), public_key=public)
    assert result.signature_valid is True
    assert result.provenance_trusted is False


def test_revoked_and_retired_keys_rejected():
    for status in (KeyStatus.REVOKED, KeyStatus.RETIRED):
        store, private, public, key_id = _key_store(status=status)
        assert verify_provenance(_provenance(private, public, key_id), trust_store=store).provenance_trusted is False


def test_checkpoint_trust_anchor_and_unknown_key_rejected():
    store, private, public, key_id = _key_store()
    checkpoint = create_checkpoint(checkpoint_id="c-1", latest_sequence=1, latest_audit_hash="sha256:x", signing_key_id=key_id, private_key=private, public_key=public, trust_anchor_id="anchor-A")
    assert verify_checkpoint(checkpoint, trust_store=store).checkpoint_trusted
    wrong = dict(checkpoint, trust_anchor_id="anchor-B")
    assert verify_checkpoint(wrong, trust_store=store).trust_anchor_valid is False
    assert verify_checkpoint(checkpoint, trust_store=TrustAnchorStore(), public_key=public).checkpoint_trusted is False


def _event(event_id, sequence, context="pipeline-A", age_seconds=-10):
    return {"event_id": event_id, "sequence": sequence, "context": context, "nonce": f"nonce-{event_id}", "timestamp": (datetime.now(timezone.utc) + timedelta(seconds=age_seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")}


def test_replay_context_sequence_and_time_boundaries():
    store = ReplayStateStore()
    assert check_replay(_event("1", 1), store=store, expected_context="pipeline-A").valid
    assert check_replay(_event("2", 2), store=store, expected_context="pipeline-A").valid
    assert check_replay(_event("3", 2), store=store, expected_context="pipeline-A").valid is False
    assert check_replay(_event("4", 1), store=store, expected_context="pipeline-A").valid is False
    assert check_replay(_event("5", 3, context="pipeline-B"), store=store, expected_context="pipeline-A").valid is False
    assert check_replay(_event("6", None), store=ReplayStateStore(), expected_context="pipeline-A").valid is False
    assert check_replay(_event("7", 1, age_seconds=-301), store=ReplayStateStore()).valid is False
    assert check_replay(_event("8", 1, age_seconds=31), store=ReplayStateStore()).valid is False


def test_replay_strict_context_and_negative_sequence_fail_closed():
    event = _event("missing-context", 1)
    result = check_replay(event, store=ReplayStateStore())
    assert result.valid is False
    assert result.failure_code == "CONTEXT_MISMATCH"
    negative = check_replay(_event("negative", -1), store=ReplayStateStore(), expected_context="pipeline-A")
    assert negative.valid is False
    assert negative.failure_code == "SEQUENCE_VIOLATION"


def test_malformed_timestamp_returns_structured_rejection():
    malformed = _event("malformed", 1)
    malformed["timestamp"] = "garbage"
    result = check_replay(malformed, store=ReplayStateStore(), expected_context="pipeline-A")
    assert result.valid is False
    assert result.failure_code == "INVALID_TIMESTAMP"


def _evidence():
    return submit_external_evidence(evidence_id="e-1", evidence_type="check", producer="analyst", producer_version="1", method="test", affected_asset="a", asset_type="MODEL", asset_version="v1", asset_digest="sha256:" + "aa" * 32, evidence_payload={"x": 1})


def test_evidence_digest_recomputed_and_trust_distinguished():
    evidence = _evidence()
    assert evidence.validate()[0]
    assert evidence.trust_state()["producer_trusted"] is False
    evidence.evidence_payload["x"] = 2
    assert evidence.validate()[0] is False
    evidence = _evidence()
    evidence.integrity_digest = "sha256:" + "00" * 32
    assert evidence.validate()[0] is False
    evidence = _evidence()
    evidence.producer = "changed"
    assert evidence.validate()[0] is False


def test_policy_falsey_downgrades_rejected():
    policy = VerificationPolicy()
    for incoming in ({"freshness_window_seconds": 0}, {"sequence_required": False}, {"strict_context_binding": False}, {"nonce_required": False}, {"policy_version": "0.1"}, {"signature_algorithm": "none"}):
        assert check_policy_compatibility(policy, incoming)[0] is False


def test_provenance_versions_and_algorithm_are_policy_checked():
    store, private, public, key_id = _key_store()
    for field, value in (("provenance_version", "9.0"), ("canonicalization_version", "9.0"), ("verification_policy_version", "9.0"), ("signature_algorithm", "rsa")):
        record = _provenance(private, public, key_id)
        record[field] = value
        result = verify_provenance(record, trust_store=store)
        assert result.provenance_trusted is False
        assert result.policy_valid is False


def test_audit_ordering_and_tampering_rejected():
    store = AuditStore()
    first = append_audit_event(store, event_type="one", payload={"v": 1})
    second = append_audit_event(store, event_type="two", payload={"v": 2})
    assert verify_audit_chain(store.list())
    second.sequence = 4
    assert verify_audit_chain(store.list()) is False
    second.sequence = 2
    second.payload["v"] = 99
    assert verify_audit_chain(store.list()) is False
    assert first.sequence == 1


def test_manifest_traversal_and_absolute_paths_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "safe.txt").write_bytes(b"safe")
    manifest = create_manifest(artifact_type=ArtifactType.MODEL, artifact_id="a", version="1", format="bin", root_path=root)
    manifest["files"][0]["normalized_path"] = "../outside.txt"
    manifest["manifest_digest"] = hash_canonical_object({k: v for k, v in manifest.items() if k != "manifest_digest"})
    assert verify_manifest_files(manifest, root)[0] is False
    manifest["files"][0]["normalized_path"] = "C:/outside.txt"
    manifest["manifest_digest"] = hash_canonical_object({k: v for k, v in manifest.items() if k != "manifest_digest"})
    assert verify_manifest_files(manifest, root)[0] is False
