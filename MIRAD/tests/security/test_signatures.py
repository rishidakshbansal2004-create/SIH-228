from datetime import datetime, timedelta, timezone

from mirad.security.evidence import submit_external_evidence
from mirad.security.keys import KeyStatus, TrustedKeyRecord, generate_ed25519_keypair, register_trusted_key
from mirad.security.policy import VerificationPolicy, check_policy_compatibility
from mirad.security.provenance import sign_provenance, verify_provenance
from mirad.security.replay import ReplayStateStore, check_replay
from mirad.security.trust_anchor import TrustAnchorStore


def test_ed25519_round_trip_and_trusted_key_registration():
    store = TrustAnchorStore()
    private_key, public_key, key_id = generate_ed25519_keypair(key_id="dev-key-1")
    record = TrustedKeyRecord(
        key_id=key_id,
        public_key=public_key,
        status=KeyStatus.ACTIVE,
        trust_anchor_id="anchor-1",
        development_only=True,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    register_trusted_key(store, record)

    payload = {
        "provenance_version": "1.0", "event_id": "evt-1", "timestamp": "2026-09-14T18:00:00Z",
        "nonce": "abc", "sequence": 1, "context": "test",
        "input_digest": "sha256:" + "aa" * 32, "dataset_identity": "dataset-1", "dataset_digest": "sha256:" + "bb" * 32,
        "model_identity": "demo-model", "model_digest": "sha256:" + "cc" * 32,
        "preprocessing_digest": "sha256:" + "dd" * 32, "inference_config_digest": "sha256:" + "ee" * 32,
        "output_digest": "sha256:" + "ff" * 32, "hash_algorithm": "sha256",
        "canonicalization_version": "1.0", "signature_algorithm": "ed25519",
        "trust_anchor_id": "anchor-1", "verification_policy_version": "1.0",
    }
    signed = sign_provenance(payload, private_key=private_key, key_id=key_id, public_key=public_key)
    assert verify_provenance(signed, trust_store=store, public_key=public_key).provenance_trusted is True


def test_replay_rejects_duplicate_event():
    store = ReplayStateStore()
    now = datetime.now(timezone.utc)
    event = {
        "event_id": "evt-1",
        "sequence": 1,
        "timestamp": (now - timedelta(seconds=30)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "context": "session-1",
        "nonce": "nonce-1",
    }
    assert check_replay(event, store=store, freshness_window_seconds=300, expected_context="session-1").valid is True
    assert check_replay(event, store=store, freshness_window_seconds=300, expected_context="session-1").valid is False


def test_policy_rejects_downgrade_and_validates_evidence():
    policy = VerificationPolicy(freshness_window_seconds=300)
    allowed, reason = check_policy_compatibility(policy, {"freshness_window_seconds": 600})
    assert allowed is False and reason == "POLICY_DOWNGRADE_DETECTED"

    evidence = submit_external_evidence(
        evidence_id="e-1",
        evidence_type="dataset-analysis",
        producer="analyst",
        producer_version="1.0",
        method="statistical-check",
        affected_asset="dataset-a",
        asset_type="DATASET",
        asset_version="v1",
        asset_digest="sha256:" + "aa" * 32,
        evidence_payload={"issue": "suspicious"},
        confidence=0.72,
        limitations=["No behavioral proof"],
    )
    assert evidence.validate()[0] is True
