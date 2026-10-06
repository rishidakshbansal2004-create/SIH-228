import json, os, sys, tempfile
from pathlib import Path

# Isolate security state so the test is deterministic and never touches a user's home directory.
tmp = tempfile.TemporaryDirectory()
os.environ["HOME"] = tmp.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trustcv_mirad_dataset_security as sec


def test_signed_provenance_replay_audit_checkpoint():
    sec.ensure_trust_anchor()
    digest = "a" * 64
    p = sec.create_dataset_provenance(
        dataset_name="images.zip", dataset_digest=digest, manifest_digest=None,
        source="Test", owner="Test", version="1", analysis_digest="b"*64, evidence_digest="c"*64,
    )
    v = sec.verify_dataset_provenance(p, expected_dataset_digest=digest)
    assert v["trusted"]
    r = sec.check_dataset_replay(p)
    assert r["valid"]
    state = sec._load_state(); state["seen_event_ids"].append(p["event_id"]); state["seen_nonces"].append(p["nonce"]); state["provenance_sequence"] = p["sequence"]; sec._save_state(state)
    assert not sec.replay_existing_provenance(p)["valid"]
    sec.append_dataset_audit_event("test", {"event_id":p["event_id"]})
    assert sec.verify_dataset_audit()["valid"]
    cp=sec.create_signed_checkpoint()
    assert sec.verify_dataset_checkpoint()["trusted"]
