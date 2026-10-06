"""Offline dataset-security smoke/demo for SIH-26228."""

import json
import tempfile
from pathlib import Path

import trustcv_mirad_dataset_security as sec


def main():
    with tempfile.TemporaryDirectory() as td:
        dataset = Path(td) / "demo_dataset.zip"
        dataset.write_bytes(b"TRUSTCV-SIH-26228-DATASET-DEMO")
        digest = sec.sha256_bytes(dataset.read_bytes())

        record = sec.create_dataset_provenance(
            dataset_name=dataset.name,
            dataset_digest=digest,
            manifest_digest=None,
            source="Synthetic Contributor A",
            owner="TrustCV Demo",
            version="1.0.0",
            analysis_digest=sec.sha256_bytes(b"analysis"),
            evidence_digest=sec.sha256_bytes(b"evidence"),
        )

        verification = sec.verify_dataset_provenance(
            record, expected_dataset_digest=digest
        )
        replay_first = sec.check_dataset_replay(record)
        if replay_first["valid"]:
            state = sec._load_state()
            state["seen_event_ids"].append(record["event_id"])
            state["seen_nonces"].append(record["nonce"])
            state["provenance_sequence"] = record["sequence"]
            sec._save_state(state)

        replay_second = sec.replay_existing_provenance(record)
        audit_event = sec.append_dataset_audit_event(
            "dataset_security_demo",
            {"dataset_sha256": digest, "event_id": record["event_id"]},
        )
        checkpoint = sec.create_signed_checkpoint()
        audit = sec.verify_dataset_audit()
        checkpoint_v = sec.verify_dataset_checkpoint()

        print(json.dumps({
            "dataset_digest": "sha256:" + digest,
            "provenance_signature_valid": verification["signature_valid"],
            "provenance_trusted": verification["trusted"],
            "first_replay_check_valid": replay_first["valid"],
            "second_replay_check_valid": replay_second["valid"],
            "second_replay_reason": replay_second["reason"],
            "audit_chain_valid": audit["valid"],
            "audit_events": audit["events"],
            "checkpoint_signature_valid": checkpoint_v.get("signature_valid"),
            "checkpoint_trusted": checkpoint_v.get("trusted"),
        }, indent=2))


if __name__ == "__main__":
    main()
