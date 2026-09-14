"""Offline, human-readable and machine-readable MIRAD security demonstration."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mirad.security.audit import AuditStore, append_audit_event, verify_audit_chain
from mirad.security.checkpoint import create_checkpoint, verify_checkpoint
from mirad.security.findings import FindingSeverity, RecommendedDisposition, create_finding
from mirad.security.hashing import hash_bytes, hash_canonical_object, hash_file
from mirad.security.identity import ArtifactType
from mirad.security.keys import KeyStatus, TrustedKeyRecord, generate_ed25519_keypair
from mirad.security.provenance import create_provenance, sign_provenance, verify_provenance
from mirad.security.registration import register_artifact
from mirad.security.replay import ReplayStateStore, check_replay
from mirad.security.trust_anchor import TrustAnchorStore
from mirad.security.verification import verify_artifact


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _event_dict(event: Any) -> dict[str, Any]:
    return event.to_dict()


def _verification_dict(result: Any) -> dict[str, Any]:
    return {name: getattr(result, name) for name in result.__dataclass_fields__}


def _attack_result(*, attack_type: str, affected_field: str, original: Any, modified: Any, verification: Any, disposition: str) -> dict[str, Any]:
    return {
        "attack_type": attack_type,
        "affected_field": affected_field,
        "original_value": original,
        "modified_value": modified,
        "signature_valid": verification.signature_valid,
        "provenance_trusted": verification.provenance_trusted,
        "failure_code": verification.failure_code,
        "disposition": disposition,
    }


def run_demo(output_dir: str | Path = "demo_output") -> dict[str, Any]:
    """Execute one deterministic security scenario and persist its evidence."""
    output_path = Path(output_dir)
    attacks_path = output_path / "attacks"
    output_path.mkdir(parents=True, exist_ok=True)
    trust_store = TrustAnchorStore()
    private_key, public_key, key_id = generate_ed25519_keypair(key_id="demo-key")
    trust_store.register_trusted_key(TrustedKeyRecord(
        key_id=key_id, public_key=public_key, status=KeyStatus.ACTIVE,
        trust_anchor_id="default", development_only=True,
    ))

    with TemporaryDirectory(prefix="mirad-demo-") as tmpdir:
        root = Path(tmpdir)
        model_file = root / "model.bin"
        dataset_file = root / "dataset.bin"
        input_file = root / "input.bin"
        output_file = root / "output.json"
        model_file.write_bytes(b"MIRAD-DEMO-MODEL-v1")
        dataset_file.write_bytes(b"MIRAD-DEMO-DATASET-v1\nitem-001\n")
        input_file.write_bytes(b"MIRAD-DEMO-INPUT\nframe-001\n")
        output_bytes = b'{"label":"vehicle","score":0.91}'
        output_file.write_bytes(output_bytes)

        model_registration = register_artifact(
            trust_store=trust_store, artifact_type=ArtifactType.MODEL,
            artifact_id="demo-model", version="v1", format="onnx", artifact_path=model_file,
        )
        dataset_registration = register_artifact(
            trust_store=trust_store, artifact_type=ArtifactType.DATASET,
            artifact_id="demo-dataset", version="v1", format="bin", artifact_path=dataset_file,
        )
        model_check = verify_artifact(candidate=model_registration.identity, trust_store=trust_store, artifact_path=model_file)
        dataset_check = verify_artifact(candidate=dataset_registration.identity, trust_store=trust_store, artifact_path=dataset_file)

        model_tampered = root / "model-tampered.bin"
        model_tampered.write_bytes(model_file.read_bytes()[:-1] + b"X")
        tampered_identity = replace(model_registration.identity, artifact_digest=hash_file(model_tampered))
        model_attack = verify_artifact(candidate=tampered_identity, trust_store=trust_store, artifact_path=model_tampered)
        dataset_tampered = root / "dataset-tampered.bin"
        dataset_tampered.write_bytes(dataset_file.read_bytes() + b"tampered")
        dataset_tampered_identity = replace(dataset_registration.identity, artifact_digest=hash_file(dataset_tampered))
        dataset_attack = verify_artifact(candidate=dataset_tampered_identity, trust_store=trust_store, artifact_path=dataset_tampered)

        input_digest = hash_file(input_file)
        output_digest = hash_file(output_file)
        preprocessing = {"resize": [640, 640], "normalization": "uint8_to_float32", "color_format": "RGB"}
        inference_config = {"confidence_threshold": 0.5, "nms": {"enabled": True, "iou": 0.45}}
        preprocessing_digest = hash_canonical_object(preprocessing)
        inference_config_digest = hash_canonical_object(inference_config)
        provenance = create_provenance(
            event_id="evt-demo-001", nonce="nonce-demo-001", sequence=1, context="offline-demo",
            input_digest=input_digest, dataset_identity="demo-dataset", dataset_digest=dataset_registration.identity.artifact_digest,
            model_identity="demo-model", model_digest=model_registration.identity.artifact_digest,
            preprocessing_digest=preprocessing_digest, inference_config_digest=inference_config_digest,
            output_digest=output_digest, metadata={"fixture": "DEMO FIXTURE - NOT A REAL OPERATIONAL MODEL"},
        )
        signed_provenance = sign_provenance(provenance, private_key=private_key, key_id=key_id, public_key=public_key)
        provenance_check = verify_provenance(signed_provenance, trust_store=trust_store)

        attack_results: list[dict[str, Any]] = []
        for field_name in ("output_digest", "model_digest"):
            altered = copy.deepcopy(signed_provenance)
            original_value = altered[field_name]
            altered[field_name] = hash_bytes((original_value + ":tampered").encode("utf-8"))
            tampered_check = verify_provenance(altered, trust_store=trust_store)
            attack_results.append(_attack_result(
                attack_type="PROVENANCE_TAMPERING", affected_field=field_name,
                original=original_value, modified=altered[field_name], verification=tampered_check,
                disposition="REJECTED",
            ))
            _write_json(attacks_path / f"provenance_{field_name}_tampered.json", altered)

        input_tampered = hash_bytes(input_file.read_bytes() + b"tampered")
        output_tampered = hash_bytes(output_bytes + b"tampered")
        _write_json(attacks_path / "input_tampered.json", {"original_digest": input_digest, "modified_digest": input_tampered})
        _write_json(attacks_path / "output_tampered.json", {"expected_digest": output_digest, "observed_digest": output_tampered})
        _write_json(attacks_path / "dataset_tampered.json", {"expected_digest": dataset_registration.identity.artifact_digest, "observed_digest": hash_file(dataset_tampered)})
        attack_results.extend([
            {"attack_type": "INPUT_TAMPERING", "affected_field": "input_digest", "original_value": input_digest, "modified_value": input_tampered, "detected": input_digest != input_tampered, "disposition": "QUARANTINE"},
            {"attack_type": "OUTPUT_TAMPERING", "affected_field": "output_digest", "original_value": output_digest, "modified_value": output_tampered, "detected": output_digest != output_tampered, "disposition": "QUARANTINE"},
            {"attack_type": "MODEL_TAMPERING", "affected_field": "artifact_digest", "original_value": model_registration.identity.artifact_digest, "modified_value": hash_file(model_tampered), "detected": not model_attack.valid, "failure_code": model_attack.failure_codes[0] if model_attack.failure_codes else None, "disposition": "QUARANTINE"},
            {"attack_type": "DATASET_TAMPERING", "affected_field": "artifact_digest", "original_value": dataset_registration.identity.artifact_digest, "modified_value": hash_file(dataset_tampered), "detected": not dataset_attack.valid, "failure_code": dataset_attack.failure_codes[0] if dataset_attack.failure_codes else None, "disposition": "REVIEW"},
        ])

        replay_store = ReplayStateStore()
        replay_event = {k: signed_provenance[k] for k in ("event_id", "timestamp", "context", "sequence", "nonce")}
        first_replay = check_replay(replay_event, store=replay_store, expected_context="offline-demo")
        second_replay = check_replay(replay_event, store=replay_store, expected_context="offline-demo")

        audit_store = AuditStore()
        audit_events = []
        for event_type, payload in (
            ("MODEL_VERIFICATION", model_check.to_dict()), ("INPUT_VERIFICATION", {"digest": input_digest}),
            ("PROVENANCE_CREATED", {"event_id": signed_provenance["event_id"]}),
            ("PROVENANCE_VERIFIED", _verification_dict(provenance_check)),
            ("REPLAY_CHECK", {"first": first_replay.valid, "replay": second_replay.failure_code}),
            ("SECURITY_FINDING", {"attack": "OUTPUT_TAMPERING"}),
            ("CHECKPOINT_CREATED", {"planned_latest_sequence": 7}),
        ):
            audit_events.append(append_audit_event(audit_store, event_type=event_type, payload=payload, actor="mirad-demo"))
        chain_valid = verify_audit_chain(audit_store.list())
        audit_json = [_event_dict(event) for event in audit_events]
        _write_json(output_path / "audit_log.json", audit_json)
        audit_tampered = copy.deepcopy(audit_json)
        audit_tampered[1]["payload"]["digest"] = output_tampered
        _write_json(attacks_path / "audit_tampered.json", audit_tampered)
        audit_attack = {"attack_type": "AUDIT_TAMPERING", "affected_field": "payload.digest", "detected": audit_tampered[1] != audit_json[1] and not verify_audit_chain([type(audit_events[0])(**item) for item in audit_tampered]), "disposition": "REJECTED", "failure_code": "AUDIT_CHAIN_BROKEN"}
        attack_results.append(audit_attack)

        checkpoint = create_checkpoint(checkpoint_id="checkpoint-demo-001", latest_sequence=len(audit_events), latest_audit_hash=audit_events[-1].current_hash, signing_key_id=key_id, private_key=private_key, public_key=public_key)
        checkpoint_check = verify_checkpoint(checkpoint, trust_store=trust_store, expected_latest_audit_hash=audit_events[-1].current_hash, expected_latest_sequence=len(audit_events))
        _write_json(output_path / "checkpoint.json", checkpoint)

        finding = create_finding(
            finding_id="finding-output-integrity-001", finding_type="OUTPUT_DIGEST_MISMATCH",
            affected_asset="demo-model/inference-output", affected_asset_type="OUTPUT", asset_version="v1",
            asset_digest=output_tampered, reason="Observed output digest differs from the digest bound to signed provenance.",
            evidence={"expected_digest": output_digest, "observed_digest": output_tampered},
            severity=FindingSeverity.HIGH, recommended_disposition=RecommendedDisposition.QUARANTINE,
            limitations=["Integrity mismatch does not establish attacker identity or malicious intent."], provenance_id=signed_provenance["event_id"],
        ).to_dict()
        _write_json(output_path / "provenance.json", signed_provenance)
        verification_result = {"schema": provenance_check.schema_valid, "signature": provenance_check.signature_valid, "signing_key": provenance_check.key_trusted, "trust_anchor": provenance_check.trust_anchor_valid, "policy": provenance_check.policy_valid, "overall": provenance_check.provenance_trusted, "provenance": _verification_dict(provenance_check), "model": model_check.to_dict(), "dataset": dataset_check.to_dict(), "checkpoint": _verification_dict(checkpoint_check)}
        _write_json(output_path / "verification_result.json", verification_result)
        report = {
            "status": "ok", "run_metadata": {"offline": True, "fixture": True, "generated_at": datetime.now(timezone.utc).isoformat()},
            "model": {"identity": model_registration.identity.to_dict(), "digest": model_registration.identity.artifact_digest, "trusted_reference": model_registration.identity.artifact_digest, "admission": model_check.to_dict()},
            "dataset": {"identity": dataset_registration.identity.to_dict(), "digest": dataset_registration.identity.artifact_digest, "admission": dataset_check.to_dict()},
            "inputs": {"digest": input_digest}, "inference": {"preprocessing": preprocessing, "preprocessing_digest": preprocessing_digest, "inference_config": inference_config, "inference_config_digest": inference_config_digest, "output_digest": output_digest},
            "provenance": signed_provenance, "provenance_verification": verification_result, "replay": {"first": _verification_dict(first_replay), "replay": _verification_dict(second_replay)},
            "audit": {"records": audit_json, "chain_valid": chain_valid}, "checkpoint": {"record": checkpoint, "verification": _verification_dict(checkpoint_check)},
            "attacks": attack_results, "findings": [finding], "security_control_trace": {"threat": "Inference record substitution", "attack": "Modify stored inference output", "evidence": "Output digest mismatch", "control": "Output digest plus signed provenance", "decision": "QUARANTINE"},
            "final_assurance": {"original_event": "TRUSTED", "attacked_events": "REJECTED_OR_QUARANTINED", "model_admission": model_check.valid, "dataset_admission": dataset_check.valid, "audit_chain": chain_valid, "checkpoint": checkpoint_check.checkpoint_trusted},
            "limitations": ["DEMO FIXTURE; not a real operational model.", "Artifact integrity does not prove behavioral safety or contributor intent.", "Audit chaining is tamper-evident, not absolute immutability."],
            "generated_artifact_paths": {"provenance": str(output_path / "provenance.json"), "audit": str(output_path / "audit_log.json"), "checkpoint": str(output_path / "checkpoint.json"), "verification": str(output_path / "verification_result.json"), "report": str(output_path / "security_demo_report.json")},
            "artifact_verification": {"valid": model_check.valid, "status": model_check.status.value}, "provenance_verification_legacy": provenance_check.provenance_trusted, "replay_check": {"valid": second_replay.valid, "status": second_replay.status, "reason": second_replay.reason}, "first_replay_check": {"valid": first_replay.valid, "status": first_replay.status},
        }
        _write_json(output_path / "security_demo_report.json", report)
        return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIRAD offline security demonstration")
    parser.add_argument("--json", action="store_true", help="Emit the actual structured report as JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_demo()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print("MIRAD - SECURITY / TRUST ASSURANCE SUMMARY")
    print("ORIGINAL -> TRUSTED")
    print(f"Model digest: {report['model']['digest']}")
    print(f"Input digest: {report['inputs']['digest']}")
    print(f"Output digest: {report['inference']['output_digest']}")
    print(f"Ed25519 key: {report['provenance']['key_id']} (private key not exposed)")
    print(f"Provenance signature: {'VALID' if report['provenance_verification']['signature'] else 'INVALID'}")
    print(f"Replay: {report['replay']['replay']['failure_code']} -> REJECTED")
    print(f"Audit chain: {'VALID' if report['audit']['chain_valid'] else 'INVALID'}")
    print(f"Checkpoint: {'TRUSTED' if report['checkpoint']['verification']['checkpoint_trusted'] else 'REJECTED'}")
    print("TAMPERED -> DETECTED -> REJECTED / QUARANTINED")
    print(f"Security findings: {len(report['findings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
