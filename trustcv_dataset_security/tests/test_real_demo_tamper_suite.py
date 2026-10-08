"""
REAL DEMO TAMPER & VERIFICATION TEST SUITE (Tests A through F)
Strictly executes tests A, B, C, D, E, F as required by Sections 17-22.
"""

import json
import base64
import tempfile
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trustcv_mirad_dataset_security import (
    verify_signed_dataset_manifest,
    sha256_bytes,
    ensure_trust_anchor,
    canonicalize,
)
from contributor_backend import get_contributor_backend


def run_real_demo_suite():
    print("=" * 80)
    print("RUNNING REAL DEMO VERIFICATION & TAMPER SUITE (TESTS A - F)")
    print("=" * 80)

    dataset_path = Path("images(1).zip")
    if not dataset_path.exists():
        dataset_path = Path("images.zip")
    assert dataset_path.exists(), f"Real dataset not found: {dataset_path}"

    real_bytes = dataset_path.read_bytes()
    real_dataset_sha256 = sha256_bytes(real_bytes)
    print(f"[INFO] Real Dataset: {dataset_path.name}")
    print(f"[INFO] Real Dataset Digest: {real_dataset_sha256}")

    manifest_path = Path("demo_output/signed_manifests/final_demo.manifest.json")
    assert manifest_path.exists(), f"Manifest artifact missing: {manifest_path}"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    print(f"[INFO] Genuine Manifest Path: {manifest_path}")
    print(f"[INFO] Genuine Manifest Digest: {manifest_sha256}")

    expected_contrib = manifest["contributor_id"]
    expected_contribution = manifest.get("contribution_id")
    expected_batch = manifest.get("batch_id")
    dataset_name = manifest.get("dataset_name", dataset_path.name)
    print(f"[INFO] Bound Contributor: {expected_contrib}")
    print(f"[INFO] Bound Contribution: {expected_contribution}")
    print(f"[INFO] Bound Batch: {expected_batch}")

    # =========================================================================
    # TEST A: GENUINE DATASET & MANIFEST VERIFICATION
    # =========================================================================
    print("\n--- TEST A: GENUINE DATASET & MANIFEST ---")
    res_a = verify_signed_dataset_manifest(
        manifest_bytes,
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id=expected_contrib,
        expected_contribution_id=expected_contribution,
        expected_batch_id=expected_batch,
        actual_dataset_name=dataset_name,
    )
    print(f"Overall Status: {res_a['overall']}")
    print(f"Manifest Integrity: {res_a['checks']['manifest_integrity']['status']}")
    print(f"Digital Signature:  {res_a['checks']['digital_signature']['status']}")
    print(f"Trusted Key:        {res_a['checks']['digital_signature']['trusted_key']}")
    print(f"Dataset Digest:     {res_a['checks']['dataset_digest']['status']}")
    print(f"Contributor:        {res_a['checks']['contributor']['status']}")
    print(f"Contribution:       {res_a['checks']['contribution']['status']}")
    print(f"Batch:              {res_a['checks']['batch']['status']}")

    assert res_a["overall"] == "PASS"
    assert res_a["checks"]["manifest_integrity"]["status"] == "VALID"
    assert res_a["checks"]["digital_signature"]["status"] == "VALID"
    assert res_a["checks"]["digital_signature"]["trusted_key"] is True
    assert res_a["checks"]["dataset_digest"]["status"] == "MATCH"
    assert res_a["checks"]["contributor"]["status"] == "MATCH"
    assert res_a["checks"]["contribution"]["status"] == "MATCH"
    assert res_a["checks"]["batch"]["status"] == "MATCH"
    print("[PASS] TEST A SUCCESSFUL: Genuine dataset verified with 100% strict PASS.")

    # =========================================================================
    # TEST B: DATASET TAMPERING (1 byte flipped in dataset)
    # =========================================================================
    print("\n--- TEST B: DATASET TAMPERING (1-BYTE MUTATION) ---")
    tampered_bytes = bytearray(real_bytes)
    # Mutate byte 100
    tampered_bytes[100] = (tampered_bytes[100] ^ 0xFF)
    tampered_dataset_sha256 = sha256_bytes(bytes(tampered_bytes))
    assert tampered_dataset_sha256 != real_dataset_sha256

    res_b = verify_signed_dataset_manifest(
        manifest_bytes,
        actual_dataset_digest=tampered_dataset_sha256,
        expected_contributor_id=expected_contrib,
        expected_contribution_id=expected_contribution,
        expected_batch_id=expected_batch,
        actual_dataset_name=dataset_name,
    )
    print(f"Overall Status:     {res_b['overall']}")
    print(f"Digital Signature:  {res_b['checks']['digital_signature']['status']}")
    print(f"Trusted Key:        {res_b['checks']['digital_signature']['trusted_key']}")
    print(f"Dataset Digest:     {res_b['checks']['dataset_digest']['status']}")

    assert res_b["overall"] == "DATASET_DIGEST_MISMATCH"
    assert res_b["checks"]["digital_signature"]["status"] == "VALID"
    assert res_b["checks"]["digital_signature"]["trusted_key"] is True
    assert res_b["checks"]["dataset_digest"]["status"] == "MISMATCH"
    print("[PASS] TEST B SUCCESSFUL: 1-byte tamper rejected with DATASET_DIGEST_MISMATCH.")

    # =========================================================================
    # TEST C: MANIFEST TAMPERING (field modified without resigning)
    # =========================================================================
    print("\n--- TEST C: MANIFEST TAMPERING (FIELD MUTATED WITHOUT RESIGNING) ---")
    tampered_manifest = dict(manifest)
    tampered_manifest["contributor_id"] = "CONTRIB-TAMPERED-ATTACKER"
    tampered_manifest_bytes = json.dumps(tampered_manifest, indent=2).encode("utf-8")

    res_c = verify_signed_dataset_manifest(
        tampered_manifest_bytes,
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id="CONTRIB-TAMPERED-ATTACKER",
        expected_contribution_id=expected_contribution,
        expected_batch_id=expected_batch,
        actual_dataset_name=dataset_name,
    )
    print(f"Overall Status:    {res_c['overall']}")
    print(f"Digital Signature: {res_c['checks']['digital_signature']['status']}")

    assert res_c["overall"] == "INVALID_SIGNATURE"
    assert res_c["checks"]["digital_signature"]["status"] == "INVALID_SIGNATURE"
    print("[PASS] TEST C SUCCESSFUL: Tampered manifest payload rejected with INVALID_SIGNATURE.")

    # =========================================================================
    # TEST D: WRONG CONTRIBUTOR BINDING (Different registered contributor)
    # =========================================================================
    print("\n--- TEST D: WRONG CONTRIBUTOR (LEGITIMATE KEY, WRONG IDENTITY) ---")
    different_contrib = "CONTRIB-A" if expected_contrib != "CONTRIB-A" else "C 4"
    res_d = verify_signed_dataset_manifest(
        manifest_bytes,
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id=different_contrib,
        expected_contribution_id=expected_contribution,
        expected_batch_id=expected_batch,
        actual_dataset_name=dataset_name,
    )
    print(f"Overall Status:      {res_d['overall']}")
    print(f"Digital Signature:   {res_d['checks']['digital_signature']['status']}")
    print(f"Trusted Key:         {res_d['checks']['digital_signature']['trusted_key']}")
    print(f"Dataset Digest:      {res_d['checks']['dataset_digest']['status']}")
    print(f"Contributor Binding: {res_d['checks']['contributor']['status']}")

    assert res_d["overall"] == "CONTRIBUTOR_MISMATCH"
    assert res_d["checks"]["digital_signature"]["status"] == "VALID"
    assert res_d["checks"]["dataset_digest"]["status"] == "MATCH"
    assert res_d["checks"]["contributor"]["status"] == "MISMATCH"
    print(f"[PASS] TEST D SUCCESSFUL: Expected '{different_contrib}' vs signed '{expected_contrib}' rejected with CONTRIBUTOR_MISMATCH.")

    # =========================================================================
    # TEST E: UNTRUSTED / WRONG KEY
    # =========================================================================
    print("\n--- TEST E: UNTRUSTED / FOREIGN KEY (VALID ED25519, UNTRUSTED ANCHOR) ---")
    foreign_priv = Ed25519PrivateKey.generate()
    foreign_pub = foreign_priv.public_key()
    foreign_pub_b64 = base64.b64encode(foreign_pub.public_bytes_raw()).decode("ascii")

    foreign_manifest = dict(manifest)
    foreign_manifest["public_key"] = foreign_pub_b64
    foreign_manifest["key_id"] = "UNTRUSTED-FOREIGN-KEY-999"
    foreign_manifest["trust_anchor_id"] = "UNTRUSTED-FOREIGN-ANCHOR"
    unsigned_foreign = {k: v for k, v in foreign_manifest.items() if k != "signature"}
    foreign_manifest["signature"] = base64.b64encode(
        foreign_priv.sign(canonicalize(unsigned_foreign))
    ).decode("ascii")

    foreign_bytes = json.dumps(foreign_manifest, indent=2).encode("utf-8")
    res_e = verify_signed_dataset_manifest(
        foreign_bytes,
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id=expected_contrib,
        expected_contribution_id=expected_contribution,
        expected_batch_id=expected_batch,
        actual_dataset_name=dataset_name,
    )
    print(f"Overall Status:    {res_e['overall']}")
    print(f"Digital Signature: {res_e['checks']['digital_signature']['status']}")
    print(f"Trusted Key:       {res_e['checks']['digital_signature']['trusted_key']}")

    assert res_e["overall"] == "MISSING_TRUSTED_KEY"
    assert res_e["checks"]["digital_signature"]["status"] == "MISSING_TRUSTED_KEY"
    assert res_e["checks"]["digital_signature"]["trusted_key"] is False
    print("[PASS] TEST E SUCCESSFUL: Foreign key rejected with MISSING_TRUSTED_KEY.")

    # =========================================================================
    # TEST F: MISSING MANIFEST
    # =========================================================================
    print("\n--- TEST F: MISSING MANIFEST (VERIFY WITHOUT MANIFEST) ---")
    res_f = verify_signed_dataset_manifest(
        b"",
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id=expected_contrib,
    )
    print(f"Overall Status: {res_f['overall']}")
    print(f"Valid:          {res_f['valid']}")
    print(f"Reason:         {res_f['reason']}")

    assert res_f["overall"] in ("FAIL", "INVALID_MANIFEST", "UNAVAILABLE")
    assert res_f["valid"] is False
    print("[PASS] TEST F SUCCESSFUL: Missing manifest rejected without passing.")

    # =========================================================================
    # TEST G: WRONG BATCH (Signed batch != Expected batch)
    # =========================================================================
    print("\n--- TEST G: WRONG BATCH BINDING ---")
    res_g = verify_signed_dataset_manifest(
        manifest_bytes,
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id=expected_contrib,
        expected_contribution_id=expected_contribution,
        expected_batch_id="BATCH-NONEXISTENT-999",
        actual_dataset_name=dataset_name,
    )
    print(f"Overall Status: {res_g['overall']}")
    print(f"Batch Status:   {res_g['checks']['batch']['status']}")
    assert res_g["overall"] == "BATCH_MISMATCH"
    assert res_g["checks"]["batch"]["status"] == "MISMATCH"
    print("[PASS] TEST G SUCCESSFUL: Mismatched batch rejected with BATCH_MISMATCH.")

    # =========================================================================
    # TEST H: MISSING PERSISTED CONTEXT (CONTRIBUTION_CONTEXT_NOT_FOUND)
    # =========================================================================
    print("\n--- TEST H: MISSING PERSISTED CONTEXT ---")
    c_db = get_contributor_backend()
    from trustcv_mirad_dataset_security import create_signed_dataset_manifest
    fake_manifest = create_signed_dataset_manifest(
        dataset_name=dataset_name,
        dataset_digest=real_dataset_sha256,
        contributor_id=expected_contrib,
        contribution_id="CNTRB-NONEXISTENT-999",
        batch_id=expected_batch,
        annotation_format="NONE",
        task_type="UNANNOTATED_IMAGE_DATASET",
    )
    fake_signed_bytes = json.dumps(fake_manifest, indent=2).encode("utf-8")
    res_h = verify_signed_dataset_manifest(
        fake_signed_bytes,
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id=expected_contrib,
        backend=c_db,
        enforce_lineage=True,
    )
    print(f"Overall Status:      {res_h['overall']}")
    print(f"Contribution Status: {res_h['checks']['contribution']['status']}")
    assert res_h["overall"] == "CONTRIBUTION_CONTEXT_NOT_FOUND"
    assert res_h["checks"]["contribution"]["status"] == "CONTRIBUTION_CONTEXT_NOT_FOUND"
    print("[PASS] TEST H SUCCESSFUL: Missing contribution rejected with CONTRIBUTION_CONTEXT_NOT_FOUND.")

    # =========================================================================
    # TEST I: PERSISTED LINEAGE INCONSISTENCY (Batch belongs to different contribution)
    # =========================================================================
    print("\n--- TEST I: PERSISTED LINEAGE INCONSISTENCY ---")
    # For CNTRB-C 4-005, the real batch in DB is BATCH-CNTRB-C 4-005-B01.
    # If the signed batch claims BATCH-CNTRB-C 4-006-B01 (or another batch from another contribution):
    inconsistent_manifest = create_signed_dataset_manifest(
        dataset_name=dataset_name,
        dataset_digest=real_dataset_sha256,
        contributor_id=expected_contrib,
        contribution_id=expected_contribution,
        batch_id="BATCH-CNTRB-C 4-006-B01",
        annotation_format="NONE",
        task_type="UNANNOTATED_IMAGE_DATASET",
    )
    inconsistent_signed_bytes = json.dumps(inconsistent_manifest, indent=2).encode("utf-8")
    res_i = verify_signed_dataset_manifest(
        inconsistent_signed_bytes,
        actual_dataset_digest=real_dataset_sha256,
        expected_contributor_id=expected_contrib,
        backend=c_db,
        enforce_lineage=True,
    )
    print(f"Overall Status: {res_i['overall']}")
    print(f"Lineage Status: {res_i['checks']['lineage']['status']}")
    assert res_i["overall"] == "PERSISTED_LINEAGE_INCONSISTENCY"
    assert res_i["checks"]["lineage"]["status"] == "PERSISTED_LINEAGE_INCONSISTENCY"
    print("[PASS] TEST I SUCCESSFUL: Inconsistent lineage rejected with PERSISTED_LINEAGE_INCONSISTENCY.")

    print("\n" + "=" * 80)
    print("ALL REAL DEMO TESTS (A through I) EXECUTED AND PASSED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    run_real_demo_suite()
