"""
Authoritative generation and independent verification of final_demo.manifest.json
for images(1).zip.
"""

import json
from pathlib import Path
from contributor_backend import get_contributor_backend
from trustcv_mirad_dataset_security import (
    create_signed_dataset_manifest,
    verify_signed_dataset_manifest,
    sha256_bytes,
    ensure_trust_anchor,
)

def run():
    dataset_path = Path("images(1).zip")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path.resolve()}")

    dataset_bytes = dataset_path.read_bytes()
    dataset_sha256 = sha256_bytes(dataset_bytes)
    print(f"Dataset Path: {dataset_path.resolve()}")
    print(f"Dataset Size: {len(dataset_bytes)} bytes")
    print(f"Dataset SHA-256: {dataset_sha256}")

    # Inspect contributor backend
    c_db = get_contributor_backend()
    registered = c_db.get_contributors()
    print(f"Registered Contributors Count: {len(registered)}")
    reg_ids = [c["contributor_id"] for c in registered]
    print(f"Registered Contributor IDs: {reg_ids}")

    # Query persistent contribution records
    persisted_contribs = c_db.get_contributions(dataset_digest=dataset_sha256)
    print(f"Persisted contributions for this dataset digest: {len(persisted_contribs)}")
    if not persisted_contribs:
        raise ValueError(f"CONTRIBUTOR_CONTEXT_UNRESOLVED: No persisted contribution records found for digest {dataset_sha256}")

    active_rec = persisted_contribs[0]
    contributor_id = active_rec["contributor_id"]
    contribution_id = active_rec["contribution_id"]
    batch_id = active_rec["batch_id"]
    print(f"Authoritative Contributor ID: {contributor_id}")
    print(f"Authoritative Contribution ID: {contribution_id}")
    print(f"Authoritative Batch ID: {batch_id}")

    if contributor_id not in reg_ids:
        raise ValueError(f"CONTRIBUTOR_CONTEXT_UNRESOLVED: Contributor {contributor_id} is not registered!")

    # Check trust anchor
    anchor = ensure_trust_anchor()
    print(f"Trust Anchor ID: {anchor['trust_anchor_id']}")
    print(f"Key ID: {anchor['key_id']}")
    print(f"Public Key: {anchor['public_key']}")

    # Annotation format & task type (honest observation from ZIP bytes: 32 images, 0 labels)
    annotation_format = "NONE / UNAVAILABLE"
    task_type = "UNANNOTATED_IMAGE_DATASET"

    # Create signed manifest using the authoritative API
    manifest = create_signed_dataset_manifest(
        dataset_name=dataset_path.name,
        dataset_digest=dataset_sha256,
        contributor_id=contributor_id,
        contribution_id=contribution_id,
        batch_id=batch_id,
        annotation_format=annotation_format,
        task_type=task_type,
    )

    out_dir = Path("demo_output/signed_manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "final_demo.manifest.json"

    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = sha256_bytes(manifest_bytes)

    print(f"\nManifest written to: {manifest_path.resolve()}")
    print(f"Manifest SHA-256: {manifest_sha256}")

    # INDEPENDENT VERIFICATION from disk bytes
    loaded_bytes = manifest_path.read_bytes()
    recalculated_dataset_sha256 = sha256_bytes(dataset_path.read_bytes())
    assert recalculated_dataset_sha256 == dataset_sha256, "Dataset digest changed on disk!"

    v_res = verify_signed_dataset_manifest(
        loaded_bytes,
        actual_dataset_digest=recalculated_dataset_sha256,
        expected_contributor_id=contributor_id,
        expected_contribution_id=contribution_id,
        expected_batch_id=batch_id,
        actual_dataset_name=dataset_path.name,
    )

    print("\n--- INDEPENDENT VERIFICATION RESULT ---")
    print(f"Overall Status: {v_res['overall']}")
    print(f"Valid: {v_res['valid']}")
    print(f"Reason: {v_res['reason']}")
    for k, v in v_res.get("checks", {}).items():
        print(f"  Check '{k}': {v.get('status')} - {v.get('reason', '')}")

    assert v_res["overall"] == "PASS", f"Verification did not PASS: {v_res}"
    print("\n[SUCCESS] Manifest independently verified as PASS!")

if __name__ == "__main__":
    run()
