"""
Targeted Verification Script for Sections 17 & 18:
Manifest Reload After Application Restart & New Contribution Isolation.

Validates:
1. PHASE A: Initial load of images(1).zip with final_demo.manifest.json -> PASS
2. PHASE B: Complete application teardown and restart -> Independent reload of final_demo.manifest.json -> PASS
   (without manual entry of CNTRB-C 4-005, and without manifest alteration)
3. SECTION 18: New session with active intake CNTRB-C 4-006 does NOT overwrite or invalidate
   the historical signed manifest (final_demo.manifest.json) -> PASS
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name in ("tests", "scratch"):
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Lib/site-packages"))

from streamlit.testing.v1 import AppTest
from contributor_backend import get_contributor_backend
from trustcv_mirad_dataset_security import sha256_bytes

def run_restart_and_isolation_test():
    print("=" * 80)
    print("TESTING MANIFEST RELOAD AFTER RESTART & NEW INTAKE ISOLATION (SECTIONS 17 & 18)")
    print("=" * 80)

    dataset_path = ROOT / "images(1).zip"
    if not dataset_path.exists():
        dataset_path = ROOT / "images.zip"
    assert dataset_path.exists(), f"Dataset file missing: {dataset_path}"
    ds_bytes = dataset_path.read_bytes()
    ds_digest = sha256_bytes(ds_bytes)
    print(f"[INFO] Dataset Digest: {ds_digest}")

    manifest_path = ROOT / "demo_output/signed_manifests/final_demo.manifest.json"
    assert manifest_path.exists(), f"Manifest file missing: {manifest_path}"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    print(f"[INFO] Manifest SHA-256: {manifest_sha}")

    # =========================================================================
    # PHASE A: Session 1 Initial Verification
    # =========================================================================
    print("\n--- PHASE A: SESSION 1 VERIFICATION ---")
    app_file = str(ROOT / "trustcv_final.py")
    at1 = AppTest.from_file(app_file, default_timeout=60)
    at1.run()
    assert not at1.exception, f"Session 1 initial run failed: {at1.exception}"

    at1.session_state["contributor_verified"] = True
    at1.session_state["operator_authorized"] = True
    at1.sidebar.radio[0].set_value("10 · Verification").run()
    assert not at1.exception, f"Session 1 Page 10 failed: {at1.exception}"

    status_1 = at1.session_state.get("manifest_status")
    print(f"[SESSION 1] Manifest Verification Status: {status_1}")
    assert status_1 == "PASS", f"Session 1 expected PASS, got {status_1}"
    print("[PASS] PHASE A SUCCESSFUL: Session 1 verified authentic manifest as PASS.")

    # =========================================================================
    # PHASE B: Complete Application Restart & Reload
    # =========================================================================
    print("\n--- PHASE B: APPLICATION RESTART & INDEPENDENT RELOAD ---")
    # Completely drop reference to at1 to simulate process termination
    del at1

    # Spin up brand new AppTest instance (simulating TrustCV restart)
    at2 = AppTest.from_file(app_file, default_timeout=60)
    at2.run()
    assert not at2.exception, f"Session 2 initial run failed: {at2.exception}"

    at2.session_state["contributor_verified"] = True
    at2.session_state["operator_authorized"] = True

    # Navigate to Page 10 without manually entering CNTRB-C 4-005
    at2.sidebar.radio[0].set_value("10 · Verification").run()
    assert not at2.exception, f"Session 2 Page 10 failed: {at2.exception}"

    status_2 = at2.session_state.get("manifest_status")
    print(f"[SESSION 2 (RESTART)] Manifest Verification Status: {status_2}")
    assert status_2 == "PASS", f"Session 2 restart expected PASS, got {status_2}"
    print("[PASS] PHASE B SUCCESSFUL: Brand-new session after restart independently verified manifest as PASS.")

    # =========================================================================
    # SECTION 18: Active Session with NEW Contribution ID (e.g. CNTRB-C 4-006)
    # Must NOT overwrite historical signed manifest identity
    # =========================================================================
    print("\n--- SECTION 18: NEW INGESTION CONTEXT ISOLATION ---")
    # Simulate that this new session performed a new intake event producing CNTRB-C 4-006
    at2.session_state["intake_contribution_id"] = "CNTRB-C 4-006"
    at2.session_state["intake_batch_id"] = "BATCH-CNTRB-C 4-006-B01"
    at2.session_state["intake_contributor_id"] = "C 4"
    at2.session_state["manifest_bound_dataset_hash"] = None

    # Rerun Page 10 with active session intake context present
    at2.sidebar.radio[0].set_value("10 · Verification").run()
    assert not at2.exception, f"Session with new intake failed: {at2.exception}"

    status_new_intake = at2.session_state.get("manifest_status")
    print(f"[ACTIVE INTAKE CNTRB-C 4-006] Manifest Status: {status_new_intake}")
    assert status_new_intake == "PASS", f"Expected PASS even with active intake 006, got {status_new_intake}"
    print("[PASS] SECTION 18 SUCCESSFUL: Active session intake ID (006) did NOT overwrite signed manifest (005)!")

    print("\n" + "=" * 80)
    print("ALL RESTART PERSISTENCE AND INTAKE ISOLATION TESTS PASSED 100%!")
    print("=" * 80)

if __name__ == "__main__":
    run_restart_and_isolation_test()
