"""
Streamlit AppTest for Page 10 Verification.
Tests on-disk manifest loading, PASS verification, tamper simulations, and clean restore.
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name in ("tests", "scratch"):
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Lib/site-packages"))

from streamlit.testing.v1 import AppTest

def test_page_10_verification_apptest():
    print("=" * 80)
    print("RUNNING PAGE 10 APPTEST: MANIFEST VERIFICATION & TAMPER SIMULATION")
    print("=" * 80)

    app_file = str(ROOT / "trustcv_final.py")
    at = AppTest.from_file(app_file, default_timeout=60)
    at.run()
    assert not at.exception, f"Initial run exception: {at.exception}"

    # 1. Authorize operator
    at.session_state["contributor_verified"] = True
    at.session_state["operator_authorized"] = True

    # 2. Navigate to Page 10 · Verification
    print("Navigating to 10 · Verification...")
    at.sidebar.radio[0].set_value("10 · Verification").run()
    assert not at.exception, f"Page 10 load exception: {at.exception}"
    print("[OK] Page 10 rendered cleanly without exception!")

    # Check manifest status in session state
    status = at.session_state.get("manifest_status")
    print(f"Page 10 Manifest Status: {status}")
    assert status == "PASS", f"Expected PASS on initial clean load, got {status}"
    print("[OK] Initial load from final_demo.manifest.json verified as PASS!")

    # 3. Simulate Tampered Dataset
    print("\nSimulating tampered dataset...")
    at.button(key="btn_sim_tamper_ds").click().run()
    assert not at.exception, f"Tamper dataset exception: {at.exception}"
    tamper_status = at.session_state.get("manifest_status")
    print(f"Status after dataset tamper simulation: {tamper_status}")
    assert tamper_status == "DATASET_DIGEST_MISMATCH"
    print("[OK] Tampered dataset correctly yielded DATASET_DIGEST_MISMATCH!")

    # 4. Restore Clean State
    print("\nRestoring clean state...")
    at.button(key="btn_restore_clean").click().run()
    assert not at.exception, f"Restore exception: {at.exception}"
    restored_status = at.session_state.get("manifest_status")
    print(f"Status after clean restore: {restored_status}")
    assert restored_status == "PASS"
    print("[OK] Clean restore restored PASS status!")

    # 5. Simulate Corrupted Signature
    print("\nSimulating corrupted signature...")
    at.button(key="btn_sim_tamper_sig").click().run()
    assert not at.exception, f"Tamper sig exception: {at.exception}"
    sig_tamper_status = at.session_state.get("manifest_status")
    print(f"Status after signature tamper simulation: {sig_tamper_status}")
    assert sig_tamper_status == "INVALID_SIGNATURE"
    print("[OK] Corrupted signature correctly yielded INVALID_SIGNATURE!")

    # 6. Restore Clean State again
    print("\nRestoring clean state after signature tamper...")
    at.button(key="btn_restore_clean").click().run()
    assert not at.exception, f"Restore exception: {at.exception}"
    assert at.session_state.get("manifest_status") == "PASS"
    print("[OK] Clean restore restored PASS status!")

    # 7. Simulate Contributor Override (Wrong Contributor)
    print("\nSimulating contributor override (wrong contributor)...")
    at.text_input(key="ctrl_override_contrib").input("CONTRIB-WRONG-USER").run()
    at.button(key="btn_apply_contrib_override").click().run()
    assert not at.exception, f"Contributor override exception: {at.exception}"
    contrib_mismatch_status = at.session_state.get("manifest_status")
    print(f"Status after contributor override: {contrib_mismatch_status}")
    assert contrib_mismatch_status == "CONTRIBUTOR_MISMATCH"
    print("[OK] Contributor override correctly yielded CONTRIBUTOR_MISMATCH!")

    # Restore clean state after contributor override
    print("\nRestoring clean state after contributor override...")
    at.button(key="btn_restore_clean").click().run()
    assert not at.exception, f"Restore exception: {at.exception}"
    assert at.session_state.get("manifest_status") == "PASS"
    print("[OK] Clean restore restored PASS status!")

    # 8. Simulate Missing Context (CONTRIBUTION_CONTEXT_NOT_FOUND)
    print("\nSimulating missing persisted context...")
    at.button(key="btn_sim_missing_context").click().run()
    assert not at.exception, f"Missing context exception: {at.exception}"
    missing_ctx_status = at.session_state.get("manifest_status")
    print(f"Status after missing context simulation: {missing_ctx_status}")
    assert missing_ctx_status == "CONTRIBUTION_CONTEXT_NOT_FOUND"
    print("[OK] Missing context correctly yielded CONTRIBUTION_CONTEXT_NOT_FOUND!")

    # 9. Restore Clean State
    print("\nRestoring clean state after missing context...")
    at.button(key="btn_restore_clean").click().run()
    assert not at.exception, f"Restore exception: {at.exception}"
    assert at.session_state.get("manifest_status") == "PASS"
    print("[OK] Clean restore restored PASS status!")

    # 10. Simulate Lineage Inconsistency (PERSISTED_LINEAGE_INCONSISTENCY)
    print("\nSimulating persisted lineage inconsistency...")
    at.button(key="btn_sim_lineage_inconsistency").click().run()
    assert not at.exception, f"Lineage inconsistency exception: {at.exception}"
    inconsistency_status = at.session_state.get("manifest_status")
    print(f"Status after lineage inconsistency simulation: {inconsistency_status}")
    assert inconsistency_status == "PERSISTED_LINEAGE_INCONSISTENCY"
    print("[OK] Lineage inconsistency correctly yielded PERSISTED_LINEAGE_INCONSISTENCY!")

    # 11. Final clean restore
    print("\nFinal clean restore...")
    at.button(key="btn_restore_clean").click().run()
    assert not at.exception, f"Final restore exception: {at.exception}"
    assert at.session_state.get("manifest_status") == "PASS"
    print("[OK] Final clean state restored to PASS!")

    print("\n" + "=" * 80)
    print("ALL PAGE 10 APPTEST VERIFICATIONS AND TAMPER INTERACTIONS PASSED 100%!")
    print("=" * 80)

if __name__ == "__main__":
    test_page_10_verification_apptest()
