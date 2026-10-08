import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name in ("tests", "scratch"):
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Lib/site-packages"))

# Isolate database for AppTest so production database is never contaminated
tmp_dir = tempfile.TemporaryDirectory()
test_db = Path(tmp_dir.name) / "test_intake_apptest.db"
os.environ["TRUSTCV_DB_PATH"] = str(test_db)

from contributor_backend import get_contributor_backend
c_db = get_contributor_backend(test_db)
c_db.register_contributor("CONTRIB-DEMO-01", display_name="Primary Contributor", organization="Test Org", source_id="PORTAL")
c_db.register_contributor("CONTRIB-DEMO-02", display_name="Secondary Contributor", organization="Test Org", source_id="PORTAL")

from streamlit.testing.v1 import AppTest

print("Initializing AppTest for trustcv_final.py...")
app_file = str(ROOT / "trustcv_final.py")
at = AppTest.from_file(app_file, default_timeout=60)
at.run()
assert not at.exception, f"Initial run exception: {at.exception}"
print("[OK] Initial app launch clean.")

# 1. Authorize Operator via session state (or verified gate)
print("Setting operator authorization...")
at.session_state["contributor_verified"] = True
at.session_state["operator_authorized"] = True

# 2. Navigate to Page 01 · Dataset Intake
print("Navigating to 01 · Dataset Intake...")
at.sidebar.radio[0].set_value("01 · Dataset Intake").run()
assert not at.exception, f"Intake page load exception: {at.exception}"
print("[OK] 01 · Dataset Intake rendered without exception!")

# Check widget presence
contrib_sel = at.selectbox(key="intake_contributor_selector")
contrib_id = at.text_input(key="intake_contribution_id")
batch_id = at.text_input(key="intake_batch_id")
print(f"Current values: Contributor={contrib_sel.value}, ContributionID={contrib_id.value}, BatchID={batch_id.value}")

# 3. Change Contribution ID
print("\nTesting user input: changing Contribution ID...")
at.text_input(key="intake_contribution_id").input("CNTRB-CUSTOM-TEST-007").run()
assert not at.exception, f"Contribution ID change exception: {at.exception}"
print(f"[OK] Contribution ID updated to: {at.text_input(key='intake_contribution_id').value}")
assert at.session_state["intake_contribution_id"] == "CNTRB-CUSTOM-TEST-007"

# 4. Change Batch ID
print("\nTesting user input: changing Batch ID...")
at.text_input(key="intake_batch_id").input("BATCH-CUSTOM-B99").run()
assert not at.exception, f"Batch ID change exception: {at.exception}"
print(f"[OK] Batch ID updated to: {at.text_input(key='intake_batch_id').value}")
assert at.session_state["intake_batch_id"] == "BATCH-CUSTOM-B99"

# 5. Change Contributor via Selectbox
print("\nTesting contributor change via selectbox...")
options = at.selectbox(key="intake_contributor_selector").options
print(f"Available contributor options: {options}")
if len(options) > 1:
    target_c = options[1]
    at.selectbox(key="intake_contributor_selector").select(target_c).run()
    assert not at.exception, f"Contributor change exception: {at.exception}"
    new_c_val = at.selectbox(key="intake_contributor_selector").value
    new_contrib_id = at.text_input(key="intake_contribution_id").value
    new_batch_id = at.text_input(key="intake_batch_id").value
    print(f"[OK] Contributor changed to: {new_c_val}")
    print(f"     ContributionID auto-synced to: {new_contrib_id}")
    print(f"     BatchID auto-synced to: {new_batch_id}")
    assert new_c_val in target_c
    assert new_contrib_id.startswith(f"CNTRB-{new_c_val}-")
    assert new_batch_id.startswith("BATCH-") and new_c_val in new_batch_id and "B01" in new_batch_id

# 6. Navigate to another page and return
print("\nTesting page navigation away and return...")
at.sidebar.radio[0].set_value("05 · Contributors").run()
assert not at.exception, f"Page 05 exception: {at.exception}"
print("[OK] Rendered 05 · Contributors cleanly.")

at.sidebar.radio[0].set_value("01 · Dataset Intake").run()
assert not at.exception, f"Page 01 return exception: {at.exception}"
print("[OK] Returned to 01 · Dataset Intake cleanly without exception!")

# 6b. Test Inline Contributor Enrollment in Dataset Intake
print("\nTesting inline contributor enrollment in Dataset Intake...")
at.text_input(key="intake_new_cid").input("CONTRIB-INLINE-E2E").run()
at.text_input(key="intake_new_name").input("Inline Test Contributor").run()
at.button(key="btn_intake_enroll").click().run()
assert not at.exception, f"Inline enrollment exception: {at.exception}"
print("[OK] Inline contributor enrolled without exception!")
assert at.session_state["intake_contributor_selector"] == "CONTRIB-INLINE-E2E"
assert at.session_state["intake_contribution_id"] == "CNTRB-CONTRIB-INLINE-E2E-001"
assert at.session_state["intake_batch_id"].startswith("BATCH-") and "CONTRIB-INLINE-E2E" in at.session_state["intake_batch_id"] and "B01" in at.session_state["intake_batch_id"]
print("[OK] Session state and context bindings verified for new inline contributor!")

# 7. Final rerun test
print("\nTesting rerun behavior...")
at.run()
assert not at.exception, f"Rerun exception: {at.exception}"
print("[OK] Rerun clean, zero exceptions.")

print("\n" + "=" * 70)
print("ALL INTAKE WIDGET LIFECYCLE TESTS PASSED 100% (ZERO EXCEPTIONS)!")
print("=" * 70)
