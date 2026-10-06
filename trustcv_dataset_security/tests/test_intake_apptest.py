import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name in ("tests", "scratch"):
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Lib/site-packages"))

from streamlit.testing.v1 import AppTest

print("Initializing AppTest for trustcv_final.py...")
app_file = str(ROOT / "trustcv_final.py")
at = AppTest.from_file(app_file, default_timeout=30)
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
    assert new_contrib_id == f"CNTRB-{new_c_val}-001"
    assert new_batch_id == f"BATCH-{new_c_val}-B01"

# 6. Navigate to another page and return
print("\nTesting page navigation away and return...")
at.sidebar.radio[0].set_value("05 · Contributors").run()
assert not at.exception, f"Page 05 exception: {at.exception}"
print("[OK] Rendered 05 · Contributors cleanly.")

at.sidebar.radio[0].set_value("01 · Dataset Intake").run()
assert not at.exception, f"Page 01 return exception: {at.exception}"
print("[OK] Returned to 01 · Dataset Intake cleanly without exception!")

# 7. Final rerun test
print("\nTesting rerun behavior...")
at.run()
assert not at.exception, f"Rerun exception: {at.exception}"
print("[OK] Rerun clean, zero exceptions.")

print("\n" + "=" * 70)
print("ALL INTAKE WIDGET LIFECYCLE TESTS PASSED 100% (ZERO EXCEPTIONS)!")
print("=" * 70)
