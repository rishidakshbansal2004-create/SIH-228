import io
import os
import sys
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
if ROOT.name in ("tests", "scratch"):
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Lib/site-packages"))

from streamlit.testing.v1 import AppTest
from contributor_backend import get_contributor_backend

print("=" * 70)
print("TESTING REAL DATASET INTAKE UPLOAD WORKFLOW VIA APPTEST")
print("=" * 70)

# Create a small valid test ZIP dataset with 2 images
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    # Image 1
    im1 = Image.new("RGB", (64, 64), color=(100, 150, 200))
    im1_bytes = io.BytesIO()
    im1.save(im1_bytes, format="PNG")
    zf.writestr("test_sample_01.png", im1_bytes.getvalue())

    # Image 2
    im2 = Image.new("RGB", (64, 64), color=(200, 100, 50))
    im2_bytes = io.BytesIO()
    im2.save(im2_bytes, format="PNG")
    zf.writestr("test_sample_02.png", im2_bytes.getvalue())

zip_bytes = zip_buffer.getvalue()
print(f"Generated test dataset ZIP in memory ({len(zip_bytes)} bytes, 2 PNG images).")

app_file = str(ROOT / "trustcv_final.py")
at = AppTest.from_file(app_file, default_timeout=45)
at.run()
assert not at.exception, f"Startup exception: {at.exception}"

# Operator authorization
at.session_state["operator_authorized"] = True
at.session_state["contributor_verified"] = True

# Navigate to 01 · Dataset Intake
print("\n[Step 1] Navigating to 01 · Dataset Intake...")
at.sidebar.radio[0].set_value("01 · Dataset Intake").run()
assert not at.exception, f"Intake page load exception: {at.exception}"

# Check available contributors
options = at.selectbox(key="intake_contributor_selector").options
print(f"Available contributors: {options}")

# Select first non-ROOT contributor or register one if needed
target_contrib = None
for opt in options:
    if "ROOT" not in opt and "UNKNOWN" not in opt:
        target_contrib = opt
        break
if not target_contrib:
    target_contrib = options[0]

print(f"Selecting contributor: {target_contrib}")
at.selectbox(key="intake_contributor_selector").select(target_contrib).run()
assert not at.exception

selected_cid = at.selectbox(key="intake_contributor_selector").value
print(f"Active contributor value: {selected_cid}")

# Set custom Contribution ID and Batch ID
custom_contrib_id = f"CNTRB-{selected_cid}-INTAKE-TEST"
custom_batch_id = f"BATCH-{selected_cid}-BT-99"

at.text_input(key="intake_contribution_id").input(custom_contrib_id).run()
assert not at.exception
at.text_input(key="intake_batch_id").input(custom_batch_id).run()
assert not at.exception

print(f"Configured intake context: Contributor={selected_cid}, Contribution={custom_contrib_id}, Batch={custom_batch_id}")

# Upload dataset using file uploader
print("\n[Step 2] Uploading dataset ZIP...")
at.file_uploader(key="DATASET_GLOBAL").upload("live_intake_test.zip", zip_bytes, "application/zip").run()
assert not at.exception, f"Exception during dataset ingestion: {at.exception}"
print("[OK] Dataset ingested cleanly via FileUploader without exception!")

# Verify dataset in session state
dataset = at.session_state.get("dataset")
assert dataset is not None, "Dataset should be parsed in session_state"
print(f"[OK] Ingested dataset kind: {dataset.get('kind')}, name: {dataset.get('name')}, digest: {dataset.get('hash')[:16]}...")
print(f"     Image count: {len(dataset.get('images', []))}")
assert len(dataset.get("images", [])) == 2

# Verify reports and checks
reports = at.session_state.get("dataset_reports", [])
print(f"[OK] Generated {len(reports)} image security reports.")
assert len(reports) == 2

# Verify MIRAD security layer record
mirad_rec = at.session_state.get("mirad_security_record")
assert mirad_rec is not None, "MIRAD security record should exist"
print(f"[OK] MIRAD security record created: trusted={mirad_rec.get('trusted')}, dataset_only={mirad_rec.get('dataset_only')}")

# Verify SQLite Contributor Backend persistence
print("\n[Step 3] Verifying SQLite Contributor Backend Persistence...")
db = get_contributor_backend()
trace = db.get_contributor_traceability(selected_cid)
print(f"Contributor {selected_cid} stats:")
print(f"  Lifetime samples: {trace['contributor']['sample_count']}")
print(f"  Batches         : {len(trace['batches'])}")
print(f"  Samples recorded: {len(trace['samples'])}")

assert trace['contributor']['sample_count'] >= 2, "Lifetime samples must be >= 2"
assert len(trace['samples']) >= 2, "At least 2 samples must be linked"

# Verify that the samples in the database are bound to our custom batch and contribution
matching_samples = [s for s in trace['samples'] if s.get('batch_id') == custom_batch_id]
print(f"  Samples bound to batch {custom_batch_id}: {len(matching_samples)}")
assert len(matching_samples) >= 2, f"Expected at least 2 samples bound to {custom_batch_id}"

# Step 4: Navigate to other core pages to ensure clean rendering with real ingested data
pages_to_test = [
    "02 · Dataset Profile",
    "03 · Integrity Analysis",
    "04 · Distribution & Spectral",
    "05 · Contributors",
    "06 · Batches",
    "07 · Findings",
    "08 · Evidence Explorer",
    "09 · Dataset Provenance",
    "10 · Verification",
    "11 · Audit Trail",
]

print("\n[Step 4] Smoke testing all pages with live ingested dataset...")
for page in pages_to_test:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"Exception on {page}: {at.exception}"
    print(f"  [OK] Rendered {page}")

print("\n" + "=" * 70)
print("REAL END-TO-END INTAKE & PERSISTENCE WORKFLOW PASSED 100%!")
print("=" * 70)
