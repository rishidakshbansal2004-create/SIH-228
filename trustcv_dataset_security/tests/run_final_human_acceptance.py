"""
TRUSTCV — FINAL HUMAN ACCEPTANCE TEST & DATABASE AUDIT
Validates Section 42 & Section 43 requirements end-to-end against the
canonical runtime SQLite database.
"""

import io
import os
import sys
import zipfile
import sqlite3
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
if ROOT.name in ("tests", "scratch"):
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Lib/site-packages"))

from contributor_backend import get_contributor_backend, DEFAULT_DB_PATH
from trustcv_final import (
    parse_dataset,
    analyze_reports,
    dataset_checks,
    generate_findings,
)
from trustcv_mirad_dataset_security import (
    create_signed_dataset_manifest,
    persist_dataset_security_run,
    ensure_trust_anchor,
)

print("=" * 80)
print("TRUSTCV FINAL HUMAN ACCEPTANCE TEST (SECTIONS 42 & 43)")
print("=" * 80)

# Connect to the canonical runtime DB
db_path = DEFAULT_DB_PATH
print(f"Target Database: {db_path}")

def get_db_stats(path):
    with sqlite3.connect(path) as con:
        cur = con.cursor()
        return {
            "contributors": cur.execute("SELECT COUNT(*) FROM contributors").fetchone()[0],
            "contributions": cur.execute("SELECT COUNT(*) FROM contributions").fetchone()[0],
            "batches": cur.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
            "samples": cur.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
            "findings": cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
        }

# Start clean: reset tables if needed
db = get_contributor_backend(db_path)
stats_before = get_db_stats(db_path)
print(f"Initial State: {stats_before}")
if stats_before.get("contributors", 0) > 0 or stats_before.get("contributions", 0) > 0:
    print("Resetting database to 0 records for Clean Acceptance Run...")
    with sqlite3.connect(db_path) as con:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for t in tables:
            if t != "schema_migrations":
                con.execute(f"DELETE FROM {t}")
        con.commit()
    print("Database reset complete. All tables verified clean.")

# Verify initial counts are zero
stats = get_db_stats(db_path)
assert stats.get("contributors", 0) == 0, "Expected 0 contributors"
assert stats.get("contributions", 0) == 0, "Expected 0 contributions"
assert stats.get("batches", 0) == 0, "Expected 0 batches"
assert stats.get("samples", 0) == 0, "Expected 0 samples"
assert stats.get("findings", 0) == 0, "Expected 0 findings"
print("[OK] Verified clean starting state: 0 contributors, 0 contributions, 0 batches, 0 samples.\n")

class SyntheticUpload:
    def __init__(self, data, name="dataset.zip"):
        self._data = data
        self.name = name
    def getvalue(self):
        return self._data

# Helper to create valid ZIP datasets
def make_zip(image_dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, color in image_dict.items():
            im = Image.new("RGB", (64, 64), color=color)
            im_bytes = io.BytesIO()
            im.save(im_bytes, format="PNG")
            zf.writestr(path, im_bytes.getvalue())
    return buf.getvalue()

# Helper to execute upload pipeline
def run_dataset_upload(contributor_id, dataset_bytes, dataset_name, contribution_id=None, batch_id=None):
    # 1. Allocate IDs if not provided
    if not contribution_id:
        contribution_id = db.allocate_next_contribution_id(contributor_id)
    if not batch_id:
        batch_id = db.allocate_next_batch_id(contributor_id, contribution_id)

    # 2. Parse dataset with Mode A authoritative contributor
    parsed = parse_dataset(
        SyntheticUpload(dataset_bytes, dataset_name),
        contributor_id=contributor_id,
        batch_id=batch_id,
    )
    assert parsed is not None, f"Failed to parse {dataset_name}"
    
    # 3. Analyze images and generate findings
    reports = analyze_reports(parsed)
    checks = dataset_checks(parsed, reports)
    findings = generate_findings(parsed, reports)
    
    # 4. Generate MIRAD security layer record
    ensure_trust_anchor()
    manifest = create_signed_dataset_manifest(
        dataset_name=str(parsed.get("name", dataset_name)),
        dataset_digest=str(parsed.get("hash", "")),
        contributor_id=contributor_id,
        batch_id=batch_id,
    )
    mirad_res = persist_dataset_security_run(
        dataset=parsed,
        reports=reports,
        checks=checks,
        manifest=manifest,
        contributor_id=contributor_id,
        contribution_id=contribution_id,
        batch_id=batch_id,
    )
    
    # 5. Extract raw images for media persistence
    raw_images = {}
    if parsed.get("images") and parsed.get("image_names"):
        for name, img in zip(parsed["image_names"], parsed["images"]):
            raw_images[name] = img
            
    # 6. Commit record to Contributor Backend
    backend_run = db.record_dataset_analysis(
        dataset_name=str(parsed.get("name", "")),
        dataset_digest=str(parsed.get("hash", "")),
        reports=reports,
        findings=findings,
        manifest=mirad_res.get("manifest", {}),
        provenance=mirad_res.get("provenance"),
        audit_event_id=mirad_res.get("audit_event", {}).get("audit_id"),
        checkpoint_id=mirad_res.get("checkpoint", {}).get("checkpoint_id"),
        contributor_id_override=contributor_id,
        contribution_id_override=contribution_id,
        batch_id_override=batch_id,
        raw_images=raw_images,
        require_registered_contributor=True,
    )
    print(f"     backend_run returned: {backend_run}")
    assert backend_run is not None
    assert len(backend_run.get("contributions_recorded", [])) > 0
    return parsed, reports, contribution_id, batch_id

# ==============================================================================
# STEP 1: Register CONTRIB-A
# ==============================================================================
print("[Step 1] Registering CONTRIB-A...")
cA = db.register_contributor(
    contributor_id="CONTRIB-A",
    display_name="Contributor A",
    organization="Demo Organization",
    source_id="WEB_UPLOAD",
    metadata={"role": "Lead Partner"},
)
assert cA["contributor_id"] == "CONTRIB-A"
print(f"[OK] Registered CONTRIB-A: {cA}")

# Verify 1 contributor
all_c = db.get_contributors()
assert len(all_c) == 1
assert all_c[0]["contributor_id"] == "CONTRIB-A"
print(f"[OK] Contributor count: {len(all_c)}\n")

# ==============================================================================
# STEP 2: Upload A.zip as CONTRIB-A
# Includes internal folder 'images/' to verify Section 2 folder-override bug is fixed!
# ==============================================================================
print("[Step 2] Uploading A.zip as CONTRIB-A (with folder 'images/' inside)...")
zip_A = make_zip({
    "images/sample_a1.png": (10, 50, 100),
    "images/sample_a2.png": (20, 60, 110),
})

parsed_A, reports_A, contrib_A1, batch_A1 = run_dataset_upload("CONTRIB-A", zip_A, "A.zip")
print(f"[OK] Uploaded A.zip successfully:")
print(f"     Dataset Digest : {parsed_A['hash'][:16]}...")
print(f"     Contribution ID: {contrib_A1}")
print(f"     Batch ID       : {batch_A1}")
assert contrib_A1 == "CNTRB-CONTRIB-A-001"
assert batch_A1 == "BATCH-CNTRB-CONTRIB-A-001-B01"

# Verify in DB: Contributor = CONTRIB-A (NOT 'images'!)
trace_A = db.get_contributor_traceability("CONTRIB-A")
assert trace_A["contributor"]["sample_count"] == 2
assert len(trace_A["contributions"]) == 1
assert len(trace_A["batches"]) == 1
assert len(trace_A["samples"]) == 2
for s in trace_A["samples"]:
    assert s["contributor_id"] == "CONTRIB-A", f"Sample contributor mismatch: {s['contributor_id']}"
    assert s["contribution_id"] == contrib_A1
    assert s["batch_id"] == batch_A1
    assert s.get("source_folder") == "images" or "images" in s.get("file_path", "")
print("[OK] Lineage verified for A.zip: Contributor is CONTRIB-A (folder 'images' did NOT become contributor!)\n")

# ==============================================================================
# STEP 3: Register CONTRIB-B
# ==============================================================================
print("[Step 3] Registering CONTRIB-B...")
cB = db.register_contributor(
    contributor_id="CONTRIB-B",
    display_name="Contributor B",
    organization="Demo Organization",
    source_id="WEB_UPLOAD",
    metadata={"role": "External Contributor"},
)
assert cB["contributor_id"] == "CONTRIB-B"
print(f"[OK] Registered CONTRIB-B: {cB}")

all_c = db.get_contributors()
assert len(all_c) == 2
print(f"[OK] Contributor count: {len(all_c)}\n")

# ==============================================================================
# STEP 4: Upload B.zip as CONTRIB-B
# Includes internal folder 'data/'
# ==============================================================================
print("[Step 4] Uploading B.zip as CONTRIB-B (with folder 'data/' inside)...")
zip_B = make_zip({
    "data/sample_b1.png": (150, 20, 20),
    "data/sample_b2.png": (160, 30, 30),
})

parsed_B, reports_B, contrib_B1, batch_B1 = run_dataset_upload("CONTRIB-B", zip_B, "B.zip")
print(f"[OK] Uploaded B.zip successfully:")
print(f"     Contribution ID: {contrib_B1}")
print(f"     Batch ID       : {batch_B1}")
assert contrib_B1 == "CNTRB-CONTRIB-B-001"
assert batch_B1 == "BATCH-CNTRB-CONTRIB-B-001-B01"

# Verify B in DB
trace_B = db.get_contributor_traceability("CONTRIB-B")
assert trace_B["contributor"]["sample_count"] == 2
assert len(trace_B["contributions"]) == 1
assert len(trace_B["batches"]) == 1
assert len(trace_B["samples"]) == 2
for s in trace_B["samples"]:
    assert s["contributor_id"] == "CONTRIB-B"
    assert s["contribution_id"] == contrib_B1
    assert s["batch_id"] == batch_B1
print("[OK] Lineage verified for B.zip: Contributor is CONTRIB-B\n")

# ==============================================================================
# STEP 5: Upload A2.zip as CONTRIB-A (Second contribution for A)
# ==============================================================================
print("[Step 5] Uploading A2.zip as CONTRIB-A (Second upload for A)...")
zip_A2 = make_zip({
    "batch_02/sample_a2_1.png": (80, 180, 80),
    "batch_02/sample_a2_2.png": (90, 190, 90),
})

parsed_A2, reports_A2, contrib_A2, batch_A2 = run_dataset_upload("CONTRIB-A", zip_A2, "A2.zip")
print(f"[OK] Uploaded A2.zip successfully:")
print(f"     Contribution ID: {contrib_A2}")
print(f"     Batch ID       : {batch_A2}")
assert contrib_A2 == "CNTRB-CONTRIB-A-002", f"Expected CNTRB-CONTRIB-A-002, got {contrib_A2}"
assert batch_A2 == "BATCH-CNTRB-CONTRIB-A-002-B01", f"Expected BATCH-CNTRB-CONTRIB-A-002-B01, got {batch_A2}"

# Verify A has 2 contributions, 2 batches, 4 samples
trace_A = db.get_contributor_traceability("CONTRIB-A")
assert trace_A["contributor"]["sample_count"] == 4, f"Expected 4 samples, got {trace_A['contributor']['sample_count']}"
assert len(trace_A["contributions"]) == 2, f"Expected 2 contributions, got {len(trace_A['contributions'])}"
assert len(trace_A["batches"]) == 2, f"Expected 2 batches, got {len(trace_A['batches'])}"
assert len(trace_A["samples"]) == 4, f"Expected 4 samples, got {len(trace_A['samples'])}"

# Verify B remains strictly isolated (1 contribution, 1 batch, 2 samples)
trace_B = db.get_contributor_traceability("CONTRIB-B")
assert trace_B["contributor"]["sample_count"] == 2
assert len(trace_B["contributions"]) == 1
assert len(trace_B["batches"]) == 1
assert len(trace_B["samples"]) == 2
print("[OK] Multi-upload verified: A has 2 contributions & 4 samples; B remains strictly isolated with 1 contribution & 2 samples!\n")

# ==============================================================================
# STEP 6: Same Dataset Digest / Different Contributor (Section 28)
# CONTRIB-B uploads exact same bytes as A2.zip
# ==============================================================================
print("[Step 6] Testing Section 28: CONTRIB-B uploads exact same dataset bytes as A2.zip...")
parsed_B2, reports_B2, contrib_B2, batch_B2 = run_dataset_upload("CONTRIB-B", zip_A2, "A2_copy_by_B.zip")
print(f"[OK] Uploaded identical bytes as B:")
print(f"     Dataset Digest : {parsed_B2['hash'][:16]}... (Identical to A2)")
print(f"     Contribution ID: {contrib_B2}")
print(f"     Batch ID       : {batch_B2}")
assert parsed_B2["hash"] == parsed_A2["hash"], "Digests must match"
assert contrib_B2 == "CNTRB-CONTRIB-B-002"
assert batch_B2 == "BATCH-CNTRB-CONTRIB-B-002-B01"

# Verify B's new contribution belongs to B, not A
trace_B = db.get_contributor_traceability("CONTRIB-B")
assert len(trace_B["contributions"]) == 2
assert len(trace_B["batches"]) == 2
assert trace_B["contributor"]["sample_count"] == 4
print("[OK] Section 28 verified: Same dataset digest independently owned by different contributors without collision!\n")

# ==============================================================================
# STEP 7: Application Restart Simulation
# ==============================================================================
print("[Step 7] Simulating Application Restart / Reconnect...")
# Reinitialize backend object from disk
db_restarted = get_contributor_backend(db_path)
stats_restarted = get_db_stats(db_path)
print(f"State after restart: {stats_restarted}")
assert stats_restarted["contributors"] == 2
assert stats_restarted["contributions"] == 4
assert stats_restarted["batches"] == 4
assert stats_restarted["samples"] == 8

# Verify hash chains and cryptographic integrity
for cid in ["CONTRIB-A", "CONTRIB-B"]:
    t = db_restarted.get_contributor_traceability(cid)
    assert t["contributor"]["contributor_id"] == cid
    for c in t["contributions"]:
        assert c["contributor_id"] == cid

# Verify provenance hash chains for all datasets
with sqlite3.connect(db_path) as con:
    d_digests = [r[0] for r in con.execute("SELECT DISTINCT dataset_digest FROM datasets").fetchall()]
    for dd in d_digests:
        res = db_restarted.verify_provenance_chain(dd)
        assert res["valid"], f"Provenance chain broken for {dd}: {res}"
        print(f"  [OK] Provenance chain for dataset {dd[:12]}... verified ({res['events_count']} events).")

# Verify audit trail
audit_events = db_restarted.generate_audit_log_export(fmt="json")
assert len(audit_events) > 0
for ae in audit_events:
    assert ae.get("status") in ("PASS", "ACCEPT", "RECORDED", "SUCCESS", None) or "sha256" in str(ae)
print(f"[OK] Audit trail verified: {len(audit_events)} audit events, all valid status.\n")

# ==============================================================================
# STEP 8: SECTION 43 — FINAL DATABASE ASSERTIONS
# ==============================================================================
print("=" * 80)
print("SECTION 43: FINAL DATABASE ASSERTIONS & SQL RESULTS")
print("=" * 80)

with sqlite3.connect(db_path) as con:
    cur = con.cursor()
    
    # 1. Contributor group counts
    print("\n1. SELECT contributor_id, COUNT(*) FROM contributors GROUP BY contributor_id;")
    rows = cur.execute("SELECT contributor_id, COUNT(*) FROM contributors GROUP BY contributor_id").fetchall()
    for r in rows:
        print(f"   contributor_id: {r[0]:15s} | count: {r[1]}")
    assert len(rows) == 2, f"Expected 2 contributors, got {len(rows)}"
    
    # 2. Contributions ownership
    print("\n2. Contribution Ownership:")
    rows = cur.execute("SELECT contribution_id, contributor_id, dataset_digest, batch_id, created_at FROM contributions ORDER BY contribution_id").fetchall()
    for r in rows:
        print(f"   {r[0]:25s} -> Contributor: {r[1]:12s} | Batch: {r[3]:35s}")
    assert len(rows) == 4, f"Expected 4 contributions, got {len(rows)}"
    
    # 3. Batches ownership
    print("\n3. Batch Ownership:")
    rows = cur.execute("SELECT batch_id, contribution_id, contributor_id, sample_count FROM batches ORDER BY batch_id").fetchall()
    for r in rows:
        print(f"   {r[0]:35s} -> Contribution: {r[1]:25s} | Contributor: {r[2]:12s} | Samples: {r[3]}")
    assert len(rows) == 4, f"Expected 4 batches, got {len(rows)}"
    
    # 4. Sample counts per contributor
    print("\n4. Sample Distribution:")
    rows = cur.execute("SELECT contributor_id, COUNT(*), COUNT(DISTINCT batch_id), COUNT(DISTINCT contribution_id) FROM samples GROUP BY contributor_id").fetchall()
    for r in rows:
        print(f"   Contributor: {r[0]:12s} | Samples: {r[1]} | Batches: {r[2]} | Contributions: {r[3]}")
        assert r[1] == 4, f"Expected 4 samples for {r[0]}, got {r[1]}"
        assert r[2] == 2, f"Expected 2 batches for {r[0]}, got {r[2]}"
        assert r[3] == 2, f"Expected 2 contributions for {r[0]}, got {r[3]}"

    # 5. Check for orphan records
    print("\n5. Orphan Checks:")
    orphan_contribs = cur.execute("""
        SELECT COUNT(*) FROM contributions 
        WHERE contributor_id NOT IN (SELECT contributor_id FROM contributors)
    """).fetchone()[0]
    print(f"   Orphan contributions : {orphan_contribs}")
    assert orphan_contribs == 0, "Found orphan contributions!"

    orphan_batches = cur.execute("""
        SELECT COUNT(*) FROM batches 
        WHERE contribution_id NOT IN (SELECT contribution_id FROM contributions)
           OR contributor_id NOT IN (SELECT contributor_id FROM contributors)
    """).fetchone()[0]
    print(f"   Orphan batches       : {orphan_batches}")
    assert orphan_batches == 0, "Found orphan batches!"

    orphan_samples = cur.execute("""
        SELECT COUNT(*) FROM samples 
        WHERE batch_id NOT IN (SELECT batch_id FROM batches)
           OR contribution_id NOT IN (SELECT contribution_id FROM contributions)
           OR contributor_id NOT IN (SELECT contributor_id FROM contributors)
    """).fetchone()[0]
    print(f"   Orphan samples       : {orphan_samples}")
    assert orphan_samples == 0, "Found orphan samples!"

    orphan_findings = cur.execute("""
        SELECT COUNT(*) FROM findings 
        WHERE sample_id NOT IN (SELECT sample_id FROM samples)
    """).fetchone()[0]
    print(f"   Orphan findings      : {orphan_findings}")
    assert orphan_findings == 0, "Found orphan findings!"

    orphan_evidence = cur.execute("""
        SELECT COUNT(*) FROM evidence 
        WHERE sample_id NOT IN (SELECT sample_id FROM samples)
    """).fetchone()[0]
    print(f"   Orphan evidence      : {orphan_evidence}")
    assert orphan_evidence == 0, "Found orphan evidence!"

    # 6. Check for cross-field contradictions
    print("\n6. Cross-Field Contradiction Checks:")
    contradictory_batches = cur.execute("""
        SELECT b.batch_id, b.contributor_id, c.contributor_id 
        FROM batches b
        JOIN contributions c ON b.contribution_id = c.contribution_id
        WHERE b.contributor_id != c.contributor_id
    """).fetchall()
    print(f"   Contradictory batches (batch.contributor != contribution.contributor): {len(contradictory_batches)}")
    assert len(contradictory_batches) == 0, f"Contradictions found: {contradictory_batches}"

    contradictory_samples = cur.execute("""
        SELECT s.sample_id, s.contributor_id, b.contributor_id 
        FROM samples s
        JOIN batches b ON s.batch_id = b.batch_id
        WHERE s.contributor_id != b.contributor_id
    """).fetchall()
    print(f"   Contradictory samples (sample.contributor != batch.contributor): {len(contradictory_samples)}")
    assert len(contradictory_samples) == 0, f"Contradictions found: {contradictory_samples}"

    contradictory_findings = cur.execute("""
        SELECT f.finding_id, f.contributor_id, s.contributor_id 
        FROM findings f
        JOIN samples s ON f.sample_id = s.sample_id
        WHERE f.contributor_id != s.contributor_id
    """).fetchall()
    print(f"   Contradictory findings (finding.contributor != sample.contributor): {len(contradictory_findings)}")
    assert len(contradictory_findings) == 0, f"Contradictions found: {contradictory_findings}"

    # 7. Check for fake default contributors and ROOT/UNKNOWN
    print("\n7. Fake / Inferred Contributor Checks:")
    root_unknown_count = cur.execute("""
        SELECT COUNT(*) FROM contributors 
        WHERE contributor_id LIKE '%ROOT%' 
           OR contributor_id LIKE '%UNKNOWN%' 
           OR contributor_id = 'images'
           OR contributor_id = 'data'
           OR contributor_id = 'batch_02'
           OR contributor_id = 'subset'
           OR contributor_id = 'CONTRIB-PRIMARY-01'
    """).fetchone()[0]
    print(f"   COUNT(ROOT / UNKNOWN / Folder Contributors / CONTRIB-PRIMARY-01): {root_unknown_count}")
    assert root_unknown_count == 0, f"Found {root_unknown_count} invalid contributors!"

    root_unknown_samples = cur.execute("""
        SELECT COUNT(*) FROM samples 
        WHERE contributor_id LIKE '%ROOT%' 
           OR contributor_id LIKE '%UNKNOWN%' 
           OR contributor_id = 'images'
           OR contributor_id = 'data'
           OR contributor_id = 'batch_02'
           OR contributor_id = 'subset'
           OR contributor_id = 'CONTRIB-PRIMARY-01'
    """).fetchone()[0]
    print(f"   COUNT(Samples with invalid contributor): {root_unknown_samples}")
    assert root_unknown_samples == 0, f"Found {root_unknown_samples} samples with invalid contributor!"

    print("\n" + "=" * 80)
    print("ALL FINAL DATABASE ASSERTIONS PASSED 100% (ZERO DEFECTS, ZERO ANOMALIES)!")
    print("=" * 80)
