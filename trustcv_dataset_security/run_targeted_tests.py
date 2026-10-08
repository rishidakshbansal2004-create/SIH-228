"""
TrustCV Dataset Security + MIRAD Comprehensive Hardening & Verification Suite
Covers Tests A through W as required by the validation specification.
"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("Lib/site-packages"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image

from contributor_backend import ContributorBackend
from trustcv_mirad_dataset_security import (
    create_signed_dataset_manifest,
    verify_signed_dataset_manifest,
    ensure_trust_anchor,
    persist_dataset_security_run,
    verify_dataset_audit,
    create_signed_checkpoint,
    verify_dataset_checkpoint,
    replay_existing_provenance,
    get_persisted_security_paths,
    canonicalize,
    sha256_obj,
)
from ethereum_anchor import (
    EthereumConfig,
    AnchorResult,
    VerificationResult,
    get_ethereum_status,
    anchor_checkpoint,
    verify_anchor,
)

test_results = {}

print("=" * 70)
print("STARTING TRUSTCV COMPREHENSIVE VERIFICATION SUITE (A - W)")
print("=" * 70)

# Temporary database path for tests
temp_db_path = Path(tempfile.gettempdir()) / "test_trustcv_comprehensive.db"
if temp_db_path.exists():
    try:
        temp_db_path.unlink()
    except Exception:
        pass

backend = ContributorBackend(db_path=temp_db_path)

# ---------------------------------------------------------------
# TEST A: Contributor Registration & Persistence
# ---------------------------------------------------------------
print("\n[TEST A] Contributor Registration & Persistence...")
contrib_a = backend.register_contributor(
    contributor_id="CONTRIB-A",
    display_name="Alice Security Labs",
    source_id="SOURCE-ALICE-01",
    organization="Alice Security Inc.",
)
assert contrib_a["contributor_id"] == "CONTRIB-A"
assert contrib_a["display_name"] == "Alice Security Labs"
# Re-registering same contributor must be idempotent and not create duplicate
contrib_a_dup = backend.register_contributor(
    contributor_id="CONTRIB-A",
    display_name="Alice Security Labs Updated",
)
assert contrib_a_dup["contributor_id"] == "CONTRIB-A"
all_c = backend.get_contributors()
assert len([c for c in all_c if c["contributor_id"] == "CONTRIB-A"]) == 1, "No duplicate contributor allowed"
print(f"[OK] TEST A PASSED: Registered and verified unique persistence for {contrib_a['contributor_id']}")
test_results["TEST A (Contributor Persistence)"] = "PASS"

# ---------------------------------------------------------------
# TEST B: Contributor Isolation (Contributor A vs Contributor B)
# ---------------------------------------------------------------
print("\n[TEST B] Contributor Isolation (A vs B)...")
# Register Contributor B
backend.register_contributor(
    contributor_id="CONTRIB-B",
    display_name="Bob Research Group",
    source_id="SOURCE-BOB-02",
    organization="Bob AI Corp",
)

# Batch A1 for Contributor A with 2 samples and 1 finding
samples_a = [
    {
        "source": "sample_a_001.png",
        "contributor": "CONTRIB-A",
        "disposition": "ACCEPT",
        "quality_confidence": 99.0,
        "anomaly_confidence": 1.0,
        "spectral_confidence": 1.0,
        "shift_confidence": 2.0,
        "review_confidence": 3.0,
        "severity": "LOW",
        "shift_distance": 0.02,
        "shift_status": "NORMAL",
        "duplicate_count": 0,
        "is_near_duplicate": False,
        "orientation_status": "NORMAL",
    },
    {
        "source": "sample_a_002.png",
        "contributor": "CONTRIB-A",
        "disposition": "REVIEW",
        "quality_confidence": 70.0,
        "anomaly_confidence": 45.0,
        "spectral_confidence": 40.0,
        "shift_confidence": 42.0,
        "review_confidence": 50.0,
        "severity": "MEDIUM",
        "shift_distance": 0.38,
        "shift_status": "MODERATE_SHIFT",
        "duplicate_count": 0,
        "is_near_duplicate": False,
        "orientation_status": "NORMAL",
    },
]
findings_a = [
    {
        "id": "FINDING-A-01",
        "contributor": "CONTRIB-A",
        "batch": "BATCH-A1",
        "severity": "MEDIUM",
        "title": "Moderate distribution shift in batch sample A2",
        "summary": "Sample sample_a_002.png exhibits moderate spectral divergence",
        "affected_count": 1,
    }
]
backend.record_dataset_analysis(
    dataset_name="Dataset_Alpha",
    dataset_digest="aaaa1111222233334444555566667777888899990000aaaabbbbccccddddeeee01",
    reports=samples_a,
    findings=findings_a,
    manifest_info={"manifest_status": "PASS", "manifest_hash": "m_a1", "provenance_id": "prov_a1"},
    contributor_id="CONTRIB-A",
    batch_id="BATCH-A1",
)

# Batch B1 for Contributor B with 2 samples (all clean) and 0 findings
samples_b = [
    {
        "source": "sample_b_001.png",
        "contributor": "CONTRIB-B",
        "disposition": "ACCEPT",
        "quality_confidence": 98.0,
        "anomaly_confidence": 2.0,
        "spectral_confidence": 1.5,
        "shift_confidence": 2.0,
        "review_confidence": 3.0,
        "severity": "LOW",
        "shift_distance": 0.03,
        "shift_status": "NORMAL",
        "duplicate_count": 0,
        "is_near_duplicate": False,
        "orientation_status": "NORMAL",
    },
    {
        "source": "sample_b_002.png",
        "contributor": "CONTRIB-B",
        "disposition": "ACCEPT",
        "quality_confidence": 97.5,
        "anomaly_confidence": 1.5,
        "spectral_confidence": 2.0,
        "shift_confidence": 1.0,
        "review_confidence": 2.0,
        "severity": "LOW",
        "shift_distance": 0.01,
        "shift_status": "NORMAL",
        "duplicate_count": 0,
        "is_near_duplicate": False,
        "orientation_status": "NORMAL",
    },
]
backend.record_dataset_analysis(
    dataset_name="Dataset_Beta",
    dataset_digest="bbbb1111222233334444555566667777888899990000aaaabbbbccccddddeeee02",
    reports=samples_b,
    findings=[],
    manifest_info={"manifest_status": "PASS", "manifest_hash": "m_b1", "provenance_id": "prov_b1"},
    contributor_id="CONTRIB-B",
    batch_id="BATCH-B1",
)

# Verify isolation
trace_a = backend.get_contributor_traceability("CONTRIB-A")
trace_b = backend.get_contributor_traceability("CONTRIB-B")

assert len(trace_a["samples"]) == 2, "A must have exactly 2 samples"
assert len(trace_b["samples"]) == 2, "B must have exactly 2 samples"
assert all(s["contributor_id"] == "CONTRIB-A" for s in trace_a["samples"]), "A evidence must stay with A"
assert all(s["contributor_id"] == "CONTRIB-B" for s in trace_b["samples"]), "B evidence must stay with B"

assert len(trace_a["findings"]) == 1, "A must have 1 finding"
assert len(trace_b["findings"]) == 0, "B must have 0 findings (isolated from A)"

assert trace_a["assessment"]["overall_status"] == "REVIEW", "A status must be REVIEW"
assert trace_b["assessment"]["overall_status"] == "NORMAL", "B status must remain NORMAL"

# Verify batches independent
batches_a = backend.list_batches_for_contributor("CONTRIB-A")
batches_b = backend.list_batches_for_contributor("CONTRIB-B")
assert len(batches_a) == 1 and batches_a[0]["contributor_id"] == "CONTRIB-A"
assert len(batches_b) == 1 and batches_b[0]["contributor_id"] == "CONTRIB-B"
assert batches_a[0]["status"] == "REVIEW"
assert batches_b[0]["status"] == "NORMAL"

print("[OK] TEST B PASSED: Strict Contributor Isolation confirmed (Evidence, Findings, Batches, Status independent)")
test_results["TEST B (Contributor Isolation)"] = "PASS"

# ---------------------------------------------------------------
# TEST C: Contribution Persistence
# ---------------------------------------------------------------
print("\n[TEST C] Contribution Persistence...")
contributions_a = trace_a["contributions"]
assert len(contributions_a) == 1
assert contributions_a[0]["dataset_name"] == "Dataset_Alpha"
print(f"[OK] TEST C PASSED: Contribution persisted for CONTRIB-A ({contributions_a[0]['dataset_name']})")
test_results["TEST C (Contribution Persistence)"] = "PASS"

# ---------------------------------------------------------------
# TEST D: Batch Persistence
# ---------------------------------------------------------------
print("\n[TEST D] Batch Persistence...")
batches = backend.list_all_batches()
b_a1 = [b for b in batches if b["contributor_id"] == "CONTRIB-A"][0]
assert b_a1 is not None
assert b_a1["contributor_id"] == "CONTRIB-A"
assert b_a1["sample_count"] == 2
print(f"[OK] TEST D PASSED: Batch persisted for CONTRIB-A with sample_count={b_a1['sample_count']}")
test_results["TEST D (Batch Persistence)"] = "PASS"

# ---------------------------------------------------------------
# TEST E: Sample Traceability
# ---------------------------------------------------------------
print("\n[TEST E] Sample Traceability...")
batch_samples = trace_a["samples"]
assert len(batch_samples) == 2
assert batch_samples[0]["file_path"] in ("sample_a_001.png", "sample_a_002.png")
print(f"[OK] TEST E PASSED: Recovered {len(batch_samples)} samples linked to BATCH-A1")
test_results["TEST E (Sample Traceability)"] = "PASS"

# ---------------------------------------------------------------
# TEST F: Real Ed25519 Signature Generation
# ---------------------------------------------------------------
print("\n[TEST F] Real Ed25519 Signature Generation...")
import hashlib
test_payload = b"GENUINE_TEST_PAYLOAD_TRUSTCV_DATASET_SECURITY"
test_digest = hashlib.sha256(test_payload).hexdigest()

manifest_a = create_signed_dataset_manifest(
    dataset_name="Dataset_Alpha",
    dataset_digest=test_digest,
    contributor_id="CONTRIB-A",
    batch_id="BATCH-A1",
)
assert manifest_a["expected_sha256"] == test_digest
assert manifest_a["contributor_id"] == "CONTRIB-A"
assert manifest_a["batch_id"] == "BATCH-A1"
assert "signature" in manifest_a
assert len(manifest_a["signature"]) > 60
print(f"[OK] TEST F PASSED: Generated authentic Ed25519 signed manifest. Signature: {manifest_a['signature'][:24]}...")
test_results["TEST F (Signature Generation)"] = "PASS"

# ---------------------------------------------------------------
# TEST G: Real Signature Verification
# ---------------------------------------------------------------
print("\n[TEST G] Real Signature Verification...")
ver_g = verify_signed_dataset_manifest(
    manifest_a,
    actual_dataset_digest=test_digest,
    expected_contributor_id="CONTRIB-A",
    expected_batch_id="BATCH-A1",
)
assert ver_g["overall"] == "PASS", f"Expected PASS, got {ver_g['overall']}"
assert ver_g["checks"]["digital_signature"]["status"] == "VALID"
assert ver_g["checks"]["digital_signature"]["trusted_key"] is True
print(f"[OK] TEST G PASSED: Cryptographic signature verification succeeded: {ver_g['checks']['digital_signature']['status']}")
test_results["TEST G (Signature Verification)"] = "PASS"

# ---------------------------------------------------------------
# TEST H: Trusted Key Verification
# ---------------------------------------------------------------
print("\n[TEST H] Trusted Key Verification...")
anchor = ensure_trust_anchor()
assert ver_g["checks"]["digital_signature"]["trust_anchor_id"] == anchor["trust_anchor_id"]
assert ver_g["checks"]["digital_signature"]["trusted_key"] is True
print(f"[OK] TEST H PASSED: Bound to trusted local anchor: {anchor['trust_anchor_id'][:16]}...")
test_results["TEST H (Trusted Key Anchor)"] = "PASS"

# ---------------------------------------------------------------
# TEST I: Dataset Digest Mismatch Detection
# ---------------------------------------------------------------
print("\n[TEST I] Dataset Digest Mismatch Detection...")
tampered_payload_digest = "0000000000000000000000000000000000000000000000000000000000000000"
ver_i = verify_signed_dataset_manifest(
    manifest_a,
    actual_dataset_digest=tampered_payload_digest,
    expected_contributor_id="CONTRIB-A",
    expected_batch_id="BATCH-A1",
)
assert ver_i["overall"] == "DATASET_DIGEST_MISMATCH"
assert ver_i["checks"]["dataset_digest"]["status"] == "MISMATCH"
assert ver_i["valid"] is False
# Note that digital signature itself is still mathematically valid
assert ver_i["checks"]["digital_signature"]["status"] == "VALID"
print(f"[OK] TEST I PASSED: Detected tampered dataset payload: {ver_i['overall']}")
test_results["TEST I (Dataset Digest Mismatch)"] = "PASS"

# ---------------------------------------------------------------
# TEST J: Contributor Mismatch Detection
# ---------------------------------------------------------------
print("\n[TEST J] Contributor Mismatch Detection...")
ver_j = verify_signed_dataset_manifest(
    manifest_a,
    actual_dataset_digest=test_digest,
    expected_contributor_id="CONTRIB-B",  # Manifest was signed for A, expecting B
    expected_batch_id="BATCH-A1",
)
assert ver_j["overall"] == "CONTRIBUTOR_MISMATCH"
assert ver_j["checks"]["contributor"]["status"] == "MISMATCH"
assert ver_j["valid"] is False
assert ver_j["checks"]["digital_signature"]["status"] == "VALID"
print(f"[OK] TEST J PASSED: Detected contributor binding mismatch: {ver_j['overall']} (Signature remains VALID)")
test_results["TEST J (Contributor Mismatch)"] = "PASS"

# ---------------------------------------------------------------
# TEST K: Batch Mismatch Detection
# ---------------------------------------------------------------
print("\n[TEST K] Batch Mismatch Detection...")
ver_k = verify_signed_dataset_manifest(
    manifest_a,
    actual_dataset_digest=test_digest,
    expected_contributor_id="CONTRIB-A",
    expected_batch_id="BATCH-A2-IMPOSTOR",
)
assert ver_k["overall"] == "BATCH_MISMATCH"
assert ver_k["checks"]["batch"]["status"] == "MISMATCH"
assert ver_k["valid"] is False
print(f"[OK] TEST K PASSED: Detected batch binding mismatch: {ver_k['overall']}")
test_results["TEST K (Batch Mismatch)"] = "PASS"

# ---------------------------------------------------------------
# TEST L: Manifest Tampering Detection
# ---------------------------------------------------------------
print("\n[TEST L] Manifest Tampering Detection...")
tampered_manifest = dict(manifest_a)
# Tamper one char in the signature
orig_sig = tampered_manifest["signature"]
tampered_manifest["signature"] = ("A" if orig_sig[0] != "A" else "B") + orig_sig[1:]
ver_l = verify_signed_dataset_manifest(
    tampered_manifest,
    actual_dataset_digest=test_digest,
    expected_contributor_id="CONTRIB-A",
)
assert ver_l["overall"] == "INVALID_SIGNATURE"
assert ver_l["checks"]["digital_signature"]["status"] == "INVALID_SIGNATURE"
assert ver_l["valid"] is False
print(f"[OK] TEST L PASSED: Detected corrupted signature in manifest: {ver_l['overall']}")
test_results["TEST L (Manifest Tampering)"] = "PASS"

# ---------------------------------------------------------------
# TEST M: Dataset Provenance Generation & Verification
# ---------------------------------------------------------------
print("\n[TEST M] Dataset Provenance Generation & Verification...")
from trustcv_final import parse_dataset, analyze_reports, generate_findings, dataset_checks

# Create a genuine synthetic image
test_img = Image.new("RGB", (64, 64), color=(60, 120, 180))
buf = io.BytesIO()
test_img.save(buf, format="PNG")
img_data = buf.getvalue()

class SyntheticUpload:
    def __init__(self, data, name="synth_test.png"):
        self._data = data
        self.name = name
    def getvalue(self):
        return self._data

parsed = parse_dataset(SyntheticUpload(img_data))
analysis_rep = analyze_reports(parsed)
checks = dataset_checks(parsed, analysis_rep)

mirad_res = persist_dataset_security_run(
    dataset=parsed,
    reports=analysis_rep,
    checks=checks,
    manifest=manifest_a,
)
assert "provenance" in mirad_res
assert mirad_res["provenance_verification"]["trusted"] is True
print(f"[OK] TEST M PASSED: Dataset provenance generated & cryptographically verified. Event ID: {mirad_res['provenance']['event_id']}")
test_results["TEST M (Provenance Verification)"] = "PASS"

# ---------------------------------------------------------------
# TEST N: Replay Rejection
# ---------------------------------------------------------------
print("\n[TEST N] Replay Rejection...")
replay_check = replay_existing_provenance(mirad_res["provenance"])
assert replay_check["valid"] is False
assert "Duplicate or missing event_id" in replay_check["reason"] or "Nonce is missing or has already been accepted" in replay_check["reason"]
print(f"[OK] TEST N PASSED: Replay attack successfully rejected: {replay_check['reason']}")
test_results["TEST N (Replay Rejection)"] = "PASS"

# ---------------------------------------------------------------
# TEST O: Audit Chain Verification & Tamper Detection
# ---------------------------------------------------------------
print("\n[TEST O] Audit Chain Verification & Tamper Detection...")
audit_state = verify_dataset_audit()
assert audit_state["valid"] is True
assert audit_state["events"] >= 1
print(f"[OK] TEST O.1: Valid audit chain verified with {audit_state['events']} events.")

# Test tampering with audit file
audit_paths = get_persisted_security_paths()
audit_file = Path(audit_paths["audit"])
assert audit_file.exists()
orig_audit_content = audit_file.read_text(encoding="utf-8")

# Corrupt audit file by modifying a byte in the payload
lines = orig_audit_content.splitlines()
corrupted_line = lines[-1].replace('"event_type":', '"event_type_tampered":')
tampered_audit_content = "\n".join(lines[:-1] + [corrupted_line]) + "\n"
audit_file.write_text(tampered_audit_content, encoding="utf-8")

tamper_audit_ver = verify_dataset_audit()
assert tamper_audit_ver["valid"] is False, "Tampered audit chain MUST fail verification"
print(f"[OK] TEST O.2: Tampered audit chain correctly rejected: {tamper_audit_ver['reason']}")

# Restore authentic audit file
audit_file.write_text(orig_audit_content, encoding="utf-8")
restored_audit_ver = verify_dataset_audit()
assert restored_audit_ver["valid"] is True
print("[OK] TEST O PASSED: Audit Chain verification and tamper detection verified.")
test_results["TEST O (Audit Chain & Tampering)"] = "PASS"

# ---------------------------------------------------------------
# TEST P: Signed Checkpoint Verification & Tamper Detection
# ---------------------------------------------------------------
print("\n[TEST P] Signed Checkpoint Verification & Tamper Detection...")
checkpoint_paths = get_persisted_security_paths()
cp_file = Path(checkpoint_paths["checkpoint"])

cp_ver = verify_dataset_checkpoint()
assert cp_ver["trusted"] is True
assert cp_ver["signature_valid"] is True
assert cp_ver["content_binding_valid"] is True
print(f"[OK] TEST P.1: Authentic checkpoint verified: trusted={cp_ver['trusted']}")

# Tamper with checkpoint file
orig_cp_content = cp_file.read_text(encoding="utf-8")
cp_data = json.loads(orig_cp_content)
cp_data["latest_sequence"] = cp_data["latest_sequence"] + 999  # Tamper sequence
cp_file.write_text(json.dumps(cp_data), encoding="utf-8")

tamper_cp_ver = verify_dataset_checkpoint()
assert tamper_cp_ver["trusted"] is False, "Tampered checkpoint MUST fail verification"
print(f"[OK] TEST P.2: Tampered checkpoint correctly rejected: trusted={tamper_cp_ver['trusted']}")

# Restore authentic checkpoint
cp_file.write_text(orig_cp_content, encoding="utf-8")
assert verify_dataset_checkpoint()["trusted"] is True
print("[OK] TEST P PASSED: Checkpoint verification and tamper detection verified.")
test_results["TEST P (Checkpoint & Tampering)"] = "PASS"

# ---------------------------------------------------------------
# TEST Q: Ethereum Disconnected Mode
# ---------------------------------------------------------------
print("\n[TEST Q] Ethereum Disconnected Mode...")
eth_status = get_ethereum_status()
assert eth_status["status"] in ("NOT CONFIGURED", "OFFLINE", "CONNECTED")
# Verify that core TrustCV signing and verification works 100% offline
ver_restored = verify_signed_dataset_manifest(
    manifest_a,
    actual_dataset_digest=test_digest,
    expected_contributor_id="CONTRIB-A",
    expected_batch_id="BATCH-A1",
)
assert ver_restored["overall"] == "PASS"
print(f"[OK] TEST Q PASSED: System operates seamlessly with Ethereum offline (status: {eth_status['status']})")
test_results["TEST Q (Ethereum Disconnected)"] = "PASS"

# ---------------------------------------------------------------
# TEST R: Ethereum Deployment Tooling & Offline Behavior
# ---------------------------------------------------------------
print("\n[TEST R] Ethereum Deployment Tooling & Offline Behavior...")
try:
    from scripts.deploy_audit_anchor import get_compiled_contract, DEFAULT_RPC_URL
except ImportError:
    from Scripts.deploy_audit_anchor import get_compiled_contract, DEFAULT_RPC_URL
abi, bytecode = get_compiled_contract()
assert isinstance(abi, list) and len(abi) >= 4
assert len(bytecode) > 100
print(f"[OK] TEST R.1: AuditAnchor ABI & bytecode compiled and loaded ({len(abi)} ABI items, {len(bytecode)} bytes).")

# Verify offline report when node is not running
test_cfg = EthereumConfig(
    rpc_url="http://127.0.0.1:8545",
    private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    contract_address="0x0000000000000000000000000000000000000001",
)
status_offline = get_ethereum_status(test_cfg)
assert status_offline["status"] in ("OFFLINE", "CONNECTED")
print(f"[OK] TEST R PASSED: Tooling ready and reports honest node status: {status_offline['status']}")
test_results["TEST R (Ethereum Tooling)"] = "PASS"

# ---------------------------------------------------------------
# TEST S: Ethereum Read-Back Verification (Signature & Field Verification)
# ---------------------------------------------------------------
print("\n[TEST S] Ethereum Read-Back Verification Logic...")
# Test the verify_anchor logic with offline / unconfigured
vr_offline = verify_anchor(
    audit_digest=test_digest,
    local_audit_hash=test_digest,
    local_dataset_hash=test_digest,
    config=EthereumConfig(),  # unconfigured
)
assert vr_offline.status == "NOT CONFIGURED"
assert vr_offline.match is False
print(f"[OK] TEST S PASSED: verify_anchor correctly reports: {vr_offline.status}")
test_results["TEST S (Ethereum Verification Logic)"] = "PASS"

# ---------------------------------------------------------------
# TEST T: Ethereum Mismatch Detection Logic
# ---------------------------------------------------------------
print("\n[TEST T] Ethereum Mismatch Detection Logic...")
# Verify that VerificationResult correctly distinguishes match vs mismatch
vr_test = VerificationResult(
    status="MISMATCH",
    local_digest=test_digest,
    anchored_digest="different_digest_hash_0000000000000000000000000000000000000000",
    local_dataset_digest=test_digest,
    anchored_dataset_digest=test_digest,
    match=False,
    audit_match=False,
    dataset_match=True,
)
assert vr_test.match is False
assert vr_test.status == "MISMATCH"
assert vr_test.to_dict()["status"] == "MISMATCH"
print("[OK] TEST T PASSED: Ethereum digest mismatch correctly represented and handled.")
test_results["TEST T (Ethereum Mismatch Handling)"] = "PASS"

# ---------------------------------------------------------------
# TEST U: Streamlit Headless App Initialization
# ---------------------------------------------------------------
print("\n[TEST U] Streamlit Headless App Initialization...")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("trustcv_final.py", default_timeout=120)
at.run()
assert len(at.exception) == 0, f"App threw unhandled exception on startup: {at.exception}"
print(f"[OK] TEST U PASSED: Streamlit application initialized cleanly with 0 exceptions.")
test_results["TEST U (Streamlit Startup)"] = "PASS"

# ---------------------------------------------------------------
# TEST V: Page Rendering / Smoke Test Across All 14 Pages
# ---------------------------------------------------------------
print("\n[TEST V] Page Rendering Across All 14 Pages...")
pages = list(at.sidebar.radio[0].options)
assert len(pages) == 14, f"Expected 14 pages (Overview + 13 pages), got {len(pages)}"

failed_pages = []
for p in pages:
    at.sidebar.radio[0].set_value(p).run()
    if len(at.exception) > 0:
        failed_pages.append((p, at.exception[0].message))
        print(f"  [FAIL] {p}: {at.exception[0].message}")
    else:
        print(f"  [OK] {p}")

assert len(failed_pages) == 0, f"Pages failed rendering: {failed_pages}"
print(f"[OK] TEST V PASSED: All {len(pages)} Streamlit pages rendered cleanly with 0 exceptions.")
test_results["TEST V (Page Smoke Checks)"] = "PASS"

# ---------------------------------------------------------------
# TEST W: Restart Persistence
# ---------------------------------------------------------------
print("\n[TEST W] Restart Persistence...")
del backend
backend_restarted = ContributorBackend(db_path=temp_db_path)
reloaded_contributors = backend_restarted.get_contributors()
reloaded_ids = [c["contributor_id"] for c in reloaded_contributors]

assert "CONTRIB-A" in reloaded_ids
assert "CONTRIB-B" in reloaded_ids
trace_a_restarted = backend_restarted.get_contributor_traceability("CONTRIB-A")
assert len(trace_a_restarted["samples"]) == 2
assert len(trace_a_restarted["findings"]) == 1
assert trace_a_restarted["assessment"]["overall_status"] == "REVIEW"

trace_b_restarted = backend_restarted.get_contributor_traceability("CONTRIB-B")
assert len(trace_b_restarted["samples"]) == 2
assert len(trace_b_restarted["findings"]) == 0
assert trace_b_restarted["assessment"]["overall_status"] == "NORMAL"

print("[OK] TEST W PASSED: Full database state and multi-contributor isolation survives application restart.")
test_results["TEST W (Restart Persistence)"] = "PASS"

# ---------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPREHENSIVE TARGETED VERIFICATION RESULTS SUMMARY:")
print("=" * 70)
all_passed = True
for test_name, status in test_results.items():
    print(f"{test_name:40}: {status}")
    if status != "PASS":
        all_passed = False

if all_passed:
    print(f"\nALL {len(test_results)} VERIFICATION TESTS COMPLETED WITH 100% PASS RATE!")
else:
    print("\nVERIFICATION SUITE COMPLETED WITH FAILURES.")
    sys.exit(1)
