import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Lib/site-packages"))

from contributor_backend import ContributorBackend, get_contributor_backend

def test_contributor_registration_coherence():
    print("=" * 70)
    print("TESTING CONTRIBUTOR REGISTRATION API COHERENCE (BUG #1 VERIFICATION)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_reg.db"
        backend = ContributorBackend(db_path=test_db)

        # 1. Test canonical source_id keyword
        res1 = backend.register_contributor(
            contributor_id="CONTRIB-TEST-SRCID-01",
            display_name="Test Source ID User",
            organization="Cyber Defense Org",
            source_id="CUSTOM_CHANNEL_01",
        )
        assert res1["contributor_id"] == "CONTRIB-TEST-SRCID-01"
        assert res1["source_id"] == "CUSTOM_CHANNEL_01"
        assert res1["display_name"] == "Test Source ID User"
        assert res1["organization"] == "Cyber Defense Org"
        print("[OK] Test 1: register_contributor with source_id passed cleanly.")

        # 2. Test legacy/alternative source keyword (the one that previously threw TypeError)
        res2 = backend.register_contributor(
            contributor_id="CONTRIB-TEST-SRC-02",
            display_name="Test Source User",
            organization="Space Research",
            source="WEB_UPLOAD_PORTAL",
        )
        assert res2["contributor_id"] == "CONTRIB-TEST-SRC-02"
        assert res2["source_id"] == "WEB_UPLOAD_PORTAL"
        assert res2["display_name"] == "Test Source User"
        print("[OK] Test 2: register_contributor with source passed cleanly without TypeError.")

        # 3. Test get_or_create_contributor with both
        rec3 = backend.get_or_create_contributor(
            contributor_id="CONTRIB-TEST-SRC-03",
            display_name="Test Direct User",
            source="INLINE_INTAKE",
        )
        assert rec3.contributor_id == "CONTRIB-TEST-SRC-03"
        assert rec3.source_id == "INLINE_INTAKE"
        print("[OK] Test 3: get_or_create_contributor with source passed cleanly.")

        # 4. Verify persistence in SQLite
        all_contributors = backend.get_contributors()
        c_ids = [c["contributor_id"] for c in all_contributors]
        assert "CONTRIB-TEST-SRCID-01" in c_ids
        assert "CONTRIB-TEST-SRC-02" in c_ids
        assert "CONTRIB-TEST-SRC-03" in c_ids
        print(f"[OK] Test 4: All {len(c_ids)} contributors correctly persisted in relational database.")

        # 5. Verify restart persistence
        del backend
        backend_restarted = ContributorBackend(db_path=test_db)
        reloaded = backend_restarted.get_contributors()
        r_ids = [c["contributor_id"] for c in reloaded]
        assert "CONTRIB-TEST-SRC-02" in r_ids
        assert any(c["source_id"] == "WEB_UPLOAD_PORTAL" for c in reloaded if c["contributor_id"] == "CONTRIB-TEST-SRC-02")
        print("[OK] Test 5: Restart persistence verified with intact source_id bindings.")

    print("\n" + "=" * 70)
    print("ALL CONTRIBUTOR REGISTRATION TESTS PASSED 100%!")
    print("=" * 70)

if __name__ == "__main__":
    test_contributor_registration_coherence()
