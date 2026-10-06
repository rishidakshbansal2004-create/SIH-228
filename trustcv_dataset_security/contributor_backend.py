"""
TrustCV Contributor Backend & Persistence Architecture.

Maintains persistent backend records for contributors, contributions,
batches, samples, and findings using local SQLite storage.

Air-gapped, fully offline, zero external dependencies.

Hierarchy:
Contributor
    ↓
Contribution
    ↓
Batch
    ↓
Dataset
    ↓
Samples / Findings / Evidence
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parent
RUNTIME = Path.home() / ".trustcv_dataset_security"
RUNTIME.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = RUNTIME / "dataset_contributors.db"
PROJECT_OUTPUT = APP / "demo_output" / "dataset_security"
PROJECT_OUTPUT.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ================================================================
# DATA MODELS
# ================================================================

@dataclass
class ContributorRecord:
    contributor_id: str
    display_name: str
    source_id: str = "DEFAULT"
    organization: str = ""
    status: str = "NORMAL"  # NORMAL, REVIEW, ELEVATED, QUARANTINE_RECOMMENDED
    sample_count: int = 0
    quarantine_count: int = 0
    avg_review_confidence: float = 0.0
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ContributionRecord:
    contribution_id: str
    contributor_id: str
    dataset_name: str
    dataset_digest: str
    source_id: str = "DEFAULT"
    manifest_digest: str | None = None
    provenance_event_id: str | None = None
    audit_id: str | None = None
    checkpoint_id: str | None = None
    status: str = "NORMAL"
    sample_count: int = 0
    quarantine_count: int = 0
    avg_review_confidence: float = 0.0
    created_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRecord:
    batch_id: str
    contribution_id: str
    contributor_id: str
    batch_name: str
    dataset_digest: str
    sample_count: int = 0
    visual_outliers: int = 0
    spectral_outliers: int = 0
    high_shift: int = 0
    quarantine_count: int = 0
    avg_review_confidence: float = 0.0
    status: str = "NORMAL"
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleRecord:
    sample_id: str
    batch_id: str
    contribution_id: str
    contributor_id: str
    file_path: str
    sample_hash: str
    phash: str = ""
    disposition: str = "ACCEPT"  # ACCEPT, REVIEW, QUARANTINE
    severity: str = "LOW"        # LOW, MEDIUM, HIGH
    review_confidence: float = 0.0
    quality_confidence: float = 100.0
    anomaly_confidence: float = 0.0
    spectral_confidence: float = 0.0
    shift_distance: float = 0.0
    shift_status: str = "REFERENCE PENDING"
    quality_issues: list[str] = field(default_factory=list)
    trigger_flags: list[str] = field(default_factory=list)
    is_visual_outlier: bool = False
    is_spectral_outlier: bool = False
    is_near_duplicate: bool = False
    duplicate_count: int = 0
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FindingRecord:
    finding_id: str
    dataset_digest: str
    contribution_id: str | None = None
    batch_id: str | None = None
    contributor_id: str | None = None
    sample_id: str | None = None
    finding_type: str = "GENERIC"
    severity: str = "LOW"
    confidence: float = 0.0
    reason: str = ""
    recommended_disposition: str = "REVIEW"
    affected_samples: list[str] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ================================================================
# SQLITE PERSISTENCE STORE
# ================================================================

class ContributorDatabase:
    """Thread-safe, air-gapped SQLite persistence backend for contributors."""

    _lock = threading.RLock()

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS contributors (
                    contributor_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    source_id TEXT DEFAULT 'DEFAULT',
                    organization TEXT DEFAULT '',
                    status TEXT DEFAULT 'NORMAL',
                    sample_count INTEGER DEFAULT 0,
                    quarantine_count INTEGER DEFAULT 0,
                    avg_review_confidence REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS contributions (
                    contribution_id TEXT PRIMARY KEY,
                    contributor_id TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_digest TEXT NOT NULL,
                    source_id TEXT DEFAULT 'DEFAULT',
                    batch_id TEXT DEFAULT '',
                    manifest_digest TEXT,
                    provenance_event_id TEXT,
                    audit_id TEXT,
                    checkpoint_id TEXT,
                    status TEXT DEFAULT 'NORMAL',
                    sample_count INTEGER DEFAULT 0,
                    quarantine_count INTEGER DEFAULT 0,
                    avg_review_confidence REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    contribution_id TEXT NOT NULL,
                    contributor_id TEXT NOT NULL,
                    batch_name TEXT NOT NULL,
                    dataset_digest TEXT NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    visual_outliers INTEGER DEFAULT 0,
                    spectral_outliers INTEGER DEFAULT 0,
                    high_shift INTEGER DEFAULT 0,
                    quarantine_count INTEGER DEFAULT 0,
                    avg_review_confidence REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'NORMAL',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (contribution_id) REFERENCES contributions(contribution_id) ON DELETE CASCADE,
                    FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    contribution_id TEXT NOT NULL,
                    contributor_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sample_hash TEXT NOT NULL,
                    phash TEXT DEFAULT '',
                    disposition TEXT DEFAULT 'ACCEPT',
                    severity TEXT DEFAULT 'LOW',
                    review_confidence REAL DEFAULT 0.0,
                    quality_confidence REAL DEFAULT 100.0,
                    anomaly_confidence REAL DEFAULT 0.0,
                    spectral_confidence REAL DEFAULT 0.0,
                    shift_distance REAL DEFAULT 0.0,
                    shift_status TEXT DEFAULT 'REFERENCE PENDING',
                    quality_issues_json TEXT DEFAULT '[]',
                    trigger_flags_json TEXT DEFAULT '[]',
                    is_visual_outlier INTEGER DEFAULT 0,
                    is_spectral_outlier INTEGER DEFAULT 0,
                    is_near_duplicate INTEGER DEFAULT 0,
                    duplicate_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE,
                    FOREIGN KEY (contribution_id) REFERENCES contributions(contribution_id) ON DELETE CASCADE,
                    FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS findings (
                    finding_id TEXT PRIMARY KEY,
                    dataset_digest TEXT NOT NULL,
                    contribution_id TEXT,
                    batch_id TEXT,
                    contributor_id TEXT,
                    sample_id TEXT,
                    finding_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    recommended_disposition TEXT NOT NULL,
                    affected_samples_json TEXT DEFAULT '[]',
                    timestamp_utc TEXT NOT NULL,
                    FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_contrib_status ON contributors(status);
                CREATE INDEX IF NOT EXISTS idx_contribution_dataset ON contributions(dataset_digest);
                CREATE INDEX IF NOT EXISTS idx_batches_contrib ON batches(contributor_id);
                CREATE INDEX IF NOT EXISTS idx_samples_contrib ON samples(contributor_id);
                CREATE INDEX IF NOT EXISTS idx_samples_batch ON samples(batch_id);
                CREATE INDEX IF NOT EXISTS idx_findings_contrib ON findings(contributor_id);
                CREATE INDEX IF NOT EXISTS idx_findings_dataset ON findings(dataset_digest);
                """)

                # Safe backward-compatible migrations for existing databases
                c_cols = [r["name"] for r in conn.execute("PRAGMA table_info(contributions)").fetchall()]
                if c_cols and "batch_id" not in c_cols:
                    conn.execute("ALTER TABLE contributions ADD COLUMN batch_id TEXT DEFAULT '';")
                conn.commit()

    # ------------------------------------------------------------
    # CONTRIBUTOR CRUD & RESOLUTION
    # ------------------------------------------------------------

    def register_contributor(
        self,
        contributor_id: str,
        *,
        display_name: str | None = None,
        source_id: str = "DEFAULT",
        organization: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience method to register/resolve a contributor and return dict."""
        rec = self.get_or_create_contributor(
            contributor_id=contributor_id,
            display_name=display_name,
            source_id=source_id,
            organization=organization,
            metadata=metadata,
        )
        return rec.to_dict()

    def get_or_create_contributor(
        self,
        contributor_id: str,
        *,
        display_name: str | None = None,
        source_id: str = "DEFAULT",
        organization: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ContributorRecord:
        """Resolve to existing contributor by ID or create a new stable record."""
        clean_id = str(contributor_id).strip()
        if not clean_id:
            clean_id = "ROOT / UNKNOWN SOURCE"

        name = display_name.strip() if display_name else clean_id
        meta = metadata or {}
        now = _utc_now_iso()

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM contributors WHERE contributor_id = ?",
                    (clean_id,),
                ).fetchone()

                if row:
                    # Contributor already exists — return record
                    return ContributorRecord(
                        contributor_id=row["contributor_id"],
                        display_name=row["display_name"],
                        source_id=row["source_id"],
                        organization=row["organization"],
                        status=row["status"],
                        sample_count=row["sample_count"],
                        quarantine_count=row["quarantine_count"],
                        avg_review_confidence=row["avg_review_confidence"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        metadata=json.loads(row["metadata_json"] or "{}"),
                    )

                conn.execute(
                    """
                    INSERT INTO contributors (
                        contributor_id, display_name, source_id, organization,
                        status, sample_count, quarantine_count, avg_review_confidence,
                        created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, 'NORMAL', 0, 0, 0.0, ?, ?, ?)
                    """,
                    (clean_id, name, source_id, organization, now, now, json.dumps(meta)),
                )
                conn.commit()

                return ContributorRecord(
                    contributor_id=clean_id,
                    display_name=name,
                    source_id=source_id,
                    organization=organization,
                    status="NORMAL",
                    sample_count=0,
                    quarantine_count=0,
                    avg_review_confidence=0.0,
                    created_at=now,
                    updated_at=now,
                    metadata=meta,
                )

    def get_contributor(self, contributor_id: str) -> ContributorRecord | None:
        clean_id = str(contributor_id).strip()
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM contributors WHERE contributor_id = ?",
                    (clean_id,),
                ).fetchone()
                if not row:
                    return None
                return ContributorRecord(
                    contributor_id=row["contributor_id"],
                    display_name=row["display_name"],
                    source_id=row["source_id"],
                    organization=row["organization"],
                    status=row["status"],
                    sample_count=row["sample_count"],
                    quarantine_count=row["quarantine_count"],
                    avg_review_confidence=row["avg_review_confidence"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )

    def list_contributors(self) -> list[ContributorRecord]:
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM contributors ORDER BY quarantine_count DESC, avg_review_confidence DESC"
                ).fetchall()
                return [
                    ContributorRecord(
                        contributor_id=r["contributor_id"],
                        display_name=r["display_name"],
                        source_id=r["source_id"],
                        organization=r["organization"],
                        status=r["status"],
                        sample_count=r["sample_count"],
                        quarantine_count=r["quarantine_count"],
                        avg_review_confidence=r["avg_review_confidence"],
                        created_at=r["created_at"],
                        updated_at=r["updated_at"],
                        metadata=json.loads(r["metadata_json"] or "{}"),
                    )
                    for r in rows
                ]

    def get_contributors(self) -> list[dict[str, Any]]:
        """Return all contributors formatted as dictionaries for UI and API consumption."""
        return [c.to_dict() for c in self.list_contributors()]

    # ------------------------------------------------------------
    # CONTRIBUTION & BATCH INGESTION
    # ------------------------------------------------------------

    def record_dataset_analysis(
        self,
        *,
        dataset_name: str,
        dataset_digest: str,
        reports: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        manifest: dict[str, Any] | None = None,
        manifest_info: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        audit_event_id: str | None = None,
        checkpoint_id: str | None = None,
        contributor_id_override: str | None = None,
        contribution_id_override: str | None = None,
        batch_id_override: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Record a full dataset analysis run into the persistent backend:
        1. Resolve contributors.
        2. Create contribution records per contributor.
        3. Partition into batches.
        4. Persist sample records with dispositions and metrics.
        5. Persist structured findings linked to contributor & samples.
        6. Recompute aggregated statuses upward (sample -> batch -> contributor).
        """
        now = _utc_now_iso()
        manifest_digest = manifest.get("expected_sha256") if manifest else None
        provenance_id = provenance.get("event_id") if provenance else None

        # Clean/normalize contributor and batch overrides
        c_override = str(contributor_id_override or kwargs.get("contributor_id") or "").strip() or None
        if c_override in ("ROOT / UNKNOWN SOURCE", "DEFAULT_CONTRIBUTOR", "", "None"):
            c_override = None

        b_override = str(batch_id_override or kwargs.get("batch_id") or "").strip() or None
        if b_override in ("", "None"):
            b_override = None

        cntrb_override = str(contribution_id_override or kwargs.get("contribution_id") or "").strip() or None
        if cntrb_override in ("", "None"):
            cntrb_override = None

        # Group reports by contributor
        by_contributor: dict[str, list[dict[str, Any]]] = {}
        for r in reports:
            raw_c = r.get("contributor") or ""
            if (not raw_c or raw_c in ("ROOT / UNKNOWN SOURCE", "DEFAULT_CONTRIBUTOR")) and c_override:
                c = c_override
                r["contributor"] = c_override
            else:
                c = raw_c or c_override or "ROOT / UNKNOWN SOURCE"
            by_contributor.setdefault(c, []).append(r)

        # If no image reports (e.g. CSV/Excel dataset), create default record from manifest/intake
        if not by_contributor:
            vendor = c_override or (manifest or {}).get("contributor_id") or (manifest or {}).get("vendor") or "DEFAULT_CONTRIBUTOR"
            by_contributor[vendor] = []

        recorded_contributions = []

        with self._lock:
            with self._get_connection() as conn:
                for contrib_id, items in by_contributor.items():
                    # 1. Ensure contributor exists
                    self.get_or_create_contributor(
                        contrib_id,
                        display_name=contrib_id,
                        source_id=contrib_id,
                        organization=(manifest or {}).get("source", ""),
                    )

                    # Compute contribution aggregates
                    n_samples = len(items)
                    quarantine_count = sum(x.get("disposition") == "QUARANTINE" for x in items)
                    review_confs = [x.get("review_confidence", 0.0) for x in items]
                    avg_review = float(sum(review_confs) / len(review_confs)) if review_confs else 0.0

                    trigger_count = sum(bool(x.get("trigger_flags")) for x in items)
                    if avg_review >= 80 or trigger_count > 0:
                        status = "QUARANTINE_RECOMMENDED"
                    elif avg_review >= 50:
                        status = "ELEVATED"
                    elif avg_review >= 25:
                        status = "REVIEW"
                    else:
                        status = "NORMAL"

                    contribution_key = cntrb_override or f"CNTRB-{contrib_id[:16]}-{dataset_digest[:12]}"
                    primary_batch_name = b_override or (items[0].get("batch_id") if items and items[0].get("batch_id") else "DEFAULT_BATCH")

                    conn.execute(
                        """
                        INSERT INTO contributions (
                            contribution_id, contributor_id, dataset_name, dataset_digest,
                            source_id, batch_id, manifest_digest, provenance_event_id, audit_id,
                            checkpoint_id, status, sample_count, quarantine_count,
                            avg_review_confidence, created_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(contribution_id) DO UPDATE SET
                            dataset_name = excluded.dataset_name,
                            batch_id = excluded.batch_id,
                            manifest_digest = excluded.manifest_digest,
                            provenance_event_id = excluded.provenance_event_id,
                            audit_id = excluded.audit_id,
                            checkpoint_id = excluded.checkpoint_id,
                            status = excluded.status,
                            sample_count = excluded.sample_count,
                            quarantine_count = excluded.quarantine_count,
                            avg_review_confidence = excluded.avg_review_confidence,
                            created_at = excluded.created_at
                        """,
                        (
                            contribution_key, contrib_id, dataset_name, dataset_digest,
                            contrib_id, primary_batch_name, manifest_digest, provenance_id, audit_event_id,
                            checkpoint_id, status, n_samples, quarantine_count,
                            round(avg_review, 1), now, json.dumps({"source": contrib_id}),
                        ),
                    )

                    # 2. Partition into batches
                    by_batch: dict[str, list[dict[str, Any]]] = {}
                    for item in items:
                        if b_override:
                            batch_name = b_override
                        elif item.get("batch_id"):
                            batch_name = str(item.get("batch_id"))
                        else:
                            parts = Path(item.get("source", "")).parts
                            batch_name = parts[1] if len(parts) >= 3 else parts[0] if len(parts) >= 2 else "DEFAULT_BATCH"
                        by_batch.setdefault(batch_name, []).append(item)

                    for b_name, b_items in by_batch.items():
                        batch_key = b_name if b_name.startswith("BATCH-") else f"BATCH-{contribution_key}-{b_name}"
                        b_samples = len(b_items)
                        b_vo = sum(bool(x.get("is_visual_outlier")) for x in b_items)
                        b_so = sum(bool(x.get("is_spectral_outlier")) for x in b_items)
                        b_hs = sum(x.get("shift_status") == "HIGH SHIFT" for x in b_items)
                        b_qc = sum(x.get("disposition") == "QUARANTINE" for x in b_items)
                        b_confs = [x.get("review_confidence", 0.0) for x in b_items]
                        b_avg = float(sum(b_confs) / len(b_confs)) if b_confs else 0.0

                        if b_avg >= 80 or any(bool(x.get("trigger_flags")) for x in b_items):
                            b_status = "QUARANTINE_RECOMMENDED"
                        elif b_avg >= 50:
                            b_status = "ELEVATED"
                        elif b_avg >= 25:
                            b_status = "REVIEW"
                        else:
                            b_status = "NORMAL"

                        conn.execute(
                            """
                            INSERT INTO batches (
                                batch_id, contribution_id, contributor_id, batch_name,
                                dataset_digest, sample_count, visual_outliers, spectral_outliers,
                                high_shift, quarantine_count, avg_review_confidence, status, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(batch_id) DO UPDATE SET
                                sample_count = excluded.sample_count,
                                visual_outliers = excluded.visual_outliers,
                                spectral_outliers = excluded.spectral_outliers,
                                high_shift = excluded.high_shift,
                                quarantine_count = excluded.quarantine_count,
                                avg_review_confidence = excluded.avg_review_confidence,
                                status = excluded.status
                            """,
                            (
                                batch_key, contribution_key, contrib_id, b_name,
                                dataset_digest, b_samples, b_vo, b_so, b_hs, b_qc,
                                round(b_avg, 1), b_status, now,
                            ),
                        )

                        # 3. Persist individual samples
                        for s_idx, item in enumerate(b_items):
                            idx = item.get("index", s_idx)
                            sample_id = item.get("sample_id") or f"SMP-{dataset_digest[:8]}-{b_name[:8]}-{idx:04d}"
                            raw_ph = item.get("phash", 0)
                            if isinstance(raw_ph, int):
                                phash_str = f"{raw_ph:064x}"
                            else:
                                phash_str = str(raw_ph or "")

                            conn.execute(
                                """
                                INSERT INTO samples (
                                    sample_id, batch_id, contribution_id, contributor_id,
                                    file_path, sample_hash, phash, disposition, severity,
                                    review_confidence, quality_confidence, anomaly_confidence,
                                    spectral_confidence, shift_distance, shift_status,
                                    quality_issues_json, trigger_flags_json,
                                    is_visual_outlier, is_spectral_outlier, is_near_duplicate,
                                    duplicate_count, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(sample_id) DO UPDATE SET
                                    batch_id = excluded.batch_id,
                                    contribution_id = excluded.contribution_id,
                                    contributor_id = excluded.contributor_id,
                                    file_path = excluded.file_path,
                                    sample_hash = excluded.sample_hash,
                                    phash = excluded.phash,
                                    disposition = excluded.disposition,
                                    severity = excluded.severity,
                                    review_confidence = excluded.review_confidence,
                                    quality_confidence = excluded.quality_confidence,
                                    anomaly_confidence = excluded.anomaly_confidence,
                                    spectral_confidence = excluded.spectral_confidence,
                                    shift_distance = excluded.shift_distance,
                                    shift_status = excluded.shift_status,
                                    quality_issues_json = excluded.quality_issues_json,
                                    trigger_flags_json = excluded.trigger_flags_json,
                                    is_visual_outlier = excluded.is_visual_outlier,
                                    is_spectral_outlier = excluded.is_spectral_outlier,
                                    is_near_duplicate = excluded.is_near_duplicate,
                                    duplicate_count = excluded.duplicate_count
                                """,
                                (
                                    sample_id, batch_key, contribution_key, contrib_id,
                                    item.get("source", ""),
                                    item.get("hash", ""),
                                    phash_str,
                                    item.get("disposition", "ACCEPT"),
                                    item.get("severity", "LOW"),
                                    float(item.get("review_confidence", 0.0)),
                                    float(item.get("quality_confidence", 100.0)),
                                    float(item.get("anomaly_confidence", 0.0)),
                                    float(item.get("spectral_confidence", 0.0)),
                                    float(item.get("shift_distance", 0.0)),
                                    item.get("shift_status", "REFERENCE PENDING"),
                                    json.dumps(item.get("quality_issues", [])),
                                    json.dumps(item.get("trigger_flags", [])),
                                    1 if item.get("is_visual_outlier") else 0,
                                    1 if item.get("is_spectral_outlier") else 0,
                                    1 if item.get("is_near_duplicate") else 0,
                                    int(item.get("duplicate_count", 0)),
                                    now,
                                ),
                            )

                    # 4. Update contributor lifetime totals
                    all_c_samples = conn.execute(
                        "SELECT COUNT(*) as c, SUM(CASE WHEN disposition='QUARANTINE' THEN 1 ELSE 0 END) as q, AVG(review_confidence) as a FROM samples WHERE contributor_id = ?",
                        (contrib_id,),
                    ).fetchone()

                    tot_samples = all_c_samples["c"] or 0
                    tot_q = all_c_samples["q"] or 0
                    c_avg = float(all_c_samples["a"] or 0.0)

                    if c_avg >= 80 or tot_q > 0:
                        c_status = "QUARANTINE_RECOMMENDED"
                    elif c_avg >= 50:
                        c_status = "ELEVATED"
                    elif c_avg >= 25:
                        c_status = "REVIEW"
                    else:
                        c_status = "NORMAL"

                    conn.execute(
                        """
                        UPDATE contributors SET
                            sample_count = ?,
                            quarantine_count = ?,
                            avg_review_confidence = ?,
                            status = ?,
                            updated_at = ?
                        WHERE contributor_id = ?
                        """,
                        (tot_samples, tot_q, round(c_avg, 1), c_status, now, contrib_id),
                    )

                    recorded_contributions.append(contribution_key)

                # 5. Persist structured findings
                for f_idx, f in enumerate(findings):
                    f_id = f.get("finding_id") or f.get("id") or f"F-{dataset_digest[:8]}-{f_idx:03d}-{now}"
                    aff = f.get("affected_samples") or ([f["sample_id"]] if "sample_id" in f else [])
                    c_id = f.get("contributor_id") or f.get("contributor") or c_override
                    b_id = f.get("batch_id") or f.get("batch") or b_override
                    contrib_ref = cntrb_override
                    s_id = f.get("sample_id")

                    if aff and len(aff) > 0:
                        # Locating sample from first affected file path or sample ID
                        s_row = conn.execute(
                            "SELECT sample_id, contributor_id, batch_id, contribution_id FROM samples WHERE file_path = ? OR sample_id = ?",
                            (aff[0], aff[0]),
                        ).fetchone()
                        if s_row:
                            c_id = c_id or s_row["contributor_id"]
                            b_id = b_id or s_row["batch_id"]
                            contrib_ref = contrib_ref or s_row["contribution_id"]
                            s_id = s_id or s_row["sample_id"]

                    if not contrib_ref and c_id:
                        c_row = conn.execute(
                            "SELECT contribution_id FROM contributions WHERE contributor_id = ? AND dataset_digest = ?",
                            (c_id, dataset_digest),
                        ).fetchone()
                        if c_row:
                            contrib_ref = c_row["contribution_id"]

                    f_type = f.get("finding_type") or f.get("type") or f.get("title") or "GENERIC"
                    reason_text = f.get("reason") or f.get("summary") or f.get("description") or ""

                    conn.execute(
                        """
                        INSERT INTO findings (
                            finding_id, dataset_digest, contribution_id, batch_id,
                            contributor_id, sample_id, finding_type, severity,
                            confidence, reason, recommended_disposition,
                            affected_samples_json, timestamp_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(finding_id) DO UPDATE SET
                            contribution_id = excluded.contribution_id,
                            batch_id = excluded.batch_id,
                            contributor_id = excluded.contributor_id,
                            sample_id = excluded.sample_id,
                            finding_type = excluded.finding_type,
                            severity = excluded.severity,
                            confidence = excluded.confidence,
                            reason = excluded.reason,
                            recommended_disposition = excluded.recommended_disposition,
                            affected_samples_json = excluded.affected_samples_json
                        """,
                        (
                            f_id, dataset_digest, contrib_ref, b_id,
                            c_id, s_id, f_type,
                            f.get("severity", "LOW"), float(f.get("confidence", 0.0)),
                            reason_text, f.get("recommended_disposition", "REVIEW"),
                            json.dumps(aff), f.get("timestamp") or now,
                        ),
                    )

                conn.commit()

        # Mirror/Export to project output directory for inspection
        self.export_snapshot()

        return {
            "dataset_digest": dataset_digest,
            "contributions_recorded": recorded_contributions,
            "contributors_count": len(by_contributor),
            "findings_recorded": len(findings),
        }

    # ------------------------------------------------------------
    # QUERY & TRACEABILITY
    # ------------------------------------------------------------

    def get_contributor_traceability(self, contributor_id: str) -> dict[str, Any]:
        """
        Reconstruct full end-to-end evidence chain for a contributor:
        Contributor
          → Contributions
             → Batches
                → Samples
                   → Findings
                      → Provenance / Manifest / Audit
        """
        with self._lock:
            with self._get_connection() as conn:
                contributor = conn.execute(
                    "SELECT * FROM contributors WHERE contributor_id = ?",
                    (contributor_id,),
                ).fetchone()
                if not contributor:
                    return {"found": False, "contributor_id": contributor_id}

                contributions = conn.execute(
                    "SELECT * FROM contributions WHERE contributor_id = ? ORDER BY created_at DESC",
                    (contributor_id,),
                ).fetchall()

                batches = conn.execute(
                    "SELECT * FROM batches WHERE contributor_id = ? ORDER BY created_at DESC",
                    (contributor_id,),
                ).fetchall()

                samples = conn.execute(
                    "SELECT * FROM samples WHERE contributor_id = ? ORDER BY review_confidence DESC",
                    (contributor_id,),
                ).fetchall()

                findings = conn.execute(
                    "SELECT * FROM findings WHERE contributor_id = ? ORDER BY confidence DESC",
                    (contributor_id,),
                ).fetchall()

                return {
                    "found": True,
                    "contributor": dict(contributor),
                    "contributions": [dict(c) for c in contributions],
                    "batches": [dict(b) for b in batches],
                    "samples": [
                        {
                            **dict(s),
                            "quality_issues": json.loads(s["quality_issues_json"] or "[]"),
                            "trigger_flags": json.loads(s["trigger_flags_json"] or "[]"),
                        }
                        for s in samples
                    ],
                    "findings": [
                        {
                            **dict(f),
                            "affected_samples": json.loads(f["affected_samples_json"] or "[]"),
                        }
                        for f in findings
                    ],
                    "assessment": {
                        "overall_status": contributor["status"],
                        "sample_count": contributor["sample_count"],
                        "quarantine_count": contributor["quarantine_count"],
                        "avg_review_confidence": contributor["avg_review_confidence"],
                    },
                }

    def get_contributor_aggregation(self, contributor_id: str) -> dict[str, Any]:
        """
        Compute real-time aggregated metrics directly from persisted SQLite records:
        - lifetime samples
        - contributions
        - batches
        - exact duplicate findings
        - near duplicate findings
        - label findings
        - OOD/shift findings
        - trigger-like anomaly indicators
        - anomaly findings
        - review percentage
        - quarantine count
        - status
        """
        with self._lock:
            with self._get_connection() as conn:
                c_row = conn.execute("SELECT * FROM contributors WHERE contributor_id = ?", (contributor_id,)).fetchone()
                if not c_row:
                    return {"found": False, "contributor_id": contributor_id}

                n_contributions = conn.execute(
                    "SELECT COUNT(*) as cnt FROM contributions WHERE contributor_id = ?", (contributor_id,)
                ).fetchone()["cnt"]

                n_batches = conn.execute(
                    "SELECT COUNT(*) as cnt FROM batches WHERE contributor_id = ?", (contributor_id,)
                ).fetchone()["cnt"]

                s_stats = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_samples,
                        SUM(CASE WHEN duplicate_count > 0 THEN 1 ELSE 0 END) as exact_dups,
                        SUM(CASE WHEN is_near_duplicate = 1 THEN 1 ELSE 0 END) as near_dups,
                        SUM(CASE WHEN shift_status = 'HIGH SHIFT' OR is_visual_outlier = 1 OR is_spectral_outlier = 1 THEN 1 ELSE 0 END) as ood_shifts,
                        SUM(CASE WHEN disposition = 'QUARANTINE' THEN 1 ELSE 0 END) as quarantines,
                        AVG(review_confidence) as avg_review
                    FROM samples WHERE contributor_id = ?
                    """,
                    (contributor_id,)
                ).fetchone()

                f_stats = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_findings,
                        SUM(CASE WHEN finding_type = 'EXACT_DUPLICATE' THEN 1 ELSE 0 END) as exact_dup_findings,
                        SUM(CASE WHEN finding_type IN ('NEAR_DUPLICATE', 'PERCEPTUAL_SIMILARITY') THEN 1 ELSE 0 END) as near_dup_findings,
                        SUM(CASE WHEN finding_type IN ('LABEL_SHIFT', 'LABEL_CONSISTENCY', 'UNANNOTATED') THEN 1 ELSE 0 END) as label_findings,
                        SUM(CASE WHEN finding_type IN ('TRIGGER_INJECTION_INDICATOR', 'TRIGGER_ANOMALY', 'ANOMALY_TRIGGER') THEN 1 ELSE 0 END) as trigger_findings
                    FROM findings WHERE contributor_id = ?
                    """,
                    (contributor_id,)
                ).fetchone()

                lifetime_samples = s_stats["total_samples"] or 0
                quarantine_count = s_stats["quarantines"] or 0
                avg_review = round(float(s_stats["avg_review"] or 0.0), 1)

                exact_dups = max(s_stats["exact_dups"] or 0, f_stats["exact_dup_findings"] or 0)
                near_dups = max(s_stats["near_dups"] or 0, f_stats["near_dup_findings"] or 0)
                label_findings = f_stats["label_findings"] or 0
                ood_shifts = s_stats["ood_shifts"] or 0
                trigger_indicators = f_stats["trigger_findings"] or 0
                total_findings = f_stats["total_findings"] or 0

                # Determine aggregated status from evidence
                if avg_review >= 80 or quarantine_count > 0 or trigger_indicators > 0:
                    status = "QUARANTINE_RECOMMENDED"
                elif avg_review >= 50 or ood_shifts > 0 or total_findings > 5:
                    status = "ELEVATED"
                elif avg_review >= 25 or total_findings > 0:
                    status = "REVIEW"
                else:
                    status = "NORMAL"

                return {
                    "found": True,
                    "contributor_id": contributor_id,
                    "display_name": c_row["display_name"],
                    "organization": c_row["organization"],
                    "source_id": c_row["source_id"],
                    "lifetime_samples": lifetime_samples,
                    "contributions": n_contributions,
                    "batches": n_batches,
                    "exact_duplicate_findings": exact_dups,
                    "near_duplicate_findings": near_dups,
                    "label_findings": label_findings,
                    "ood_shift_findings": ood_shifts,
                    "trigger_anomaly_indicators": trigger_indicators,
                    "anomaly_findings": total_findings,
                    "review_percentage": avg_review,
                    "quarantine_count": quarantine_count,
                    "status": status,
                    "created_at": c_row["created_at"],
                    "updated_at": c_row["updated_at"],
                }

    def list_batches_for_contributor(self, contributor_id: str) -> list[dict[str, Any]]:
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM batches WHERE contributor_id = ? ORDER BY avg_review_confidence DESC",
                    (contributor_id,),
                ).fetchall()
                return [dict(r) for r in rows]

    def list_all_batches(self) -> list[dict[str, Any]]:
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM batches ORDER BY avg_review_confidence DESC"
                ).fetchall()
                return [dict(r) for r in rows]

    def list_all_findings(self) -> list[dict[str, Any]]:
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM findings ORDER BY confidence DESC"
                ).fetchall()
                return [
                    {
                        **dict(r),
                        "affected_samples": json.loads(r["affected_samples_json"] or "[]"),
                    }
                    for r in rows
                ]

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        """Fetch a single batch record by ID."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
                return dict(row) if row else None

    def get_contribution(self, contribution_id: str) -> dict[str, Any] | None:
        """Fetch a single contribution record by ID."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM contributions WHERE contribution_id = ?", (contribution_id,)).fetchone()
                return dict(row) if row else None

    def get_samples_by_dataset(self, dataset_digest: str) -> list[dict[str, Any]]:
        """Fetch all samples belonging to a dataset across all contributors and batches."""
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT s.* FROM samples s
                    JOIN batches b ON s.batch_id = b.batch_id
                    WHERE b.dataset_digest = ?
                    ORDER BY s.sample_id
                    """,
                    (dataset_digest,),
                ).fetchall()
                return [
                    {
                        **dict(r),
                        "quality_issues": json.loads(r["quality_issues_json"] or "[]"),
                        "trigger_flags": json.loads(r["trigger_flags_json"] or "[]"),
                    }
                    for r in rows
                ]

    def get_samples_by_batch(self, batch_id: str) -> list[dict[str, Any]]:
        """Fetch all samples belonging to a specific batch."""
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute("SELECT * FROM samples WHERE batch_id = ? ORDER BY sample_id", (batch_id,)).fetchall()
                return [
                    {
                        **dict(r),
                        "quality_issues": json.loads(r["quality_issues_json"] or "[]"),
                        "trigger_flags": json.loads(r["trigger_flags_json"] or "[]"),
                    }
                    for r in rows
                ]

    def export_snapshot(self) -> Path:
        """Export JSON snapshot of contributors and contributions to project output."""
        export_file = PROJECT_OUTPUT / "contributors_backend_export.json"
        with self._lock:
            with self._get_connection() as conn:
                contributors = [dict(r) for r in conn.execute("SELECT * FROM contributors").fetchall()]
                contributions = [dict(r) for r in conn.execute("SELECT * FROM contributions").fetchall()]
                batches = [dict(r) for r in conn.execute("SELECT * FROM batches").fetchall()]
                findings = [dict(r) for r in conn.execute("SELECT * FROM findings").fetchall()]

        data = {
            "exported_at": _utc_now_iso(),
            "contributors_count": len(contributors),
            "contributions_count": len(contributions),
            "batches_count": len(batches),
            "findings_count": len(findings),
            "contributors": contributors,
            "contributions": contributions,
            "batches": batches,
            "findings": findings,
        }
        export_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return export_file


# Global singleton instance
_BACKEND_INSTANCE: ContributorDatabase | None = None


def get_contributor_backend(db_path: Path | str | None = None) -> ContributorDatabase:
    global _BACKEND_INSTANCE
    if _BACKEND_INSTANCE is None or db_path is not None:
        _BACKEND_INSTANCE = ContributorDatabase(db_path)
    return _BACKEND_INSTANCE


ContributorBackend = ContributorDatabase
