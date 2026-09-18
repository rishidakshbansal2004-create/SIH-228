"""Persistent MIRAD-backed, hash-chained audit ledger for TrustCV runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mirad.security.audit import AuditEvent, AuditStore, append_audit_event, verify_audit_chain

ZERO_HASH = None


class AuditLedger:
    """Persist MIRAD AuditEvent chains as JSONL, one independent chain per run."""

    def __init__(self, path: str = "audit/mirad_audit_ledger.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _raw_entries(self):
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"_invalid_json": line}

    @staticmethod
    def _run_id(entry: dict[str, Any]) -> str | None:
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        return payload.get("run_id") or entry.get("run_id")

    @staticmethod
    def _to_event(entry: dict[str, Any]) -> AuditEvent | None:
        try:
            return AuditEvent(
                audit_id=str(entry["audit_id"]),
                event_type=str(entry["event_type"]),
                timestamp=str(entry["timestamp"]),
                payload=dict(entry.get("payload") or {}),
                previous_hash=entry.get("previous_hash"),
                current_hash=str(entry["current_hash"]),
                sequence=int(entry["sequence"]),
                actor=entry.get("actor"),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def entries(self, run_id: str | None = None) -> list[dict[str, Any]]:
        result = []
        for entry in self._raw_entries():
            if not isinstance(entry, dict) or "current_hash" not in entry:
                continue
            if run_id is not None and self._run_id(entry) != run_id:
                continue
            normalized = dict(entry)
            normalized["record_hash"] = normalized.get("current_hash")
            result.append(normalized)
        return result

    def events(self, run_id: str) -> list[AuditEvent]:
        events = []
        for entry in self.entries(run_id):
            event = self._to_event(entry)
            if event is not None:
                events.append(event)
        return events

    def last(self, run_id: str) -> AuditEvent | None:
        events = self.events(run_id)
        return events[-1] if events else None

    def last_hash(self, run_id: str) -> str | None:
        last = self.last(run_id)
        return last.current_hash if last else None

    def append(
        self,
        event: dict[str, Any],
        *,
        run_id: str,
        event_type: str | None = None,
        actor: str = "TrustCV",
    ) -> dict[str, Any]:
        """Create a real MIRAD AuditEvent, then persist its canonical hash chain entry."""
        payload = dict(event)
        payload["run_id"] = run_id
        store = AuditStore()
        for existing in self.events(run_id):
            store.append(existing)
        sequence = len(store.list()) + 1
        audit_event = append_audit_event(
            store,
            event_type=event_type or str(payload.get("type") or "trustcv.event"),
            payload=payload,
            actor=actor,
            sequence=sequence,
            audit_id=f"{run_id}:{sequence:04d}",
        )
        record = audit_event.to_dict()
        record["run_id"] = run_id
        record["record_hash"] = audit_event.current_hash
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        return record

    def verify(self, run_id: str | None = None):
        """Verify one run, or every MIRAD chain stored in this file."""
        if run_id is not None:
            events = self.events(run_id)
            if not events:
                return False, "no audit events for run"
            ok = verify_audit_chain(events)
            return ok, "MIRAD audit chain valid" if ok else "MIRAD audit chain verification failed"

        run_ids = []
        for entry in self.entries():
            rid = self._run_id(entry)
            if rid and rid not in run_ids:
                run_ids.append(rid)
        if not run_ids:
            return True, "no MIRAD audit events recorded"
        for rid in run_ids:
            ok, msg = self.verify(rid)
            if not ok:
                return False, f"{rid}: {msg}"
        return True, f"all {len(run_ids)} MIRAD audit chains valid"
