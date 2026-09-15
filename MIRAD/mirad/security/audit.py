"""Append-only audit records with chain verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mirad.security.canonical import canonicalize
from mirad.security.hashing import hash_bytes


@dataclass
class AuditEvent:
    """Single append-only audit record."""

    audit_id: str
    event_type: str
    timestamp: str
    payload: dict[str, Any]
    previous_hash: str | None
    current_hash: str
    sequence: int
    actor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "audit_id": self.audit_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
            "sequence": self.sequence,
        }
        if self.actor is not None:
            payload["actor"] = self.actor
        return payload


class AuditStore:
    """In-memory append-only audit chain store."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self._events.append(event)

    def list(self) -> list[AuditEvent]:
        return list(self._events)

    def last(self) -> AuditEvent | None:
        return self._events[-1] if self._events else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_audit_event(
    store: AuditStore,
    *,
    event_type: str,
    payload: dict[str, Any],
    actor: str | None = None,
    sequence: int | None = None,
    audit_id: str | None = None,
) -> AuditEvent:
    """Append a new audit event to the chain and bind it to the previous record hash."""
    previous = store.last()
    previous_hash = previous.current_hash if previous else None
    event_sequence = sequence if sequence is not None else (len(store.list()) + 1)
    record = {
        "audit_id": audit_id or f"audit-{event_sequence:06d}",
        "event_type": event_type,
        "timestamp": _utc_now_iso(),
        "payload": payload,
        "previous_hash": previous_hash,
        "sequence": event_sequence,
        "actor": actor,
    }
    current_hash = hash_bytes(canonicalize(record))
    event = AuditEvent(
        audit_id=record["audit_id"],
        event_type=event_type,
        timestamp=record["timestamp"],
        payload=payload,
        previous_hash=previous_hash,
        current_hash=current_hash,
        sequence=event_sequence,
        actor=actor,
    )
    store.append(event)
    return event


def verify_audit_chain(events: list[AuditEvent]) -> bool:
    """Verify the append-only audit hash chain and ordering."""
    for idx, event in enumerate(events):
        if event.sequence != idx + 1:
            return False
        if idx == 0:
            expected_previous = None
        else:
            expected_previous = events[idx - 1].current_hash
        if event.previous_hash != expected_previous:
            return False
        recomputed = hash_bytes(canonicalize({
            "audit_id": event.audit_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "payload": event.payload,
            "previous_hash": event.previous_hash,
            "sequence": event.sequence,
            "actor": event.actor,
        }))
        if recomputed != event.current_hash:
            return False
    return True


def reconstruct_audit(store: AuditStore) -> list[AuditEvent]:
    return store.list()
