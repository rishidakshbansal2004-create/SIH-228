"""Replay protection and freshness validation for signed security events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from mirad.security.errors import SecurityErrorCode


@dataclass
class ReplayCheckResult:
    """Outcome of replay/freshness validation."""

    valid: bool
    status: str
    checks: dict[str, bool] = field(default_factory=dict)
    failure_code: str | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


class ReplayStateStore:
    """Simple in-memory replay state keyed by event scope and event identity."""

    def __init__(self) -> None:
        self._seen_event_ids: dict[str, dict[str, Any]] = {}
        self._seen_nonces: dict[str, str] = {}
        self._last_sequence: dict[str, int] = {}

    def record(self, *, event: dict[str, Any], scope: str) -> None:
        event_id = str(event["event_id"])
        nonce = str(event.get("nonce", ""))
        sequence = int(event.get("sequence", -1))
        self._seen_event_ids[f"{scope}:{event_id}"] = event
        self._seen_nonces[f"{scope}:{nonce}"] = event_id
        self._last_sequence[scope] = max(self._last_sequence.get(scope, sequence), sequence)

    def seen_event(self, *, event_id: str, scope: str) -> bool:
        return f"{scope}:{event_id}" in self._seen_event_ids

    def seen_nonce(self, *, nonce: str, scope: str) -> bool:
        return f"{scope}:{nonce}" in self._seen_nonces

    def last_sequence(self, scope: str) -> int | None:
        return self._last_sequence.get(scope)


def check_replay(
    event: dict[str, Any],
    *,
    store: ReplayStateStore,
    freshness_window_seconds: int = 300,
    allowed_skew_seconds: int = 30,
    scope: str = "session",
    now: datetime | None = None,
    expected_context: str | None = None,
    strict_context_binding: bool = True,
    strict_sequence: bool = True,
) -> ReplayCheckResult:
    """Validate freshness, uniqueness, sequence, and context for a security event."""
    now_value = now or _utc_now()
    result = ReplayCheckResult(valid=True, status="VALID", checks={}, failure_code=None, evidence={"scope": scope, "event": event})

    event_id = str(event.get("event_id", ""))
    context = str(event.get("context", ""))
    nonce = str(event.get("nonce", ""))
    sequence = event.get("sequence")
    timestamp_value = event.get("timestamp")

    if not event_id:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.DUPLICATE_EVENT.value
        result.reason = "Event is missing a unique event_id."
        return result
    if strict_context_binding and expected_context is None:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.CONTEXT_MISMATCH.value
        result.reason = "Strict context binding requires an expected context."
        result.checks["context_valid"] = False
        return result
    if strict_context_binding and context != expected_context:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.CONTEXT_MISMATCH.value
        result.reason = "Event context does not match the expected context."
        result.checks["context_valid"] = False
        return result
    if not nonce:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.NONCE_REUSE_DETECTED.value
        result.reason = "Event is missing a nonce."
        return result
    if timestamp_value is None:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.STALE_EVENT.value
        result.reason = "Event is missing a timestamp."
        return result
    if strict_sequence and (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1):
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.SEQUENCE_VIOLATION.value
        result.reason = "A numeric sequence is required by strict replay policy."
        return result

    try:
        timestamp = _parse_timestamp(str(timestamp_value))
    except (TypeError, ValueError, OverflowError):
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.INVALID_TIMESTAMP.value
        result.reason = "Event timestamp is not a valid UTC timestamp."
        result.checks["timestamp_valid"] = False
        return result
    if store.seen_event(event_id=event_id, scope=scope):
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.DUPLICATE_EVENT.value
        result.reason = "Duplicate event_id seen within the replay scope."
        result.checks["duplicate_event"] = False
        return result
    if store.seen_nonce(nonce=nonce, scope=scope):
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.NONCE_REUSE_DETECTED.value
        result.reason = "Nonce already accepted in the same replay scope."
        result.checks["nonce_reuse"] = False
        return result

    freshness_deadline = now_value - timedelta(seconds=freshness_window_seconds)
    if timestamp < freshness_deadline:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.STALE_EVENT.value
        result.reason = "Event is older than the allowed freshness window."
        result.checks["freshness"] = False
        return result

    if timestamp > now_value + timedelta(seconds=allowed_skew_seconds):
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.FUTURE_EVENT.value
        result.reason = "Event timestamp exceeds the configured skew allowance."
        result.checks["freshness"] = False
        return result

    last_sequence = store.last_sequence(scope)
    if last_sequence is not None and sequence <= last_sequence:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.SEQUENCE_VIOLATION.value
        result.reason = "Sequence number violates scoped ordering requirements."
        result.checks["sequence_valid"] = False
        return result

    if not context:
        result.valid = False
        result.status = "INVALID"
        result.failure_code = SecurityErrorCode.CONTEXT_MISMATCH.value
        result.reason = "Missing event context binding."
        result.checks["context_valid"] = False
        return result

    result.checks = {
        "duplicate_event": True,
        "nonce_reuse": True,
        "freshness": True,
        "sequence_valid": True,
        "context_valid": True,
    }
    store.record(event=event, scope=scope)
    return result
