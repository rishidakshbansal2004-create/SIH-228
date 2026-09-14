"""Deterministic canonical serialization for cryptographic hashing and signing."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from mirad.security.constants import (
    CANONICALIZATION_VERSION,
    SUPPORTED_CANONICALIZATION_VERSIONS,
)
from mirad.security.errors import SecurityErrorCode


class CanonicalizationError(ValueError):
    """Raised when an object cannot be canonically serialized."""

    def __init__(self, message: str, code: SecurityErrorCode = SecurityErrorCode.CANONICALIZATION_ERROR):
        super().__init__(message)
        self.code = code


def canonicalize(obj: Any, *, version: str = CANONICALIZATION_VERSION) -> bytes:
    """Return deterministic UTF-8 bytes for *obj* using versioned canonical JSON rules."""
    if version not in SUPPORTED_CANONICALIZATION_VERSIONS:
        raise CanonicalizationError(
            f"Unsupported canonicalization version: {version}",
            SecurityErrorCode.UNSUPPORTED_CANONICALIZATION_VERSION,
        )
    return _encode_value(obj).encode("utf-8")


def canonicalize_to_text(obj: Any, *, version: str = CANONICALIZATION_VERSION) -> str:
    """Return deterministic canonical JSON text."""
    return canonicalize(obj, version=version).decode("utf-8")


def _encode_value(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, Decimal):
        return _encode_decimal(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return _encode_object(value)
    raise CanonicalizationError(
        f"Unsupported canonical type: {type(value).__name__}",
        SecurityErrorCode.CANONICALIZATION_ERROR,
    )


def _encode_object(obj: dict[Any, Any]) -> str:
    if not isinstance(obj, dict):
        raise CanonicalizationError("Expected mapping", SecurityErrorCode.CANONICALIZATION_ERROR)
    keys = sorted(str(k) for k in obj.keys())
    if len(keys) != len(obj):
        raise CanonicalizationError("Duplicate canonical keys detected", SecurityErrorCode.CANONICALIZATION_ERROR)
    parts: list[str] = []
    for key in keys:
        original_key = key
        for candidate in obj:
            if str(candidate) == key:
                original_key = candidate
                break
        parts.append(f"{_encode_string(key)}:{_encode_value(obj[original_key])}")
    return "{" + ",".join(parts) + "}"


def _encode_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _encode_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise CanonicalizationError(
            "Non-finite floats are not canonicalizable",
            SecurityErrorCode.CANONICALIZATION_ERROR,
        )
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0"
    text = format(value, ".17g")
    if "e" not in text and "E" not in text and "." not in text and "inf" not in text:
        text = f"{text}.0"
    return text


def _encode_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise CanonicalizationError(
            "Non-finite decimals are not canonicalizable",
            SecurityErrorCode.CANONICALIZATION_ERROR,
        )
    normalized = format(value.normalize(), "f")
    if "." not in normalized:
        normalized = f"{normalized}.0"
    return normalized
