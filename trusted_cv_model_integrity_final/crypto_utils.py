"""TrustCV cryptographic helpers delegated to MIRAD's canonical hashing layer."""
from __future__ import annotations

from mirad.security.canonical import canonicalize
from mirad.security.hashing import hash_bytes as _mirad_hash_bytes
from mirad.security.hashing import hash_canonical_object, hash_file as _mirad_hash_file


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj):
    return canonicalize(obj)


def sha256_bytes(data):
    """Return the historical TrustCV hex form for UI/backward compatibility."""
    return _mirad_hash_bytes(data).split(":", 1)[1]


def sha256_file(path):
    """Return the historical TrustCV hex form using MIRAD streaming SHA-256."""
    return _mirad_hash_file(path).split(":", 1)[1]


def mirad_sha256_bytes(data):
    return _mirad_hash_bytes(data)


def mirad_sha256_file(path):
    return _mirad_hash_file(path)


def mirad_hash_object(obj):
    return hash_canonical_object(obj)
