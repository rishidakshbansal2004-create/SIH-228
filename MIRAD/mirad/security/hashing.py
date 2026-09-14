"""SHA-256 hashing utilities with a stable digest representation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, BinaryIO

from mirad.security.canonical import canonicalize
from mirad.security.constants import CANONICALIZATION_VERSION, DIGEST_PREFIX, HASH_CHUNK_SIZE


def _format_digest(digest_bytes: bytes) -> str:
    return f"{DIGEST_PREFIX}{digest_bytes.hex()}"


def parse_digest(digest: str) -> bytes:
    """Validate and parse a ``sha256:<hex>`` digest string."""
    if not digest.startswith(DIGEST_PREFIX):
        raise ValueError(f"Digest must start with '{DIGEST_PREFIX}'")
    hex_part = digest[len(DIGEST_PREFIX) :]
    if len(hex_part) != 64:
        raise ValueError("SHA-256 digest must be 64 hex characters")
    try:
        return bytes.fromhex(hex_part)
    except ValueError as exc:
        raise ValueError("Digest contains invalid hex characters") from exc


def hash_bytes(data: bytes) -> str:
    """Hash raw bytes and return ``sha256:<hex>``."""
    return _format_digest(hashlib.sha256(data).digest())


def hash_text(text: str) -> str:
    """Hash UTF-8 encoded text."""
    return hash_bytes(text.encode("utf-8"))


def hash_stream(stream: BinaryIO, *, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Hash a binary stream using bounded memory."""
    hasher = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
    return _format_digest(hasher.digest())


def hash_file(path: str | Path, *, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Hash a file from disk using streaming reads."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("rb") as handle:
        return hash_stream(handle, chunk_size=chunk_size)


def hash_canonical_object(
    obj: Any,
    *,
    canonicalization_version: str = CANONICALIZATION_VERSION,
) -> str:
    """Hash a structured object after deterministic canonical serialization."""
    canonical_bytes = canonicalize(obj, version=canonicalization_version)
    return hash_bytes(canonical_bytes)
