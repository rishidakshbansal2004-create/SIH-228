"""Deterministic multi-file artifact manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from mirad.security.constants import MANIFEST_SCHEMA_VERSION, SUPPORTED_MANIFEST_SCHEMA_VERSIONS
from mirad.security.errors import SecurityErrorCode
from mirad.security.hashing import hash_canonical_object, hash_file
from mirad.security.identity import ArtifactType


def normalize_relative_path(path: str | Path) -> str:
    """Normalize a relative path for manifest entries (forward slashes, no absolutes)."""
    text = str(path).replace("\\", "/")
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise ValueError(f"Absolute paths are not permitted in manifests: {path}")
    parts: list[str] = []
    for part in text.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"Parent path segments are not permitted: {path}")
        parts.append(part)
    return "/".join(parts)


def _iter_files(root: Path) -> Iterable[tuple[str, Path]]:
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            full_path = Path(dirpath) / filename
            rel = normalize_relative_path(full_path.relative_to(root))
            yield rel, full_path


def create_manifest(
    *,
    artifact_type: ArtifactType,
    artifact_id: str,
    version: str,
    format: str,
    root_path: str | Path,
    manifest_version: str = MANIFEST_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a deterministic manifest for all files under *root_path*."""
    if manifest_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported manifest schema version: {manifest_version}")

    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact root not found: {root}")

    files: list[dict[str, Any]] = []
    for rel_path, file_path in sorted(_iter_files(root), key=lambda item: item[0]):
        files.append(
            {
                "normalized_path": rel_path,
                "size": file_path.stat().st_size,
                "digest": hash_file(file_path),
            }
        )

    manifest: dict[str, Any] = {
        "manifest_version": manifest_version,
        "artifact_type": artifact_type.value,
        "artifact_id": artifact_id,
        "version": version,
        "format": format,
        "files": files,
    }
    manifest["manifest_digest"] = hash_canonical_object(manifest)
    return manifest


def verify_manifest_files(
    manifest: dict[str, Any],
    root_path: str | Path,
) -> tuple[bool, list[str]]:
    """
    Verify on-disk files against a manifest.

    Returns ``(valid, failure_codes)``.
    """
    failures: list[str] = []

    if manifest.get("manifest_version") not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        failures.append(SecurityErrorCode.UNSUPPORTED_SCHEMA_VERSION.value)
        return False, failures

    expected_digest = manifest.get("manifest_digest")
    manifest_without_digest = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    computed_digest = hash_canonical_object(manifest_without_digest)
    if expected_digest != computed_digest:
        failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
        return False, failures

    root = Path(root_path)
    if not root.is_dir():
        failures.append(SecurityErrorCode.ARTIFACT_NOT_FOUND.value)
        return False, failures

    entries = manifest.get("files")
    if not isinstance(entries, list):
        failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
        return False, failures

    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
            continue

        rel_path = entry.get("normalized_path")
        expected_file_digest = entry.get("digest")
        expected_size = entry.get("size")

        if not isinstance(rel_path, str):
            failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
            continue

        try:
            normalized_path = normalize_relative_path(rel_path)
        except ValueError:
            failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
            continue
        if normalized_path != rel_path:
            failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
            continue

        if rel_path in seen_paths:
            failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
            continue
        seen_paths.add(rel_path)

        file_path = root / Path(*normalized_path.split("/"))
        try:
            if file_path.is_symlink() or file_path.resolve().relative_to(root.resolve()) is None:
                failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
                continue
        except ValueError:
            failures.append(SecurityErrorCode.MANIFEST_INVALID.value)
            continue
        if not file_path.is_file():
            failures.append(SecurityErrorCode.MANIFEST_CONTENT_MISMATCH.value)
            continue

        actual_size = file_path.stat().st_size
        if expected_size != actual_size:
            failures.append(SecurityErrorCode.MANIFEST_CONTENT_MISMATCH.value)

        actual_digest = hash_file(file_path)
        if expected_file_digest != actual_digest:
            failures.append(SecurityErrorCode.MANIFEST_CONTENT_MISMATCH.value)

    # Detect unexpected files when policy requires full directory match.
    on_disk = {rel for rel, _ in _iter_files(root)}
    if on_disk != seen_paths:
        failures.append(SecurityErrorCode.MANIFEST_CONTENT_MISMATCH.value)

    return len(failures) == 0, sorted(set(failures))
