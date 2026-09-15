"""Essential manifest smoke tests."""

from mirad.security.identity import ArtifactType
from mirad.security.manifest import create_manifest, verify_manifest_files


def _write_tree(root, files):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_deterministic_manifest(tmp_path):
    root = tmp_path / "artifact"
    _write_tree(root, {"weights.bin": b"weights", "config.json": b'{"a":1}'})

    m1 = create_manifest(
        artifact_type=ArtifactType.MODEL,
        artifact_id="model-a",
        version="1.0.0",
        format="pytorch",
        root_path=root,
    )
    m2 = create_manifest(
        artifact_type=ArtifactType.MODEL,
        artifact_id="model-a",
        version="1.0.0",
        format="pytorch",
        root_path=root,
    )
    assert m1 == m2
    assert m1["manifest_digest"].startswith("sha256:")


def test_modified_file_detected(tmp_path):
    root = tmp_path / "artifact"
    _write_tree(root, {"data.txt": b"original"})

    manifest = create_manifest(
        artifact_type=ArtifactType.DATASET,
        artifact_id="ds-1",
        version="1.0.0",
        format="directory",
        root_path=root,
    )

    valid, failures = verify_manifest_files(manifest, root)
    assert valid is True
    assert failures == []

    (root / "data.txt").write_bytes(b"tampered")
    valid, failures = verify_manifest_files(manifest, root)
    assert valid is False
    assert "MANIFEST_CONTENT_MISMATCH" in failures
