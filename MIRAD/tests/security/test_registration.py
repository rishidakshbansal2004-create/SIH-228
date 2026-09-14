"""Essential registration and verification smoke tests."""

from mirad.security.identity import ArtifactIdentity, ArtifactType
from mirad.security.manifest import create_manifest
from mirad.security.registration import register_artifact
from mirad.security.trust_anchor import TrustAnchorStore
from mirad.security.verification import verify_artifact


def test_register_and_verify_single_file(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx-bytes")

    store = TrustAnchorStore()
    registration = register_artifact(
        trust_store=store,
        artifact_type=ArtifactType.MODEL,
        artifact_id="detector-v1",
        version="1.0.0",
        format="onnx",
        artifact_path=model_path,
    )

    candidate = ArtifactIdentity(
        artifact_type=ArtifactType.MODEL,
        artifact_id="detector-v1",
        version="1.0.0",
        artifact_digest=registration.identity.artifact_digest,
        format="onnx",
    )
    result = verify_artifact(candidate=candidate, trust_store=store, artifact_path=model_path)
    assert result.valid is True
    assert result.status.value == "VERIFIED"


def test_modified_artifact_mismatch(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"original")

    store = TrustAnchorStore()
    registration = register_artifact(
        trust_store=store,
        artifact_type=ArtifactType.MODEL,
        artifact_id="detector-v1",
        version="1.0.0",
        format="onnx",
        artifact_path=model_path,
    )

    model_path.write_bytes(b"substituted")
    candidate = ArtifactIdentity(
        artifact_type=ArtifactType.MODEL,
        artifact_id="detector-v1",
        version="1.0.0",
        artifact_digest=registration.identity.artifact_digest,
        format="onnx",
    )
    result = verify_artifact(candidate=candidate, trust_store=store, artifact_path=model_path)
    assert result.valid is False
    assert "MODEL_DIGEST_MISMATCH" in result.failure_codes
    assert "MODEL_SUBSTITUTION_DETECTED" in result.failure_codes


def test_multi_file_manifest_registration(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "labels.json").write_bytes(b"[]")
    (root / "images").mkdir()
    (root / "images" / "a.jpg").write_bytes(b"jpg")

    store = TrustAnchorStore()
    registration = register_artifact(
        trust_store=store,
        artifact_type=ArtifactType.DATASET,
        artifact_id="coco-mini",
        version="1.0.0",
        format="coco",
        artifact_path=root,
    )

    manifest = create_manifest(
        artifact_type=ArtifactType.DATASET,
        artifact_id="coco-mini",
        version="1.0.0",
        format="coco",
        root_path=root,
    )
    result = verify_artifact(
        candidate=registration.identity,
        trust_store=store,
        artifact_path=root,
        manifest=manifest,
    )
    assert result.valid is True
