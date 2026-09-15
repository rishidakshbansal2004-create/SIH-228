"""Essential hashing smoke tests."""

import io

from mirad.security.hashing import hash_bytes, hash_file, hash_stream, hash_text, parse_digest


def test_hash_bytes_known_vector():
    # Empty string SHA-256
    assert hash_bytes(b"") == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_text_deterministic():
    assert hash_text("mirad") == hash_text("mirad")
    assert hash_text("mirad") != hash_text("MIRAD")


def test_hash_stream_matches_bytes():
    data = b"streamed artifact bytes"
    stream = io.BytesIO(data)
    assert hash_stream(stream) == hash_bytes(data)


def test_hash_file(tmp_path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"file content")
    assert hash_file(file_path) == hash_bytes(b"file content")


def test_parse_digest_roundtrip():
    digest = hash_bytes(b"abc")
    assert parse_digest(digest) == bytes.fromhex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
