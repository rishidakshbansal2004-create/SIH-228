"""Essential canonicalization smoke tests."""

from mirad.security.canonical import canonicalize, canonicalize_to_text
from mirad.security.hashing import hash_canonical_object


def test_same_object_same_bytes():
    obj = {"b": 2, "a": 1, "nested": {"z": True, "y": None}}
    assert canonicalize(obj) == canonicalize({"a": 1, "b": 2, "nested": {"y": None, "z": True}})


def test_key_order_independent():
    first = canonicalize_to_text({"z": 3, "a": 1, "m": 2})
    second = canonicalize_to_text({"a": 1, "m": 2, "z": 3})
    assert first == second
    assert first == '{"a":1,"m":2,"z":3}'


def test_hash_canonical_object_deterministic():
    obj = {"model_id": "detector", "threshold": 0.5, "tags": ["a", "b"]}
    assert hash_canonical_object(obj) == hash_canonical_object({"tags": ["a", "b"], "model_id": "detector", "threshold": 0.5})
