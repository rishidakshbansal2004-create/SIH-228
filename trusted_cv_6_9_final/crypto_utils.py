"""Utilities used by Phase 9 audit records. Model signing belongs to Phase 4;
Phase 9 only records hashes/commitments and maintains an audit chain."""
import base64, hashlib, json
from datetime import datetime, timezone

def utc_now(): return datetime.now(timezone.utc).isoformat()
def canonical_json(obj): return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
