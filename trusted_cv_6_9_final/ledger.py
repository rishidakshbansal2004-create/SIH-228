"""Phase 9: append-only tamper-evident inference ledger."""
import json
from pathlib import Path
from crypto_utils import canonical_json,sha256_bytes,utc_now

class AuditLedger:
    def __init__(self,path="audit/audit_ledger.jsonl"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.path.touch(exist_ok=True)
    def entries(self):
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():yield json.loads(line)
    def last_hash(self):
        last=None
        for last in self.entries():pass
        return last["record_hash"] if last else "0"*64
    def append(self,event):
        previous=self.last_hash(); index=sum(1 for _ in self.entries())
        body={"index":index,"timestamp_utc":utc_now(),"previous_hash":previous,"event":event}
        body["record_hash"]=sha256_bytes(canonical_json(body))
        with self.path.open("a",encoding="utf-8") as f:f.write(json.dumps(body,sort_keys=True)+"\n")
        return body
    def verify(self):
        previous="0"*64
        for i,r in enumerate(self.entries(),1):
            if r.get("previous_hash")!=previous:return False,f"broken previous_hash at line {i}"
            actual=r.get("record_hash"); body=dict(r); body.pop("record_hash",None)
            if sha256_bytes(canonical_json(body))!=actual:return False,f"tampered record at line {i}"
            previous=actual
        return True,"ledger valid"
