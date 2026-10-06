import io
import sys
import types
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
class _SS(dict):
    pass
_fake_st = types.SimpleNamespace(session_state=_SS(), set_page_config=lambda **_: None, runtime=types.SimpleNamespace(exists=lambda: False))
sys.modules.setdefault("streamlit", _fake_st)
sys.path.insert(0, str(ROOT))
SRC = (ROOT / "trustcv_final.py").read_text(encoding="utf-8")
SRC = SRC[:SRC.index("# UI\n")]
NS = {"__file__": str(ROOT / "trustcv_final.py"), "st": _fake_st}
exec(compile(SRC, str(ROOT / "trustcv_final.py"), "exec"), NS)
parse_dataset = NS["parse_dataset"]
analyze_reports = NS["analyze_reports"]

class Upload:
    def __init__(self, name, raw): self.name, self._raw = name, raw
    def getvalue(self): return self._raw

def image_bytes(fmt="JPEG", size=(96, 96)):
    b = io.BytesIO(); Image.new("RGB", size, (120, 150, 180)).save(b, fmt); return b.getvalue()

def make_zip(kind):
    imgs = [(f"contributor_{i%2}/img_{i}.jpg", image_bytes(size=(96+i,96))) for i in range(8)]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        if kind == "direct":
            for n,b in imgs: z.writestr(n,b)
        elif kind == "nested":
            inner=io.BytesIO()
            with zipfile.ZipFile(inner,"w",zipfile.ZIP_DEFLATED) as iz:
                for n,b in imgs: iz.writestr(n,b)
            z.writestr("images.zip",inner.getvalue())
        elif kind == "extensionless":
            for i,(_,b) in enumerate(imgs): z.writestr(f"src/img_{i}",b)
        elif kind == "mislabelled":
            for i,(_,b) in enumerate(imgs): z.writestr(f"src/img_{i}.data",b)
    return out.getvalue()

def test_direct_nested_and_mislabelled():
    for kind in ("direct", "nested", "extensionless", "mislabelled"):
        d=parse_dataset(Upload("images.zip",make_zip(kind)))
        assert len(d["images"]) == 8, (kind, len(d["images"]))
        assert len(analyze_reports(d)) == 8

def test_corrupt_zip_fails_cleanly():
    try:
        parse_dataset(Upload("images.zip", b"not a zip"))
    except ValueError as e:
        assert "Invalid or corrupt ZIP dataset" in str(e)
    else:
        raise AssertionError("corrupt ZIP was accepted")
