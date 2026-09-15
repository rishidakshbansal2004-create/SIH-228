# TrustCV ONE FINAL — Security Grade Prototype

This version fixes the prior verification failure semantics.

### Model verification
- SHA-256 identity
- optional signed manifest
- vendor/version/provenance checks
- safe PyTorch `weights_only=True` inspection
- explicit execution gate
- exact load error when an authorized model cannot load

IMPORTANT: TrustCV intentionally does not deserialize arbitrary uploaded `.pt` objects before
the model has passed the authenticity gate. A `.pt` file is commonly a pickle-based artifact,
and unsafe deserialization can execute Python code. A safe inspection rejection therefore means
"not safely inspectable by this adapter", not automatically "corrupt" or "malicious".

### Dataset
CSV, Excel, image ZIP, individual image.
Visual orientation, duplicate and visual outlier screens.

### Input
Test image, hash, model inference (authorized/existing model), reference OOD and shift.

### Reliability
Benign brightness and flip consistency.

### Audit
Hash-linked local audit ledger with export.

### Run
```powershell
python -m pip install -r requirements.txt
python -m streamlit run trustcv_final.py --server.port 8503
```

Open:
http://localhost:8503

Your existing `C:\Users\vasan\yolo11n.pt` is automatically detected as the local demo baseline.

For an uploaded model to be executed, supply a signed manifest containing:
expected_sha256, vendor, version, provenance, signature, public_key_pem.

Hash PASS alone never means authenticity PASS.


## Create a signed demo manifest

```powershell
python sign_model.py C:\Users\vasan\yolo11n.pt "Approved Demo Vendor" "1.0.0"
```

This creates:
- `trustcv_model_manifest.json` — upload this with the model
- `trustcv_private_key.pem` — KEEP PRIVATE

An uploaded model without a trusted manifest remains in **EXECUTION LOCKED** state.
That is intentional: TrustCV must not blindly execute an untrusted `.pt` checkpoint.
