# TrustCV Dataset Security V4 — Verification Report

## Scope
Dataset-only TrustCV application. Existing visual UI was preserved; the changes are limited to dataset ingestion robustness, state safety, and the MIRAD-derived dataset security layer.

## Automated checks
- Python syntax compilation: PASS
- `pytest`: 3/3 PASS
- Direct ZIP image ingestion: PASS (8/8)
- Nested ZIP image ingestion: PASS (8/8)
- Extensionless image ingestion: PASS (8/8)
- Mislabelled image ingestion: PASS (8/8)
- Real uploaded image samples available in the workspace: PASS (9/9 decoded)
- Corrupt ZIP handling: PASS (fails cleanly with a structured ValueError)
- Ed25519 provenance signing/verification: PASS
- Dataset digest binding: PASS
- Replay protection: PASS (first accepted, duplicate rejected)
- Audit hash chain: PASS
- Signed audit checkpoint: PASS
- Full source compile (`compileall`): PASS

## Runtime compatibility note
The final ZIP intentionally does not bundle a platform-specific Python environment. Install the requirements on the target Windows machine with its local Python installation. Streamlit's documented default per-file upload limit is 200 MB; this build configures the local app limit to 500 MB.

## Ingestion behavior
ZIP scanning is bounded to avoid unbounded archive traversal. Images are decoded by content rather than trusting filename extensions, nested ZIPs are supported to depth 4, and truncated image decoding receives an isolated fallback. Obvious metadata/text and blocked executable/script members are not treated as images.
