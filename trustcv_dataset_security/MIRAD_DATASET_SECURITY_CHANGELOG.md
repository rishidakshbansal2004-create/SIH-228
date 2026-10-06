# Dataset Security Modulation — V1

Base: TrustCV_DATASET_SECURITY_FINAL_V5.

The existing TrustCV dataset UI and dataset-analysis workflow are preserved.
The modulation adds a dataset-only MIRAD-derived security backend.

Added:
- Trusted Ed25519 dataset signer/trust anchor.
- Canonical SHA-256 dataset identity for security records.
- Signed dataset provenance (dataset/manifest/source/version/analysis/evidence only).
- Strict provenance verification and trust-anchor binding.
- Freshness, nonce, event-ID and sequence replay protection.
- Hash-linked append-only dataset audit events.
- Ed25519-signed audit checkpoints.
- Structured security verification results.
- Automatic inclusion of dataset security records in the existing Evidence ZIP/JSON.
- Offline CLI smoke demo (`dataset_security_demo.py`).

Intentionally excluded:
- Model upload/integrity logic.
- Model provenance fields.
- Changes to the existing visual UI.
- Cloud services, external APIs or mandatory blockchain infrastructure.

## V3 dataset ingestion hardening
- Recursive nested ZIP image discovery up to bounded depth 3.
- Pillow content sniffing for extensionless/mislabelled images.
- TIFF/TIFF/GIF/AVIF/HEIC/HEIF image suffix support.
- No UI changes.
