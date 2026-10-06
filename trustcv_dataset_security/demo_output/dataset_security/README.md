# Dataset Security Artifacts

This directory is populated by the TrustCV dataset-only security layer after a dataset is processed.

Runtime-generated files:
- `provenance.json` — signed dataset provenance bound to the exact dataset SHA-256.
- `audit_log.json` — current dataset security audit chain.
- `checkpoint.json` — signed checkpoint bound to the audit head.
- `verification_result.json` — structured provenance, replay, audit and checkpoint verification.

No private signing key is stored in this directory.
