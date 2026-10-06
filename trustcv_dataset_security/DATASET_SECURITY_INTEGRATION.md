# TrustCV Dataset Security Integration

This module is the dataset-side security boundary for SIH-26228.

It is intentionally segregated from model upload/integrity. The model-side
assurance layer can consume the dataset identity later, but this application
does not add model fields or model verification logic.

## Controls

1. **Dataset identity** — SHA-256 over the exact uploaded bytes.
2. **Trusted signer** — Ed25519 key registered as a local trust anchor.
3. **Signed dataset provenance** — canonical record binds dataset digest,
   manifest digest, source/owner/version and analysis/evidence digests.
4. **Replay protection** — event ID, nonce, freshness window, context and
   monotonic provenance sequence.
5. **Audit integrity** — append-only hash chain with sequence and previous hash.
6. **Checkpoint** — Ed25519-signed anchor of the current audit chain head.
7. **Structured verification** — separate schema, signature, trust-anchor,
   binding, replay, audit and checkpoint results.
8. **Evidence export** — the existing TrustCV Evidence ZIP contains the new
   dataset security records.

## MIRAD relationship

The implementation adapts the cryptographic and assurance patterns from the
MIRAD security prototype: canonicalization, SHA-256 identity, trusted Ed25519
verification keys, signed provenance, replay state, hash-linked audit records
and signed checkpoints.

This is an adaptation to the SIH dataset boundary, not a copy of the full
MIRAD ML/model pipeline.

## Files generated at runtime

```text
%USERPROFILE%\.trustcv_dataset_security\
  dataset_signing_private.pem
  dataset_trust_anchor.json
  dataset_security_state.json
  dataset_security_audit.jsonl
  dataset_security_checkpoint.json
```

These files are deliberately outside the Git repository.

## Trust interpretation

- A SHA-256 match establishes byte identity.
- A valid Ed25519 signature establishes cryptographic authenticity of the
  signed record **only when the signing key is trusted**.
- A digest mismatch establishes difference; it does not by itself establish
  malicious intent.
- Statistical dataset findings remain screening evidence and must not be
  interpreted as proof that a contributor is malicious.
- A valid audit chain is tamper-evident; it is not an availability guarantee
  or an immutable distributed ledger.
