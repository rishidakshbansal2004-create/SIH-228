# MIRAD Security / Trust Subsystem

MIRAD provides a compact offline security and trust layer for model and dataset integrity workflows. It is self-contained and does not require external services, cloud infrastructure, unsafe model loading, or network access.

## Included capabilities

- Deterministic SHA-256 hashing and canonical serialization
- Manifest generation and verification for artifact directories
- Artifact registration and verification against trusted references
- Ed25519 key generation, trusted key registration, and signature verification
- Provenance signing and verification for lineage events
- Replay protection with freshness, uniqueness, and sequence checks
- Append-only audit-chain style event logging and checkpointing
- Evidence-envelope validation and structured findings
- Policy compatibility checks that reject downgrade attempts
- Offline CLI demo and local attack-lab exercises

## Architecture and trust model

- Trust is explicit: an artifact only becomes trusted after registration in a trust anchor store.
- Raw Ed25519 signature validity is distinct from trusted verification. Trusted provenance and checkpoints require a registered active key, matching trust anchor, supported policy, and valid signature.
- Security results are structured objects rather than bare booleans.
- Replay protection binds event identity, nonce, UTC freshness, sequence, and an expected context.
- Evidence integrity is recomputed and compared. A valid unsigned envelope does not imply a trusted evidence producer.
- Findings distinguish severity, confidence, and analyst disposition, including `INCONCLUSIVE`.

## Threat model and assumptions

The subsystem detects artifact modification/substitution, provenance and signature tampering, untrusted or revoked signers, trust-anchor confusion, replay and ordering abuse, audit-chain tampering, checkpoint mismatch, evidence tampering, policy downgrade, and manifest traversal.

It assumes the trust-anchor store, signing-key lifecycle, and replay state are controlled by the deployment. The local audit implementation is tamper-evident hash-chained storage; it does not by itself provide absolute immutability, deletion resistance, or availability guarantees. Production deployments may add durable external anchoring or KMS/HSM controls without changing this offline baseline.

## Security demonstration

The security demonstration uses deterministic offline fixtures, computes real artifact and configuration digests, signs the complete provenance record with a development-only Ed25519 key, and persists structured results for a future UI. The original event is trusted; tampered artifacts, signed fields, outputs, replayed events, and audit records are rejected or quarantined.

```bash
python -m mirad.security.cli
py -m mirad.security.cli --json
```

It writes `demo_output/provenance.json`, `audit_log.json`, `checkpoint.json`, `verification_result.json`, `security_demo_report.json`, and records under `demo_output/attacks/`.

The JSON report is the future UI contract: provenance, verification, audit, checkpoint, attacks, findings, and overall assurance are available without parsing terminal output.

## Automated security tests

```bash
py -m pytest -q
```

## Attack lab

The attack lab emits 33 structured scenarios with `attack_id`, mutation, expected and actual results, pass status, and evidence. Run it directly with:

```bash
py -m mirad.security.attack_lab
```

## Status

The implementation is offline and air-gapped. Development private keys are generated in memory and never written to output. Production deployments should use a KMS/HSM or secure secret manager. Hash chaining is tamper-evident rather than absolute immutability, and artifact integrity does not establish behavioral safety or contributor intent.
