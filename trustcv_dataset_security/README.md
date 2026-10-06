# TrustCV — Dataset Authority & Security Assurance

**Smart India Hackathon 2026 — SIH-26228**

This build focuses on the training-data integrity and analyst-facing
assurance part of the problem statement. It is intentionally a
**dataset-only** application.

The application is designed to operate locally/offline and combines
access control, cryptographic identity, signed provenance, archive
integrity, image-quality screening, duplicate detection, anomaly
screening, spectral-style analysis, distribution-shift analysis,
source-level aggregation and a tamper-evident audit ledger.

## Security pipeline

```text
Authority Passkey
       |
       v
Secure Dataset Intake
       |
       v
SHA-256 Dataset Identity
       |
       +--> ZIP CRC / Unsafe Extension Screen
       |
       +--> Readable Image Extraction
       |
       +--> Signed Dataset Manifest
       |       |
       |       +--> Digital Signature
       |       +--> Source / Owner / Version / Time
       |
       +--> Exact + Near Duplicate Detection
       |
       +--> EXIF + Image Quality
       |
       +--> Visual Outlier Screening
       |
       +--> SVD Spectral Screening
       |
       +--> Per-image Distribution Shift
       |
       +--> Contributor / Source Aggregation
       |
       v
Image Risk Matrix
       |
       v
Accept / Review / Quarantine
       |
       v
Hash-linked Audit Ledger
```

## Key capabilities

- Authority passkey before dataset onboarding
- SHA-256 fingerprinting
- Signed dataset manifest verification
- RSA-PSS / SHA-256 signature checking
- Dataset provenance checks
- ZIP CRC integrity validation
- Unsafe archive-member extension screening
- Readable image extraction checks
- Exact duplicate detection
- Perceptual near-duplicate screening
- EXIF orientation review
- Image quality and acquisition checks
- Isolation Forest visual outlier screening
- SVD spectral-style screening
- Per-image distribution-shift evidence
- Per-image quality/anomaly/spectral/shift/review confidence
- Accept / Review / Quarantine disposition
- Contributor/source-level evidence aggregation
- CSV/Excel duplicate and missing-value screening
- Hash-linked, tamper-evident audit trail
- Offline/local operation

## Authority passkey

Create the local authority credential once:

```bash
python init_passkey.py
```

The application stores only a SHA-256 hash of the passkey in
`TrustCV_access_config.json`.

The passkey is required before the dataset uploader is enabled.

For the prototype this is a local access-control mechanism. A production
system should replace or augment it with managed identities, certificates,
hardware-backed credentials, or an organization-controlled authorization
service.

`TrustCV_access_config.json` is ignored by Git and must never be committed.

## Signed dataset manifest

Create a manifest for one exact dataset file:

```bash
python sign_dataset.py "C:\\path\\to\\dataset.zip" "Demo Source" "Approved Demo Owner" "1.0.0"
```

The command creates:

- `trustcv_dataset_manifest.json`
- `trustcv_dataset_private_key.pem`

Upload the dataset and the signed manifest to the dashboard.

Never commit the private key.

## Per-image risk evidence

The Image Risk Matrix reports for every image:

- Quality confidence
- Visual anomaly confidence
- Spectral screening confidence
- Shift distance
- Shift confidence
- Shift status
- Exact duplicate status
- Near-duplicate status
- Orientation status
- Review confidence
- Severity
- Recommended disposition

The confidence values are screening-confidence values, **not calibrated
probabilities of malicious compromise**.

## SVD spectral screen

The prototype includes a spectral-style SVD layer because the reference
Spectral Signatures research direction analyzes abnormal structure in a
representation space.

This build applies SVD to a compact handcrafted visual descriptor.
It should therefore be described as **paper-inspired spectral screening**,
not as an exact reproduction of a learned-representation method.

## Contributor aggregation

When the ZIP contains source folders such as:

```text
contributor_A/image01.jpg
contributor_A/image02.jpg
contributor_B/image03.jpg
```

TrustCV aggregates image-level evidence into source-level statistics.

If contributor metadata is not available, the application reports
`ROOT / UNKNOWN SOURCE`.

## Analyst disposition

```text
ACCEPT      = no high-priority screening signal
REVIEW      = evidence requires analyst attention
QUARANTINE  = multiple screening signals justify isolation for investigation
```

These are prototype dispositions, not automatic proof of an attack.

## Run locally

```bash
pip install -r requirements.txt
python init_passkey.py
streamlit run trustcv_final.py
```

## Supported dataset inputs

- ZIP containing JPG/JPEG/PNG/BMP/WEBP images
- CSV
- XLSX / XLS
- Standalone images

The ZIP inventory also records candidate annotation files useful for
common vision-dataset structures.

## Offline / air-gapped operation

The core application performs hashing, signature verification, parsing,
image analysis, scoring and audit generation locally and has no required
external API dependency.

## Limitations

- Visual outliers are statistical signals, not proof of poisoning.
- Distribution shift can be legitimate operational drift.
- Contributor aggregation depends on available metadata.
- The passkey is local prototype access control, not enterprise IAM.
- The SVD layer is a compact-descriptor spectral screen and not an exact
  learned-representation implementation.
- Confidence values are not calibrated attack probabilities.

## Repository safety

Do not commit:

- `TrustCV_access_config.json`
- `*.pem`
- `*.key`
- runtime logs
- generated artifacts

## V4 presentation / analyst features

- Contributor Passkey naming for professional onboarding
- Animated 3D security telemetry cube
- Interactive 3D anomaly/shift/spectral evidence map
- Technical control-plane presentation instead of a long theory block
- PDF analyst report export
- CSV image-risk matrix export
- JSON evidence export
- ZIP evidence bundle export


## V5 final upgrades

- Contributor Passkey wording
- Detailed SHA-256 telemetry and independent hash recomputation
- Provenance and signer-key fingerprint telemetry
- Expanded archive cybersecurity checks
- Local proof-of-work blockchain-style evidence ledger
- Animated 3D TrustCV security core
- Animated 3D blockchain stack
- Interactive 3D evidence maps
- PDF, CSV, JSON, HTML and ZIP exports

The blockchain component is a local proof-of-work hash chain for
tamper-evident evidence. It is not a decentralised public blockchain or a
consensus network.


## MIRAD-derived dataset security integration

This V5 build keeps the existing TrustCV dataset UI and analysis workflow intact,
while adding a separate cryptographic assurance backend derived from the security
patterns implemented in the MIRAD prototype.

**Dataset scope only:** this integration does not authenticate, fingerprint or
provenance-bind models. Model upload/integrity is intentionally outside this
application.

At the dataset trust boundary it adds:

- trusted Ed25519 dataset signing key and trust-anchor binding
- canonical SHA-256 identity for security records
- signed dataset provenance containing dataset digest, manifest digest,
  contributor/source metadata and analysis/evidence digests
- strict provenance schema and signature verification
- freshness, nonce, event-ID and sequence replay protection
- append-only SHA-256 audit chain with previous-hash linkage
- signed audit checkpoint bound to the current audit head
- structured verification results for provenance, replay, audit and checkpoint
- security evidence included automatically in the existing Evidence ZIP/JSON
  exports

Runtime security records are stored under:

```text
%USERPROFILE%\.trustcv_dataset_security\
```

The private Ed25519 key is kept outside the repository. The application never
uses a supplied public key as a trust anchor merely because its signature is
mathematically valid.

The existing RSA-PSS manifest format remains readable for backward
compatibility, but it is reported as `REVIEW` unless the new trusted Ed25519
dataset trust anchor is used.

### Security flow

```text
Contributor Passkey
        |
        v
Dataset Bytes
        |
        +--> SHA-256 Dataset Identity
        |
        +--> Signed Manifest / Trusted Ed25519 Key
        |
        +--> Existing Dataset Analysis
        |      (quality / duplicates / anomaly / SVD / shift / source evidence)
        |
        v
Canonical Analysis + Evidence Digests
        |
        v
Signed Dataset Provenance
        |
        +--> Schema / Signature / Trust Anchor
        +--> Freshness / Nonce / Event ID / Sequence
        |
        v
Hash-Linked Dataset Audit Event
        |
        v
Signed Audit Checkpoint
```

The resulting records are dataset-specific and contain no model provenance
fields. The controls are tamper-evident and fail closed on signature,
trust-anchor, freshness, replay, sequence or audit-chain failures.


### Local intake attestation
If no external signed dataset manifest is supplied, TrustCV generates a local Ed25519 intake attestation bound to the exact uploaded SHA-256. This is an internal trust anchor, not an external vendor claim. Inspectable runtime artifacts are written to `demo_output/dataset_security/`.
