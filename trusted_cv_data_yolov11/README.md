# TrustCV — Trustworthy Computer Vision Integrity Assurance

> Smart India Hackathon 2026 — SIH-228

TrustCV is a security-oriented Computer Vision assurance layer designed to verify AI models, datasets, and inference inputs before they are trusted in a multi-contributor pipeline.

The system combines cryptographic verification, provenance checks, dataset integrity analysis, anomaly screening, Out-of-Distribution (OOD) detection, distribution-shift analysis, reliability testing, and audit logging into a unified security workflow.

---

## Problem Statement

Modern Computer Vision systems often depend on AI models and datasets received from multiple contributors, vendors, repositories, and deployment environments.

This creates several security and trust risks, including:

- Model tampering
- Untrusted model artifacts
- Dataset corruption
- Data poisoning
- Duplicate or anomalous samples
- Out-of-Distribution inputs
- Distribution shift
- Inconsistent model behaviour
- Lack of traceable verification evidence

TrustCV addresses these risks through a layered verification and trust-assurance pipeline.

---

## TrustCV Security Pipeline

```text
                 ┌───────────────────────┐
                 │    MODEL ARTIFACT     │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  MODEL VERIFICATION  │
                 │                       │
                 │  SHA-256             │
                 │  Digital Signature   │
                 │  Vendor              │
                 │  Version             │
                 │  Provenance          │
                 │  Safe Inspection     │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   DATASET SECURITY   │
                 │                       │
                 │   Integrity          │
                 │   Duplicate Check    │
                 │   Visual Screening   │
                 │   Outlier Detection  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │      INPUT & OOD     │
                 │                       │
                 │   Input Integrity    │
                 │   OOD Detection      │
                 │   Distribution Shift │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │     RELIABILITY      │
                 │                       │
                 │   Consistency Tests  │
                 │   Perturbation Tests │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │     AUDIT LEDGER     │
                 │                       │
                 │ Verification Trace   │
                 │ Security Evidence    │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   TRUST / RISK       │
                 │      DECISION        │
                 └───────────────────────┘


## Key Features

### 1. Model Verification

TrustCV establishes a security gate before model execution.

The verification layer checks:

- SHA-256 artifact fingerprint
- Known-good hash comparison
- Digital signature verification
- Vendor authentication
- Version verification
- Model provenance
- Safe checkpoint inspection
- Model execution authorization

The system does not blindly execute arbitrary uploaded model artifacts.

### 2. Dataset Security

TrustCV evaluates datasets at both the integrity and visual-content levels.

Checks include:

- Dataset SHA-256 fingerprint
- File integrity
- Visual orientation
- Exact duplicate detection
- Visual outlier screening
- Poison/anomaly triage

A dataset can be cryptographically intact while still containing unusual or suspicious content, so integrity and content-level trust are treated as separate security layers.

### 3. Input & OOD Analysis

TrustCV evaluates inference inputs before relying on model predictions.

The input analysis layer includes:

- Input integrity verification
- Image quality checks
- Model inference
- Out-of-Distribution detection
- Distribution-shift analysis
- Risk assessment

This allows the system to identify inputs that differ significantly from the expected reference distribution.

### 4. Reliability Analysis

TrustCV evaluates prediction consistency under controlled input variations.

Example checks include:

- Brightness perturbation
- Horizontal flip consistency
- Prediction stability

Large changes in model behaviour can become an additional signal for investigation.

### 5. Audit Ledger

TrustCV maintains traceable verification evidence across the security workflow.

The audit layer helps record:

- Verification events
- Security results
- Artifact fingerprints
- Trust decisions
- Evidence used during verification

This improves traceability and accountability throughout the pipeline.

---

## Security Architecture

TrustCV follows a layered trust model:

```text
Integrity
    |
    v
Authenticity
    |
    v
Provenance
    |
    v
Behavioural Analysis
    |
    v
OOD / Distribution Analysis
    |
    v
Reliability
    |
    v
Auditability
    |
    v
Trust / Risk Decision
```

Each layer contributes evidence instead of relying on a single security score.

---

## Model Verification Workflow

```text
Upload Model
     |
     v
Calculate SHA-256
     |
     v
Compare Identity
     |
     v
Verify Signed Manifest
     |
     v
Verify Vendor
     |
     v
Verify Version
     |
     v
Verify Provenance
     |
     v
Safe Checkpoint Inspection
     |
     v
Execution Authorization
```

If the model does not satisfy the authorization requirements, TrustCV can block execution.

---

## Dataset Verification Workflow

```text
Upload Dataset
      |
      v
Calculate Dataset Hash
      |
      v
Verify File Integrity
      |
      v
Analyze Image Orientation
      |
      v
Detect Exact Duplicates
      |
      v
Visual Outlier / Poison Screening
      |
      v
Dataset Security Assessment
```

---

## OOD & Distribution Shift Workflow

```text
Upload Test Image
       |
       v
Input Integrity
       |
       v
Image Quality
       |
       v
Model Inference
       |
       v
Reference Comparison
       |
       v
OOD Analysis
       |
       v
Distribution Shift
       |
       v
Risk Assessment
```

---

## Reliability Workflow

```text
Original Input
      |
      +----> Original Prediction
      |
      +---- Brightness ----> Perturbed Prediction
      |
      +---- Flip ----------> Transformed Prediction
                               |
                               v
                        Consistency Analysis
                               |
                               v
                        Reliability Signal
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Interface | Streamlit |
| Programming | Python |
| Computer Vision | OpenCV |
| Object Detection | Ultralytics YOLO |
| Deep Learning Framework | PyTorch |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Cryptography | RSA-PSS / SHA-256 |
| Visualization | Streamlit Charts / Metrics |
| Deployment | Local / Streamlit-compatible |

---

## Project Structure

```text
trusted_cv_data_yolov11/
|
├── trustcv_final.py
├── requirements.txt
├── sign_model.py
├── trustcv_model_manifest.json
├── README.md
└── artifacts/
```

---

## Running the Application

### 1. Clone the repository

```bash
git clone <repository-url>
cd SIH-228/trusted_cv_data_yolov11
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
streamlit run trustcv_final.py
```

The TrustCV dashboard will open in the browser.

---

## Model Authorization

TrustCV uses a signed model manifest to associate a model artifact with trusted metadata.

The authorization flow is:

```text
Model File
    |
    v
SHA-256 Fingerprint
    |
    v
Signed Manifest
    |
    v
Digital Signature
    |
    v
Vendor Verification
    |
    v
Version Verification
    |
    v
Provenance Verification
    |
    v
Execution Authorization
```

The private signing key must never be committed to the repository.

---

## Cryptographic Integrity vs Trust

TrustCV intentionally separates different security concepts.

### Integrity

> Has the artifact changed relative to the expected byte-level fingerprint?

### Authenticity

> Is the artifact associated with an approved signer or authorization record?

### Behavioural Trust

> Does the model or input behave consistently with the expected system characteristics?

A successful SHA-256 comparison alone does not establish complete semantic safety.

---

## Example Model Verification Result

A successfully authorized model can produce results such as:

```text
SHA-256 / Identity        PASS
Digital Signature        PASS
Vendor Authentication    PASS
Version Verification     PASS
Provenance               PASS
Execution Gate           AUTHORIZED
```

---

## Example Dataset Verification Result

```text
Dataset Hash             PASS
File Integrity           PASS
Visual Orientation       PASS
Exact Duplicates         PASS
Visual Outlier Screen    PASS / REVIEW
```

The visual outlier screen is an investigation signal and should not automatically be interpreted as proof of malicious poisoning.

---

## Example Input Analysis

```text
Input Integrity
      |
      v
Image Quality
      |
      v
Model Inference
      |
      v
OOD Detection
      |
      v
Distribution Shift
      |
      v
Risk Decision
```

Possible outcomes include:

```text
IN-DISTRIBUTION
LOW SHIFT
LOW RISK
```

or:

```text
OOD DETECTED
SIGNIFICANT SHIFT
REVIEW / RISK
```

---

## Security Demonstration

The prototype can demonstrate two main scenarios.

### Scenario A — Trusted Pipeline

```text
Authorized Model
       |
       v
Verified Dataset
       |
       v
Normal Input
       |
       v
Low OOD / Low Shift
       |
       v
Trusted
```

### Scenario B — Suspicious Pipeline

```text
Unusual Dataset / Input
       |
       v
Visual or Behavioural Signal
       |
       v
OOD / Distribution Shift
       |
       v
Security Review
       |
       v
Risk Detected
```

---

## Safe Model Handling

Model files can contain serialized Python/PyTorch objects. TrustCV therefore treats uploaded artifacts cautiously instead of blindly deserializing arbitrary objects.

The prototype uses a verification and authorization gate before allowing model execution.

This is an important part of the TrustCV security architecture.

---

## Auditability

The audit layer helps preserve evidence from the verification process, including:

- Artifact fingerprints
- Verification results
- Security events
- Trust decisions
- Analysis outputs

The objective is to make the final decision explainable and traceable instead of relying on a single opaque score.

---

## Limitations

TrustCV is a prototype security-assurance system.

Several detection layers are intended as screening or triage mechanisms rather than absolute proof of malicious activity.

For example:

- SHA-256 verifies byte-level identity, not semantic safety.
- OOD detection does not prove that an input is malicious.
- Visual outlier detection can generate false positives.
- Dataset anomaly screening is evidence for review, not definitive proof of poisoning.
- Generic object-detection models are not authenticity classifiers for real-world military assets.

---

## Demo Data

Synthetic data may be used to demonstrate:

- Dataset integrity checks
- Visual anomalies
- Duplicate detection
- OOD behaviour
- Distribution-shift analysis
- Risk assessment

The defence-themed demonstration data is fictional/synthetic and does not represent real military locations, assets, personnel, or operational intelligence.

---

## Future Scope

Planned improvements can include:

- Trusted vendor public-key registry
- Centralized signed model registry
- Remote provenance verification
- Advanced dataset poisoning detection
- Model behaviour fingerprinting
- Continuous deployment security integration
- Role-based approval workflows
- Centralized SOC monitoring
- Real-time security alerts
- Secure artifact lifecycle management

---

## Why TrustCV?

Traditional Computer Vision pipelines often focus primarily on whether a model produces a prediction.

TrustCV adds another question:

> **Can we trust the model, the data, and the input before trusting the prediction?**

The system therefore treats AI security as a pipeline rather than a single verification step.

---

## Team

**Smart India Hackathon 2026 — SIH-228**

**Project:** TrustCV

**Focus:** Trust, Integrity and Security Assurance for Computer Vision Systems

