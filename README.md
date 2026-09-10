# Trusted CV
### A Trust, Integrity & Security Layer for Computer Vision Models

> **Smart India Hackathon — SIH 2026**  
> A model-aware computer vision security pipeline for detecting suspicious models, rejecting unreliable inputs, validating inference integrity, and maintaining a tamper-evident audit trail.

---

## Overview

Modern computer-vision systems can fail in more ways than simple prediction errors.

A model may contain a hidden **backdoor trigger**, receive an **out-of-distribution input**, produce unstable predictions under small perturbations, or generate a high-confidence prediction that should not be trusted.

**Trusted CV** is designed as a security and governance layer around a deployed computer-vision model.

Instead of treating every model prediction as trustworthy, the system evaluates:

- whether the **model itself may be compromised**
- whether the **incoming image is suspicious or distribution-shifted**
- whether the **prediction is sufficiently reliable**
- whether the prediction remains stable under small perturbations
- whether the complete inference event can be **audited and cryptographically verified**

The repository currently implements:

- **Phase 4 — Model Integrity / Backdoor Verification**
- **Phase 6 — OOD & Distribution-Shift Gate**
- **Phase 7 — Trusted YOLO Inference**
- **Phase 8 — Inference Integrity & Reliability**
- **Phase 9 — Cryptographic Audit Ledger**

The final deployment pipeline is deliberately **calibration-free at runtime** and does not require a calibration-image folder for normal inference.


# Phase 4 — Model Integrity

Phase 4 addresses a different question:

> **Can the supplied model itself be trusted?**

This stage supports both **white-box** and **black-box** access.

## White-Box Verification

When model weights are available, Trusted CV can perform deeper model-level inspection:

```text
                    Model Weights
                         │
                         ▼
                ┌─────────────────┐
                │ Neural Cleanse   │
                │ Stage 1          │
                └────────┬────────┘
                         │
                 Suspicious class?
                    /           \
                  No             Yes
                  │               │
                  │               ▼
                  │       ┌───────────────┐
                  │       │ Ablation      │
                  │       │ Stage 2       │
                  │       └───────┬───────┘
                  │               │
                  └───────┬───────┘
                          ▼
                    Final assessment
```

### Neural Cleanse

The white-box pipeline uses **Neural Cleanse** as the primary trigger-detection mechanism.

It searches for anomalous class-specific trigger patterns and identifies a candidate suspicious class when the recovered trigger exhibits abnormal characteristics.

The repository also exposes the recovered trigger information generated during this process.

### Ablation Validation

If Neural Cleanse identifies a suspicious class, the system performs a second-stage **ablation validation**.

Rather than treating Neural Cleanse alone as definitive evidence, the candidate trigger is tested through a separate validation stage.

This creates a two-stage verification pathway:

**Detection → Independent validation**

The Phase 4 implementation explicitly runs Stage 2 only when Stage 1 produces a candidate suspicious class.

### Black-Box Mode

When only forward-pass access is available and model weights/gradients are unavailable:

```text
Black-box Model
      │
      ▼
   STRIP
      │
      ▼
Entropy-based
trigger detection
```

Neural Cleanse and gradient-based ablation are unavailable in this mode.

The system therefore falls back to **STRIP**, while explicitly recording this limitation in the assessment report.

---


# System Architecture for 6-9

```text
                         ┌─────────────────────────┐
                         │      Input Image        │
                         │   / Webcam Frame        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────┐
                    │ Phase 6                      │
                    │ OOD / Distribution-Shift Gate│
                    │                               │
                    │ • Confidence                 │
                    │ • Confidence margin          │
                    │ • Detection density           │
                    │ • Perturbation consistency    │
                    │ • Image quality               │
                    └───────────────┬───────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    Suspicious              Acceptable
                         │                     │
                         ▼                     ▼
                  Flag / Gate            ┌──────────────┐
                  decision               │ Phase 7      │
                                        │ YOLO Inference│
                                        └──────┬───────┘
                                               │
                                               ▼
                                  ┌────────────────────────┐
                                  │ Phase 8                │
                                  │ Inference Integrity    │
                                  │                        │
                                  │ • Calibration proxy   │
                                  │ • Confidence           │
                                  │ • Robustness           │
                                  │ • OOD consistency      │
                                  └────────────┬───────────┘
                                               │
                                               ▼
                                  ┌────────────────────────┐
                                  │ Final Disposition      │
                                  │                        │
                                  │ ACCEPT / REVIEW /      │
                                  │ QUARANTINE             │
                                  └────────────┬───────────┘
                                               │
                                               ▼
                                  ┌────────────────────────┐
                                  │ Phase 9                │
                                  │ SHA-256 Audit Ledger   │
                                  │                        │
                                  │ Immutable inference    │
                                  │ event + record hash    │
                                  └────────────────────────┘
```

The deployment pipeline connects Phases 6–9 directly in `run_pipeline.py`. Each processed frame goes through the OOD gate, inference, integrity evaluation, and finally the audit ledger.
---
# Phase 6 — OOD / Distribution-Shift Gate

Phase 6 is a **model-derived deployment gate**.

Its purpose is to identify inputs that appear inconsistent with the model's expected operating distribution before relying on the full inference result.

The gate uses multiple signals:

### 1. Detection Confidence

Low model confidence increases the risk that the input is unfamiliar or ambiguous.

### 2. Confidence Margin

The difference between the strongest and second-strongest detections is used as an additional uncertainty signal.

### 3. Detection Density

Unexpected detection density can indicate unusual input content.

### 4. Perturbation Consistency

The model is tested against small changes to the image.

A trustworthy prediction should remain reasonably stable under mild transformations.

### 5. Image Quality

Basic image statistics such as:

- brightness
- contrast
- saturation
- sharpness

are inspected to identify severe image-quality shifts.

These signals are combined into an OOD/distribution-shift risk assessment.

> **Important:** Phase 6 is a deployment gate, not a universal semantic OOD classifier. Its thresholds are engineering heuristics derived from model behaviour.

---

# Phase 7 — Trusted Inference

Once the input passes the deployment gate, the system performs the full YOLO inference using the supplied model.

The inference output contains the detected objects and associated information such as:

```text
class
confidence
bounding box
```

The pipeline can operate on:

- a single image
- a live webcam stream

The final implementation uses `TrustedDetector` as the inference layer.

---

# Phase 8 — Inference Integrity

A prediction being produced does **not automatically mean that it should be trusted**.

Phase 8 therefore evaluates the integrity of the inference result.

The reliability score combines four major components:

```text
Reliability Score
        │
        ├── 25% Calibration proxy
        ├── 30% Robustness
        ├── 25% Confidence
        └── 20% OOD consistency
```

The implementation computes:

```text
score =
    0.25 × calibration
  + 0.30 × robustness
  + 0.25 × confidence
  + 0.20 × shift_consistency
```

If the Phase 6 OOD gate flags the input, the overall score is capped to prevent an apparently confident prediction from being treated as reliable.

## Reliability States

| Score | Status |
|---|---|
| ≥ 0.75 | `RELIABLE` |
| 0.50 – 0.74 | `LOW_CONFIDENCE` |
| < 0.50 | `FLAGGED` |

The system also produces per-check dispositions:

```text
ACCEPT
REVIEW
QUARANTINE
```

The overall disposition follows the most severe finding:

```text
QUARANTINE
    ↓
REVIEW
    ↓
ACCEPT
```

This allows the system to expose **why** an inference was considered unreliable instead of returning only one opaque score.

---

# Robustness Verification

Phase 8 performs lightweight robustness testing using small input perturbations.

The current implementation evaluates variants including:

- brightness increase
- brightness decrease
- small resize perturbation

The top prediction is compared against the original prediction using:

- class consistency
- bounding-box IoU

A prediction that changes significantly under a small perturbation receives a lower robustness score.

---

# Phase 9 — Cryptographic Audit Ledger

Trusted CV does not stop at producing a prediction.

Every inference event is recorded in an append-oriented audit ledger.

Each event contains information such as:

```text
timestamp
input source
model SHA-256
access level
final disposition
detections
Phase 6 result
Phase 8 integrity result
limitations
```

The event is canonicalized and hashed using **SHA-256** before being appended to the ledger.

```text
Inference Event
      │
      ▼
Canonical JSON
      │
      ▼
SHA-256
      │
      ▼
Audit Record
```

This provides a cryptographically verifiable record of what the system observed and how it evaluated the inference.

The final pipeline generates an `inference_record_sha256` for each event and exposes the resulting ledger record hash.

---

# Why This Approach?

Traditional computer-vision pipelines usually look like:

```text
Image → Model → Prediction
```

Trusted CV changes this to:

```text
Image
  ↓
Is the input trustworthy?
  ↓
Run inference
  ↓
Is the inference reliable?
  ↓
What should we do with the result?
  ↓
Record the decision cryptographically
```

And before deployment:

```text
Model
  ↓
Is the model itself trustworthy?
  ↓
Backdoor / trigger verification
  ↓
Deploy only after integrity assessment
```

This creates a layered security architecture rather than relying on a single detector.

---

# Decision Model

Trusted CV uses three practical dispositions:

### ACCEPT

The inference passes the relevant checks and can proceed normally.

### REVIEW

The system identifies uncertainty or degraded reliability that should be inspected.

### QUARANTINE

A serious integrity or distribution-shift signal is detected and the result should not be trusted automatically.

This distinction is important for real-world deployment because uncertainty should not always result in a hard failure.

---

# Repository Structure

```text
SIH-228/
│
├── phase4_trigger/
│   │
│   ├── neural_cleanse.py
│   ├── ablation_stage2.py
│   ├── phase4_model_integrity.py
│   ├── poison_and_train.py
│   └── strip_cifar.py
│
├── trusted_cv_6_9_final/
│   │
│   ├── run_pipeline.py
│   ├── inference.py
│   ├── ood_gate.py
│   ├── phase8_integrity.py
│   ├── ledger.py
│   ├── crypto_utils.py
│   ├── calibrate.py
│   ├── best.pt
│   ├── requirements.txt
│   ├── test.png
│   └── audit/
│
└── README.md
```

The repository separates the model-security stage from the final trusted-inference deployment pipeline.

---

# Technology Stack

| Component | Technology |
|---|---|
| Model inference | YOLO |
| Deep learning | PyTorch |
| Computer vision | OpenCV |
| Numerical processing | NumPy |
| Backdoor detection | Neural Cleanse |
| Black-box detection | STRIP |
| Model validation | Ablation testing |
| OOD / shift detection | Model-derived multi-signal gate |
| Integrity analysis | Confidence + robustness + shift consistency |
| Cryptography | SHA-256 |
| Interface | Python CLI + OpenCV webcam interface |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/rishidakshbansal2004-create/SIH-228.git
cd SIH-228
```

Enter the final deployment pipeline:

```bash
cd trusted_cv_6_9_final
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the System

## Image Inference

Run the complete Phase 6–9 pipeline on an image:

```bash
python run_pipeline.py --image test.png --weights best.pt
```

The system prints:

```text
[PHASE 6]
OOD / distribution-shift assessment

[PHASE 7]
YOLO detections

[PHASE 8]
Inference integrity report

[PHASE 9]
Cryptographic record hash
```

This is the primary demonstration mode supported by the repository.

---

## Webcam Mode

Run live inference:

```bash
python run_pipeline.py --camera 0 --weights best.pt
```

The webcam interface displays:

- detected objects
- confidence
- Phase 6 OOD status
- OOD risk score
- Phase 8 integrity status
- reliability score
- calibration component
- robustness component
- confidence component
- distribution-shift consistency

Press:

```text
Q
```

or

```text
ESC
```

to stop the application.

---

# Optional Runtime Controls

### Allow OOD Inputs

```bash
python run_pipeline.py \
    --image test.png \
    --weights best.pt \
    --allow-ood
```

### Disable Robustness Probe

```bash
python run_pipeline.py \
    --image test.png \
    --weights best.pt \
    --no-robustness
```

### Adjust Detection Thresholds

```bash
python run_pipeline.py \
    --image test.png \
    --weights best.pt \
    --conf 0.25 \
    --iou 0.45
```

---

# Phase 4 Usage

The Phase 4 model-integrity module supports explicit access-level selection.

### White-box

```bash
cd phase4_trigger

python phase4_model_integrity.py \
    --weights <MODEL_WEIGHTS> \
    --access_level white_box
```

### Black-box

```bash
python phase4_model_integrity.py \
    --weights <MODEL_WEIGHTS> \
    --access_level black_box
```

The resulting assessment is written to:

```text
phase4_report.json
```

The white-box route can combine Neural Cleanse and conditional ablation validation, while the black-box route falls back to STRIP.

---

# Output

A typical Phase 8 result contains:

```json
{
  "status": "RELIABLE",
  "reliability_score": 0.82,
  "disposition": "accept",
  "components": {
    "calibration": 0.91,
    "robustness": 0.80,
    "confidence": 0.88,
    "uncertainty": 0.12,
    "ood_distribution_consistency": 0.76
  }
}
```

The actual values depend on the input and model.

For every inference, the system also records the corresponding Phase 6 result, detections, limitations and cryptographic record information.

---

# Security Philosophy

Trusted CV follows a **defence-in-depth** approach.

### Layer 1 — Model Integrity

Detect possible backdoors and suspicious trigger behaviour.

### Layer 2 — Input Integrity

Identify distribution shift and suspicious inputs.

### Layer 3 — Prediction Integrity

Measure confidence and prediction stability.

### Layer 4 — Decision Governance

Convert multiple signals into:

```text
ACCEPT / REVIEW / QUARANTINE
```

### Layer 5 — Auditability

Cryptographically record every inference event.

```text
Model Security
      +
Input Security
      +
Inference Reliability
      +
Cryptographic Auditability
      =
Trusted Computer Vision
```

---

# Important Limitations

This prototype intentionally makes conservative claims.

### Model-derived OOD detection

Phase 6 is **not a universal semantic OOD detector**. It derives its signals from the behaviour of the trusted model and image statistics.

### Accuracy

Actual model accuracy requires labelled ground-truth data. The live Phase 8 pipeline therefore does not claim an accuracy value without ground truth.

### Calibration

The live system uses a confidence-based calibration proxy rather than claiming a true Expected Calibration Error (ECE) without labelled data.

### Robustness

The current robustness test uses lightweight photometric and resize perturbations. It is not an adversarial-robustness certification.

### Backdoor detection

Backdoor detection is handled separately through Phase 4 model verification rather than being claimed as part of the Phase 6 OOD detector.

---

# Project Status

| Phase | Component | Status |
|---|---|---|
| Phase 4 | Model integrity | Implemented |
| Phase 4 | Neural Cleanse | Implemented |
| Phase 4 | Ablation validation | Implemented |
| Phase 4 | STRIP fallback | Implemented |
| Phase 6 | OOD / shift gate | Implemented |
| Phase 7 | Trusted YOLO inference | Implemented |
| Phase 8 | Inference integrity | Implemented |
| Phase 8 | Robustness checking | Implemented |
| Phase 9 | SHA-256 audit ledger | Implemented |

---

# Key Design Principle

> **A prediction is not automatically trustworthy just because a model produced it.**

Trusted CV evaluates the model, the input, the inference behaviour, and the audit trail as separate security layers.

The goal is not simply:

```text
"Did the model detect something?"
```

but:

```text
"Can we trust this model,
this input,
this prediction,
and this decision —
and can we prove what happened afterwards?"
```

---

# Smart India Hackathon

**Project:** Trusted CV  
**Repository:** `SIH-228`  
**Domain:** AI / Computer Vision / Cybersecurity / Trustworthy AI  
**Focus:** Secure and auditable deployment of computer-vision models

---

## Team

Built for **Smart India Hackathon 2026**.

---

## License

Add the project's chosen license here if/when the repository is formally licensed.
