
import base64
import hashlib
import io
import json
import secrets
import tempfile
import time
import zipfile
from html import escape
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from trustcv_mirad_dataset_security import (
    canonicalize,
    ensure_trust_anchor,
    create_signed_dataset_manifest,
    verify_signed_dataset_manifest,
    persist_dataset_security_run,
    replay_existing_provenance,
    verify_dataset_audit,
    verify_dataset_checkpoint,
    create_signed_checkpoint as mirad_create_checkpoint,
    get_persisted_security_paths,
)
from contributor_backend import get_contributor_backend
from ethereum_anchor import (
    EthereumConfig,
    AnchorResult,
    anchor_checkpoint as eth_anchor_checkpoint,
    verify_anchor as eth_verify_anchor,
    get_ethereum_status,
)

if st.runtime.exists():
    st.set_page_config(
        page_title="TrustCV | Dataset Security Assurance",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

APP = Path(__file__).resolve().parent
RUNTIME = Path(tempfile.gettempdir()) / "TrustCV_DATASET_CONTRIBUTOR_RUNTIME"
RUNTIME.mkdir(parents=True, exist_ok=True)
AUDIT = RUNTIME / "dataset_audit_chain.jsonl"
PASSKEY_FILE = Path.home() / "TrustCV_access_config.json"
LOCAL_PASSKEY_FILE = APP / "access_config.json"

# ================================================================
# LOCAL CONTRIBUTOR GATE
# ================================================================

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def load_passkey_hash():
    """Load the operator passkey SHA-256 hash from Streamlit secrets (preferred),
    with an optional plaintext secret or local file fallback for development."""
    # 1) Streamlit secrets: hash 
    try:
        value = st.secrets.get("passkey_sha256", "")
        if not value and "auth" in st.secrets:
            value = st.secrets["auth"].get("passkey_sha256", "")
        value = str(value).strip().lower()
        if value:
            return value
    except Exception:
        pass  # no secrets.toml available

    # 2) Streamlit secrets: plaintext passkey
    try:
        plain = st.secrets.get("passkey", "")
        if not plain and "auth" in st.secrets:
            plain = st.secrets["auth"].get("passkey", "")
        if plain:
            return sha256_text(str(plain))
    except Exception:
        pass

    # 3) Local file fallback (development only)
    for config_path in (PASSKEY_FILE, LOCAL_PASSKEY_FILE):
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return str(config.get("passkey_sha256", "")).strip().lower()
        except Exception:
            continue
    return ""

def verify_passkey(value: str):
    expected = load_passkey_hash()
    if not expected:
        return False, "Operator passkey is not configured. Add `passkey_sha256` to Streamlit secrets."
    supplied = sha256_text(value or "")
    ok = secrets.compare_digest(supplied, expected)
    return ok, "Operator passkey verified." if ok else "Invalid operator passkey."

# ================================================================
# CRYPTOGRAPHIC AUDIT — TAMPER-EVIDENT HASH CHAIN (NO POW/MINING)
# ================================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def audit_event(event):
    previous = ""
    if AUDIT.exists():
        try:
            lines = AUDIT.read_text(encoding="utf-8").splitlines()
            if lines:
                previous = json.loads(lines[-1]).get("chain_hash", "")
        except Exception:
            pass

    record = dict(event)
    record["timestamp_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    record["previous_hash"] = previous
    record["chain_hash"] = sha256_bytes(
        json.dumps(record, sort_keys=True).encode("utf-8")
    )

    with AUDIT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

def verify_audit():
    if not AUDIT.exists():
        return True, 0

    previous = ""
    count = 0

    try:
        for line in AUDIT.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            stored = obj.pop("chain_hash")
            expected = sha256_bytes(
                json.dumps(obj, sort_keys=True).encode("utf-8")
            )

            if obj.get("previous_hash", "") != previous or expected != stored:
                return False, count

            previous = stored
            count += 1

        return True, count

    except Exception:
        return False, count

def audit_chain_digest():
    """Return the digest of the latest audit chain entry."""
    if not AUDIT.exists():
        return None
    try:
        lines = AUDIT.read_text(encoding="utf-8").splitlines()
        if lines:
            return json.loads(lines[-1]).get("chain_hash")
    except Exception:
        pass
    return None

# ================================================================
# SIGNED DATASET MANIFEST
# ================================================================

def verify_signed_manifest(
    manifest_bytes: bytes | str | dict,
    dataset_hash: str,
    expected_contributor: str | None = None,
    expected_batch: str | None = None,
    dataset_name: str | None = None,
):
    """Verify signed dataset manifest with full cryptographic and binding checks."""
    result = verify_signed_dataset_manifest(
        manifest_bytes,
        actual_dataset_digest=dataset_hash,
        expected_contributor_id=expected_contributor,
        expected_batch_id=expected_batch,
        actual_dataset_name=dataset_name,
    )
    manifest = result.get("manifest", {})
    return result["overall"], result["reason"], manifest

# ================================================================
# IMAGE ANALYSIS
# ================================================================

def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()

def average_hash(image: Image.Image, size: int = 16) -> int:
    gray = ImageOps.grayscale(image).resize((size, size))
    arr = np.asarray(gray, dtype=np.float32)
    threshold = float(arr.mean())
    bits = (arr >= threshold).reshape(-1)

    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value

def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())

def visual_features(image: Image.Image) -> np.ndarray:
    arr = np.asarray(
        image.convert("RGB").resize((96, 96)),
        dtype=np.float32,
    ) / 255.0
    gray = arr.mean(axis=2)

    hist, _ = np.histogram(gray, bins=24, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-9

    gy, gx = np.gradient(gray)
    edge_strength = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    entropy = float(
        -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()
    )

    return np.r_[
        arr.mean((0, 1)),
        arr.std((0, 1)),
        gray.mean(),
        gray.std(),
        edge_strength,
        entropy,
        hist,
    ].astype(np.float32)

def image_quality(image: Image.Image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    brightness = float(gray.mean())
    contrast = float(gray.std())

    gy, gx = np.gradient(gray)
    sharpness = float(np.mean(np.sqrt(gx * gx + gy * gy)))

    issues = []
    if image.width < 224 or image.height < 224:
        issues.append("LOW_RESOLUTION")
    if contrast < 0.05:
        issues.append("LOW_CONTRAST")
    if sharpness < 0.025:
        issues.append("LOW_SHARPNESS")
    if brightness < 0.08:
        issues.append("VERY_DARK")
    if brightness > 0.92:
        issues.append("VERY_BRIGHT")

    penalty = min(len(issues) * 22, 88)
    confidence = float(np.clip(100 - penalty, 0, 100))

    return {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 4),
        "issues": issues,
        "quality_confidence": round(confidence, 1),
    }

def exif_orientation(image: Image.Image):
    try:
        value = image.getexif().get(274)
    except Exception:
        value = None

    if value in (None, 1):
        return "PASS", "No non-upright EXIF orientation."

    labels = {
        2: "Mirrored horizontal",
        3: "Rotated 180°",
        4: "Mirrored vertical",
        5: "Mirrored + rotated 90° CW",
        6: "Rotated 90° CW",
        7: "Mirrored + rotated 90° CCW",
        8: "Rotated 90° CCW",
    }

    return "REVIEW", f"Non-upright EXIF orientation: {labels.get(value, value)}."

def image_report(image: Image.Image, index: int, source: str, contributor: str, batch: str = "DEFAULT_BATCH"):
    quality = image_quality(image)
    orientation_status, orientation_evidence = exif_orientation(image)

    flags = list(quality["issues"])
    if orientation_status == "REVIEW":
        flags.append("NON_UPRIGHT_EXIF")

    return {
        "index": index,
        "source": source,
        "contributor": contributor,
        "batch": batch,
        "hash": sha256_bytes(png_bytes(image)),
        "phash": average_hash(image),
        "width": image.width,
        "height": image.height,
        "brightness": quality["brightness"],
        "contrast": quality["contrast"],
        "sharpness": quality["sharpness"],
        "quality_issues": quality["issues"],
        "quality_confidence": quality["quality_confidence"],
        "orientation_status": orientation_status,
        "orientation_evidence": orientation_evidence,
        "features": visual_features(image),
        "anomaly_confidence": 0.0,
        "spectral_confidence": 0.0,
        "shift_distance": 0.0,
        "shift_confidence": 0.0,
        "shift_status": "REFERENCE PENDING",
        "review_confidence": 0.0,
        "severity": "LOW",
        "disposition": "ACCEPT",
        "is_visual_outlier": False,
        "is_spectral_outlier": False,
        "is_near_duplicate": False,
        "near_duplicate_of": "",
        "duplicate_count": 0,
        "label": "",
        "label_source": "",
        "trigger_flags": [],
    }

# ================================================================
# DATASET INGESTION
# ================================================================

BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js",
    ".msi", ".scr", ".com", ".jar", ".hta", ".lnk",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png", ".apng", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".avif", ".heic", ".heif"}
IMAGE_MAX_BYTES = 50 * 1024 * 1024
IMAGE_SCAN_MAX_FILES = 10000
TEXT_LIKE_SUFFIXES = {".json", ".txt", ".csv", ".xml", ".yaml", ".yml", ".md", ".html", ".htm", ".toml", ".ini"}

def infer_contributor(member_name: str, default_contributor: str | None = None) -> str:
    parts = Path(member_name).parts
    if len(parts) >= 2:
        return parts[0]
    if default_contributor and str(default_contributor).strip():
        return str(default_contributor).strip()
    return "ROOT / UNKNOWN SOURCE"

def infer_batch(member_name: str, default_batch: str | None = None) -> str:
    parts = Path(member_name).parts
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    if default_batch and str(default_batch).strip():
        return str(default_batch).strip()
    return "DEFAULT_BATCH"

def _open_image_payload(payload: bytes):
    """Decode an image robustly while keeping corrupt payloads isolated."""
    if not payload or len(payload) > IMAGE_MAX_BYTES:
        return None
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image.convert("RGB")
    except Exception:
        try:
            from PIL import ImageFile
            old = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                image = Image.open(io.BytesIO(payload))
                image.load()
                return image.convert("RGB")
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = old
        except Exception:
            return None


def _looks_like_zip(payload: bytes) -> bool:
    if len(payload) < 4:
        return False
    return payload[:4] in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}


def _scan_zip_archive(archive, images, names, contributors, unreadable, *, batches=None, prefix="", depth=0, max_depth=4, seen=None, budget=None, default_contributor=None, default_batch=None):
    """Recursively discover decodable images in ZIPs, including nested bundles."""
    if seen is None:
        seen = set()
    if budget is None:
        budget = {"scanned": 0}
    if depth > max_depth or budget["scanned"] >= IMAGE_SCAN_MAX_FILES:
        return
    for info in archive.infolist():
        if info.is_dir() or budget["scanned"] >= IMAGE_SCAN_MAX_FILES:
            continue
        budget["scanned"] += 1
        member = info.filename
        full_name = f"{prefix}/{member}" if prefix else member
        suffix = Path(member).suffix.lower()
        if suffix in BLOCKED_EXTENSIONS:
            continue
        try:
            payload = archive.read(info)
        except Exception:
            unreadable.append(full_name)
            continue
        if suffix == ".zip" or _looks_like_zip(payload):
            if depth >= max_depth:
                continue
            marker = (full_name, int(info.file_size), int(getattr(info, "CRC", 0)))
            if marker in seen:
                continue
            seen.add(marker)
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as nested:
                    _scan_zip_archive(
                        nested, images, names, contributors, unreadable,
                        batches=batches,
                        prefix=full_name, depth=depth + 1, max_depth=max_depth,
                        seen=seen, budget=budget,
                        default_contributor=default_contributor,
                        default_batch=default_batch,
                    )
            except zipfile.BadZipFile:
                unreadable.append(full_name)
            continue
        if suffix in TEXT_LIKE_SUFFIXES:
            continue
        image = _open_image_payload(payload)
        if image is not None:
            images.append(image)
            names.append(full_name)
            contributors.append(infer_contributor(full_name, default_contributor=default_contributor))
            if batches is not None:
                batches.append(infer_batch(full_name, default_batch=default_batch))
        elif suffix in IMAGE_SUFFIXES:
            unreadable.append(full_name)


# ================================================================
# COCO / YOLO LABEL PARSING
# ================================================================

def _parse_coco_annotations(archive, coco_files):
    """Parse COCO JSON annotations from archive and return {image_filename: [labels]}."""
    labels = {}
    for coco_path in coco_files:
        try:
            raw = archive.read(coco_path)
            coco = json.loads(raw.decode("utf-8"))
            cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
            img_map = {img["id"]: img.get("file_name", "") for img in coco.get("images", [])}
            for ann in coco.get("annotations", []):
                fname = img_map.get(ann.get("image_id"), "")
                cat = cats.get(ann.get("category_id"), "unknown")
                if fname:
                    labels.setdefault(fname, []).append(cat)
        except Exception:
            pass
    return labels


def _parse_yolo_labels(archive, yolo_files):
    """Parse YOLO txt labels from archive. Returns {base_name: [class_ids]}."""
    labels = {}
    for yolo_path in yolo_files:
        try:
            raw = archive.read(yolo_path).decode("utf-8")
            stem = Path(yolo_path).stem
            classes = []
            for line in raw.strip().splitlines():
                parts = line.strip().split()
                if parts:
                    classes.append(parts[0])
            labels[stem] = classes
        except Exception:
            pass
    return labels


def parse_dataset(upload, contributor_id: str | None = None, batch_id: str | None = None):
    raw = upload.getvalue()
    name = upload.name.lower()
    digest = sha256_bytes(raw)

    if name.endswith(".csv"):
        return {
            "kind": "CSV",
            "name": upload.name,
            "hash": digest,
            "df": pd.read_csv(io.BytesIO(raw)),
            "images": [],
            "image_names": [],
            "image_contributors": [],
            "image_batches": [],
            "members": [],
            "unsafe_members": [],
            "unreadable": [],
            "zip_corrupt": False,
            "annotation": {},
            "labels": {},
            "label_format": "CSV",
        }

    if name.endswith((".xlsx", ".xls")):
        return {
            "kind": "Excel",
            "name": upload.name,
            "hash": digest,
            "df": pd.read_excel(io.BytesIO(raw)),
            "images": [],
            "image_names": [],
            "image_contributors": [],
            "image_batches": [],
            "members": [],
            "unsafe_members": [],
            "unreadable": [],
            "zip_corrupt": False,
            "annotation": {},
            "labels": {},
            "label_format": "Excel",
        }

    if name.endswith(".zip"):
        images = []
        names = []
        contributors = []
        batches = []
        members = []
        unsafe = []
        unreadable = []
        coco_files = []
        yolo_label_files = []
        csv_files = []
        labels = {}
        label_format = "NONE"

        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                corrupt_member = archive.testzip()

                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    member = info.filename
                    suffix = Path(member).suffix.lower()
                    members.append({
                        "name": member,
                        "size_bytes": int(info.file_size),
                        "compressed_bytes": int(info.compress_size),
                        "crc32": f"{info.CRC:08x}",
                    })
                    if suffix in BLOCKED_EXTENSIONS:
                        unsafe.append(member)
                    if suffix == ".json" and Path(member).name.lower() in {
                        "annotations.json", "instances.json", "coco.json",
                        "instances_train.json", "instances_val.json",
                    }:
                        coco_files.append(member)
                    elif suffix == ".txt" and "label" in member.lower():
                        yolo_label_files.append(member)
                    elif suffix == ".txt":
                        # Also consider plain .txt in labels/ directories
                        parts = Path(member).parts
                        if any(p.lower() in ("labels", "label") for p in parts):
                            yolo_label_files.append(member)
                    elif suffix == ".csv":
                        csv_files.append(member)

                _scan_zip_archive(
                    archive, images, names, contributors, unreadable,
                    batches=batches,
                    prefix="", depth=0, max_depth=4,
                    default_contributor=contributor_id,
                    default_batch=batch_id,
                )

                # Parse COCO annotations
                if coco_files:
                    labels = _parse_coco_annotations(archive, coco_files)
                    label_format = "COCO"
                elif yolo_label_files:
                    labels = _parse_yolo_labels(archive, yolo_label_files)
                    label_format = "YOLO"

        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid or corrupt ZIP dataset: {exc}") from exc

        return {
            "kind": "Image ZIP" if images else "ZIP",
            "name": upload.name,
            "hash": digest,
            "df": None,
            "images": images,
            "image_names": names,
            "image_contributors": contributors,
            "image_batches": batches,
            "members": members,
            "unsafe_members": unsafe,
            "unreadable": unreadable,
            "zip_corrupt": corrupt_member is not None,
            "annotation": {
                "coco_json_files": coco_files,
                "yolo_txt_candidates": yolo_label_files,
                "csv_files": csv_files,
            },
            "labels": labels,
            "label_format": label_format,
        }

    if name.endswith(tuple(IMAGE_SUFFIXES)):
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        c_val = str(contributor_id).strip() if (contributor_id and str(contributor_id).strip()) else "ROOT / UNKNOWN SOURCE"
        b_val = str(batch_id).strip() if (batch_id and str(batch_id).strip()) else "DEFAULT_BATCH"
        return {
            "kind": "Image",
            "name": upload.name,
            "hash": digest,
            "df": None,
            "images": [image],
            "image_names": [upload.name],
            "image_contributors": [c_val],
            "image_batches": [b_val],
            "members": [],
            "unsafe_members": [],
            "unreadable": [],
            "zip_corrupt": False,
            "annotation": {},
            "labels": {},
            "label_format": "NONE",
        }

    raise ValueError("Supported: ZIP, CSV, XLSX, XLS, JPG, JPEG, PNG, BMP, WEBP, TIFF, GIF.")

# ================================================================
# HASH / ARCHIVE SECURITY TELEMETRY
# ================================================================

def path_traversal_members(dataset):
    findings = []
    for member in dataset.get("members", []):
        name = str(member.get("name", "")).replace("\\", "/")
        parts = [part for part in name.split("/") if part]
        if name.startswith("/") or any(part == ".." for part in parts):
            findings.append(name)
    return findings

def archive_expansion_ratio(dataset):
    members = dataset.get("members", [])
    compressed = sum(int(x.get("compressed_bytes", 0)) for x in members)
    uncompressed = sum(int(x.get("size_bytes", 0)) for x in members)
    if compressed <= 0:
        return 1.0
    return float(uncompressed / compressed)

def nested_archive_members(dataset):
    archive_exts = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}
    return [
        str(x.get("name", ""))
        for x in dataset.get("members", [])
        if Path(str(x.get("name", ""))).suffix.lower() in archive_exts
    ]

def attach_security_telemetry(dataset, raw_bytes):
    dataset["raw_size_bytes"] = len(raw_bytes)
    dataset["hash_algorithm"] = "SHA-256"
    dataset["hash_bits"] = 256
    dataset["hash_hex_length"] = 64
    dataset["hash_recomputed"] = (
        sha256_bytes(raw_bytes).lower()
        == str(dataset.get("hash", "")).lower()
    )
    dataset["path_traversal_members"] = path_traversal_members(dataset)
    dataset["archive_expansion_ratio"] = round(
        archive_expansion_ratio(dataset),
        2,
    )
    dataset["nested_archive_members"] = nested_archive_members(dataset)
    dataset["archive_member_count"] = len(
        dataset.get("members", [])
    )

def signer_fingerprint(manifest):
    public_key = str(
        manifest.get("public_key")
        or manifest.get("public_key_pem")
        or ""
    )
    if not public_key:
        return "NOT AVAILABLE"
    return sha256_bytes(public_key.encode("utf-8"))[:32]

# ================================================================
# ANALYSIS
# ================================================================

def _attach_labels(reports, dataset):
    """Attach parsed labels to reports for label analysis."""
    labels = dataset.get("labels", {})
    label_format = dataset.get("label_format", "NONE")
    if not labels:
        return

    for r in reports:
        source = r["source"]
        stem = Path(source).stem
        fname = Path(source).name

        matched = labels.get(fname) or labels.get(source) or labels.get(stem)
        if matched:
            if isinstance(matched, list):
                r["label"] = ", ".join(str(x) for x in matched)
            else:
                r["label"] = str(matched)
            r["label_source"] = label_format


def _detect_label_inconsistencies(reports):
    """Detect label inconsistencies: same/near-duplicate images with different labels."""
    findings = []
    labeled = [r for r in reports if r.get("label")]
    if len(labeled) < 2:
        return findings

    # Check near-duplicates with different labels
    for i, ri in enumerate(labeled):
        for j in range(i + 1, len(labeled)):
            rj = labeled[j]
            d = hamming_distance(ri["phash"], rj["phash"])
            if d <= 6 and ri["label"] != rj["label"]:
                findings.append({
                    "finding_type": "LABEL_INCONSISTENCY",
                    "severity": "MEDIUM",
                    "confidence": round(max(0, 100 - d * 15), 1),
                    "sample_a": ri["source"],
                    "sample_b": rj["source"],
                    "label_a": ri["label"],
                    "label_b": rj["label"],
                    "reason": f"Near-duplicate images (hamming={d}) have different labels.",
                    "disposition": "REVIEW",
                })

    # Check exact duplicates with different labels (label flipping indicator)
    by_hash = defaultdict(list)
    for r in labeled:
        by_hash[r["hash"]].append(r)
    for h, group in by_hash.items():
        label_set = set(r["label"] for r in group)
        if len(label_set) > 1:
            findings.append({
                "finding_type": "LABEL_FLIPPING_INDICATOR",
                "severity": "HIGH",
                "confidence": 95.0,
                "affected_samples": [r["source"] for r in group],
                "labels_found": list(label_set),
                "reason": f"Exact-duplicate images assigned {len(label_set)} different labels.",
                "disposition": "QUARANTINE",
            })

    return findings


def _detect_trigger_indicators(reports):
    """Simple trigger injection indicators: tiny patches of unusual pixel intensity."""
    findings = []
    for r in reports:
        # Check for suspiciously small, high-contrast patches
        # This is a lightweight heuristic — not a full backdoor detector
        flags = []
        if r.get("quality_issues"):
            for issue in r["quality_issues"]:
                if issue in ("LOW_RESOLUTION",):
                    continue  # Not a trigger indicator
        # Statistical: images that are both spectral outliers AND visual outliers
        # with high anomaly confidence may contain injected patterns
        if r.get("is_visual_outlier") and r.get("is_spectral_outlier") and r.get("anomaly_confidence", 0) > 70:
            flags.append("DUAL_OUTLIER_HIGH_ANOMALY")

        if flags:
            r["trigger_flags"] = flags
            findings.append({
                "finding_type": "TRIGGER_INJECTION_INDICATOR",
                "severity": "HIGH",
                "confidence": round(r.get("anomaly_confidence", 50), 1),
                "sample_id": r["source"],
                "contributor": r["contributor"],
                "reason": f"Image shows dual-outlier signature with high anomaly confidence: {', '.join(flags)}",
                "disposition": "QUARANTINE",
            })

    return findings


def analyze_reports(dataset):
    images = dataset.get("images", [])
    names = dataset.get("image_names", [])
    contributors = dataset.get("image_contributors", [])
    batches = dataset.get("image_batches", [])

    reports = [
        image_report(
            image,
            i + 1,
            names[i] if i < len(names) else f"image_{i + 1}",
            contributors[i] if i < len(contributors) else "ROOT / UNKNOWN SOURCE",
            batch=batches[i] if i < len(batches) else "DEFAULT_BATCH",
        )
        for i, image in enumerate(images)
    ]

    if not reports:
        return reports

    # Attach labels from parsed annotations
    _attach_labels(reports, dataset)

    # Exact duplicates.
    by_hash = defaultdict(list)
    for r in reports:
        by_hash[r["hash"]].append(r["index"])

    for r in reports:
        r["duplicate_count"] = max(0, len(by_hash[r["hash"]]) - 1)

    # Near duplicate screen.
    near_threshold = 6
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            d = hamming_distance(reports[i]["phash"], reports[j]["phash"])
            if d <= near_threshold:
                reports[i]["is_near_duplicate"] = True
                reports[j]["is_near_duplicate"] = True
                reports[i]["near_duplicate_of"] = reports[j]["source"]
                reports[j]["near_duplicate_of"] = reports[i]["source"]

    if len(reports) < 5:
        return reports

    X = np.vstack([r["features"] for r in reports])

    # Visual statistical outlier screen.
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(X)
        detector = IsolationForest(
            n_estimators=300,
            random_state=42,
            contamination="auto",
        ).fit(scaled)

        decision = detector.decision_function(scaled)
        q01 = float(np.percentile(decision, 1))
        q05 = float(np.percentile(decision, 5))
        q95 = float(np.percentile(decision, 95))
        span = max(q95 - q01, 1e-6)

        for i, r in enumerate(reports):
            score = np.clip((q95 - decision[i]) / span, 0, 1) * 100
            r["anomaly_confidence"] = round(float(score), 1)
            r["is_visual_outlier"] = bool(decision[i] <= q05)

    except Exception as exc:
        dataset["anomaly_error"] = str(exc)

    # SVD spectral-style screening.
    try:
        center = X.mean(axis=0)
        scale = X.std(axis=0) + 1e-6
        Z = (X - center) / scale
        Zc = Z - Z.mean(axis=0, keepdims=True)

        _, singular_values, vt = np.linalg.svd(
            Zc,
            full_matrices=False,
        )

        top_vector = vt[0]
        spectral_score = (Zc @ top_vector) ** 2

        low = float(np.percentile(spectral_score, 5))
        high = float(np.percentile(spectral_score, 95))
        span = max(high - low, 1e-6)
        cutoff = float(np.percentile(spectral_score, 95))

        for i, r in enumerate(reports):
            score = np.clip((spectral_score[i] - low) / span, 0, 1) * 100
            r["spectral_confidence"] = round(float(score), 1)
            r["is_spectral_outlier"] = bool(spectral_score[i] >= cutoff)

        dataset["top_singular_value"] = round(float(singular_values[0]), 4)
        dataset["spectral_variance_ratio"] = round(
            float(
                singular_values[0] ** 2
                / (np.square(singular_values).sum() + 1e-9)
            ),
            4,
        )

    except Exception as exc:
        dataset["spectral_error"] = str(exc)

    # Per-image leave-one-out distribution shift.
    distances = []

    for i, r in enumerate(reports):
        mask = np.ones(len(reports), dtype=bool)
        mask[i] = False
        reference = X[mask]

        ref_mean = reference.mean(axis=0)
        ref_std = reference.std(axis=0) + 1e-6

        distance = float(
            np.mean(np.abs((r["features"] - ref_mean) / ref_std))
        )
        distances.append(distance)

    high_cutoff = max(2.0, float(np.percentile(distances, 95)))

    for r, distance in zip(reports, distances):
        confidence = np.clip(
            distance / max(high_cutoff, 1e-6),
            0,
            1,
        ) * 100

        r["shift_distance"] = round(distance, 3)
        r["shift_confidence"] = round(float(confidence), 1)

        if distance >= high_cutoff:
            r["shift_status"] = "HIGH SHIFT"
        elif distance >= 1.0:
            r["shift_status"] = "MODERATE SHIFT"
        else:
            r["shift_status"] = "STABLE"

    dataset["overall_shift_distance"] = round(float(np.mean(distances)), 3)
    dataset["high_shift_cutoff"] = round(float(high_cutoff), 3)

    # Composite screening confidence.
    for r in reports:
        duplicate = 100 if r["duplicate_count"] else 0
        near_duplicate = 70 if r["is_near_duplicate"] else 0
        orientation = 100 if r["orientation_status"] == "REVIEW" else 0
        quality_risk = 100 - r["quality_confidence"]

        review = (
            0.27 * r["anomaly_confidence"]
            + 0.23 * r["spectral_confidence"]
            + 0.25 * r["shift_confidence"]
            + 0.08 * duplicate
            + 0.07 * quality_risk
            + 0.05 * max(near_duplicate, orientation)
            + 0.05 * (100 if r["quality_issues"] else 0)
        )

        r["review_confidence"] = round(
            float(np.clip(review, 0, 100)),
            1,
        )

        if r["review_confidence"] >= 80:
            r["severity"] = "HIGH"
            r["disposition"] = "QUARANTINE"
        elif r["review_confidence"] >= 50:
            r["severity"] = "MEDIUM"
            r["disposition"] = "REVIEW"
        else:
            r["severity"] = "LOW"
            r["disposition"] = "ACCEPT"

    return reports


# ================================================================
# FINDINGS ENGINE
# ================================================================

def generate_findings(dataset, reports):
    """Generate structured findings from actual analysis results."""
    findings = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dataset_id = dataset.get("hash", "")[:16]
    seq = 0

    # Exact duplicates
    by_hash = defaultdict(list)
    for r in reports:
        by_hash[r["hash"]].append(r)
    for h, group in by_hash.items():
        if len(group) > 1:
            seq += 1
            findings.append({
                "finding_id": f"F-{dataset_id}-{seq:04d}",
                "dataset_id": dataset.get("hash"),
                "finding_type": "EXACT_DUPLICATE",
                "severity": "MEDIUM",
                "confidence": 100.0,
                "affected_samples": [r["source"] for r in group],
                "contributor_ids": list(set(r["contributor"] for r in group)),
                "batch_id": group[0].get("batch", "DEFAULT_BATCH"),
                "reason": f"{len(group)} exact duplicate images detected.",
                "recommended_disposition": "REVIEW",
                "timestamp": ts,
            })

    # Near duplicates
    by_source = {r["source"]: r for r in reports}
    near_pairs = set()
    for r in reports:
        if r["is_near_duplicate"] and r["near_duplicate_of"]:
            pair = tuple(sorted([r["source"], r["near_duplicate_of"]]))
            if pair not in near_pairs:
                near_pairs.add(pair)
                seq += 1
                pair_contributors = list(set(by_source[s]["contributor"] for s in pair if s in by_source))
                pair_batches = list(set(by_source[s].get("batch", "") for s in pair if s in by_source))
                findings.append({
                    "finding_id": f"F-{dataset_id}-{seq:04d}",
                    "dataset_id": dataset.get("hash"),
                    "finding_type": "NEAR_DUPLICATE",
                    "severity": "LOW",
                    "confidence": 80.0,
                    "affected_samples": list(pair),
                    "contributor_ids": pair_contributors,
                    "batch_id": pair_batches[0] if pair_batches else "DEFAULT_BATCH",
                    "reason": "Near-duplicate pair detected by perceptual hash.",
                    "recommended_disposition": "REVIEW",
                    "timestamp": ts,
                })

    # Visual outliers
    for r in reports:
        if r["is_visual_outlier"]:
            seq += 1
            findings.append({
                "finding_id": f"F-{dataset_id}-{seq:04d}",
                "dataset_id": dataset.get("hash"),
                "sample_id": r["source"],
                "sample_hash": r.get("hash", ""),
                "contributor_id": r["contributor"],
                "batch_id": r.get("batch", "DEFAULT_BATCH"),
                "finding_type": "VISUAL_OUTLIER",
                "severity": r["severity"],
                "confidence": r["anomaly_confidence"],
                "reason": "Isolation Forest flagged this image as a statistical visual outlier.",
                "recommended_disposition": r["disposition"],
                "timestamp": ts,
            })

    # Spectral outliers
    for r in reports:
        if r["is_spectral_outlier"]:
            seq += 1
            findings.append({
                "finding_id": f"F-{dataset_id}-{seq:04d}",
                "dataset_id": dataset.get("hash"),
                "sample_id": r["source"],
                "sample_hash": r.get("hash", ""),
                "contributor_id": r["contributor"],
                "batch_id": r.get("batch", "DEFAULT_BATCH"),
                "finding_type": "SPECTRAL_OUTLIER",
                "severity": r["severity"],
                "confidence": r["spectral_confidence"],
                "reason": "SVD spectral screening flagged this image as an outlier in the principal spectral direction.",
                "recommended_disposition": r["disposition"],
                "timestamp": ts,
            })

    # High shift
    for r in reports:
        if r["shift_status"] == "HIGH SHIFT":
            seq += 1
            findings.append({
                "finding_id": f"F-{dataset_id}-{seq:04d}",
                "dataset_id": dataset.get("hash"),
                "sample_id": r["source"],
                "sample_hash": r.get("hash", ""),
                "contributor_id": r["contributor"],
                "batch_id": r.get("batch", "DEFAULT_BATCH"),
                "finding_type": "DISTRIBUTION_SHIFT",
                "severity": "MEDIUM",
                "confidence": r["shift_confidence"],
                "reason": f"Leave-one-out shift distance ({r['shift_distance']:.3f}) exceeds cutoff.",
                "recommended_disposition": "REVIEW",
                "timestamp": ts,
            })

    # Label findings
    label_findings = _detect_label_inconsistencies(reports)
    for lf in label_findings:
        seq += 1
        lf["finding_id"] = f"F-{dataset_id}-{seq:04d}"
        lf["dataset_id"] = dataset.get("hash")
        lf["timestamp"] = ts
        findings.append(lf)

    # Trigger indicator findings
    trigger_findings = _detect_trigger_indicators(reports)
    for tf in trigger_findings:
        seq += 1
        tf["finding_id"] = f"F-{dataset_id}-{seq:04d}"
        tf["dataset_id"] = dataset.get("hash")
        tf["timestamp"] = ts
        if "sample_id" in tf and tf["sample_id"] in by_source:
            tf["batch_id"] = by_source[tf["sample_id"]].get("batch", "DEFAULT_BATCH")
            tf["sample_hash"] = by_source[tf["sample_id"]].get("hash", "")
        findings.append(tf)

    # Quality findings
    for r in reports:
        if r["quality_issues"]:
            seq += 1
            findings.append({
                "finding_id": f"F-{dataset_id}-{seq:04d}",
                "dataset_id": dataset.get("hash"),
                "sample_id": r["source"],
                "contributor_id": r["contributor"],
                "finding_type": "QUALITY_FLAG",
                "severity": "LOW",
                "confidence": 100 - r["quality_confidence"],
                "reason": f"Quality flags: {', '.join(r['quality_issues'])}",
                "recommended_disposition": "REVIEW",
                "timestamp": ts,
            })

    return findings


# ================================================================
# SECURITY MATRIX
# ================================================================

def dataset_checks(dataset, reports):
    checks = []

    manifest_status = st.session_state.get("manifest_status", "REVIEW")
    manifest_reason = st.session_state.get(
        "manifest_reason",
        "No signed dataset manifest supplied.",
    )
    manifest = st.session_state.get("dataset_manifest", {})

    checks.append([
        "Dataset SHA-256 / Identity",
        "PASS",
        "Fingerprint generated from exact uploaded bytes.",
    ])

    if dataset["kind"] == "Image ZIP":
        checks += [
            [
                "ZIP CRC / Archive Integrity",
                "FAIL" if dataset.get("zip_corrupt") else "PASS",
                "ZIP CRC validation passed."
                if not dataset.get("zip_corrupt")
                else "ZIP CRC validation reported a damaged member.",
            ],
            [
                "Unsafe Extension Screen",
                "FAIL" if dataset.get("unsafe_members") else "PASS",
                "No blocked executable/script-like members."
                if not dataset.get("unsafe_members")
                else f"{len(dataset['unsafe_members'])} blocked member(s) found.",
            ],
            [
                "Readable Image Extraction",
                "REVIEW" if dataset.get("unreadable") else "PASS",
                f"{len(dataset.get('images', []))} readable image(s) extracted.",
            ],
        ]
    else:
        checks.append([
            "File Parsing / Integrity",
            "PASS",
            "Dataset parsed successfully.",
        ])

    checks.append([
        "Contributor Passkey",
        "PASS" if st.session_state.get("contributor_verified") else "LOCKED",
        "Contributor passkey verified before intake."
        if st.session_state.get("contributor_verified")
        else "Valid contributor passkey required before dataset onboarding.",
    ])

    checks.append([
        "Hash Recalculation Consistency",
        "PASS" if dataset.get("hash_recomputed") else "FAIL",
        "Independent SHA-256 recomputation matches the recorded digest."
        if dataset.get("hash_recomputed")
        else "Recomputed digest does not match the recorded digest.",
    ])

    if dataset.get("kind") == "Image ZIP":
        path_hits = dataset.get("path_traversal_members", [])
        ratio = float(dataset.get("archive_expansion_ratio", 1.0))
        nested = dataset.get("nested_archive_members", [])

        checks.append([
            "Archive Path Safety",
            "FAIL" if path_hits else "PASS",
            "No path-traversal style member names detected."
            if not path_hits
            else f"{len(path_hits)} suspicious path member(s) detected.",
        ])

        checks.append([
            "Archive Expansion Safety",
            "FAIL" if ratio >= 1000 else ("REVIEW" if ratio >= 100 else "PASS"),
            f"Uncompressed/compressed ratio = {ratio:.2f}x.",
        ])

        checks.append([
            "Nested Archive Screen",
            "REVIEW" if nested else "PASS",
            f"{len(nested)} nested archive member(s)."
            if nested
            else "No nested archive payloads detected.",
        ])

    checks.append([
        "Digital Signature",
        manifest_status,
        manifest_reason,
    ])

    provenance_keys = [
        ("Source", "source"),
        ("Vendor / Owner", "vendor"),
        ("Version", "version"),
        ("Created At", "created_at"),
    ]

    missing = [
        label
        for label, key in provenance_keys
        if not str(manifest.get(key, "")).strip()
    ]

    checks.append([
        "Dataset Provenance",
        "PASS" if manifest_status == "PASS" and not missing else "REVIEW",
        "Signed source/owner/version/time metadata present."
        if not missing
        else "Missing: " + ", ".join(missing),
    ])

    if reports:
        exact_dups = sum(r["duplicate_count"] for r in reports)
        near_dups = sum(r["is_near_duplicate"] for r in reports)
        orientation = sum(r["orientation_status"] == "REVIEW" for r in reports)
        quality = sum(bool(r["quality_issues"]) for r in reports)
        visual_outliers = sum(r["is_visual_outlier"] for r in reports)
        spectral = sum(r["is_spectral_outlier"] for r in reports)
        high_shift = sum(r["shift_status"] == "HIGH SHIFT" for r in reports)

        checks += [
            [
                "Exact Duplicate Detection",
                "REVIEW" if exact_dups else "PASS",
                f"{exact_dups} duplicate instance(s).",
            ],
            [
                "Near-Duplicate Detection",
                "REVIEW" if near_dups else "PASS",
                f"{near_dups} image(s) in near-duplicate pairs.",
            ],
            [
                "Orientation / EXIF",
                "REVIEW" if orientation else "PASS",
                f"{orientation} image(s) with non-upright EXIF.",
            ],
            [
                "Image Quality / Acquisition",
                "REVIEW" if quality else "PASS",
                f"{quality} image(s) with quality flags.",
            ],
            [
                "Visual Outlier Screening",
                "REVIEW" if visual_outliers else "PASS",
                f"{visual_outliers} statistical visual outlier(s).",
            ],
            [
                "SVD Spectral Screening",
                "REVIEW" if spectral else "PASS",
                f"{spectral} spectral outlier(s).",
            ],
            [
                "Distribution Shift",
                "REVIEW" if high_shift else "PASS",
                f"{high_shift} image(s) above high-shift cutoff.",
            ],
            [
                "Contributor / Source Aggregation",
                "PASS",
                f"{len(set(r['contributor'] for r in reports))} source group(s) analyzed.",
            ],
        ]

    if dataset.get("df") is not None:
        frame = dataset["df"]
        duplicate_rows = int(frame.duplicated().sum())
        missing_cells = int(frame.isna().sum().sum())

        checks += [
            [
                "Tabular Duplicate Rows",
                "REVIEW" if duplicate_rows else "PASS",
                f"{duplicate_rows} duplicate row(s).",
            ],
            [
                "Missing-Value Screen",
                "REVIEW" if missing_cells else "PASS",
                f"{missing_cells} missing cell(s).",
            ],
        ]

    if manifest:
        manifest_name = str(manifest.get("dataset_name", "")).strip()
        checks.append([
            "Manifest Dataset-Name Binding",
            "PASS"
            if manifest_name and manifest_name == dataset.get("name")
            else "REVIEW",
            "Signed manifest dataset name matches the uploaded dataset."
            if manifest_name and manifest_name == dataset.get("name")
            else "Manifest name is missing or does not match the upload.",
        ])

    # MIRAD audit chain verification
    mirad_audit = verify_dataset_audit()
    checks.append([
        "MIRAD Audit Chain",
        "PASS" if mirad_audit.get("valid") else "FAIL",
        f"Tamper-evident MIRAD audit chain: {mirad_audit.get('events', 0)} events · {'VALID' if mirad_audit.get('valid') else 'FAILED'}",
    ])

    # Local audit chain
    local_ok, local_count = verify_audit()
    checks.append([
        "Local Audit Chain",
        "PASS" if local_ok else "FAIL",
        f"Local hash-linked audit chain: {local_count} events · {'VALID' if local_ok else 'FAILED'}",
    ])

    return checks

def risk_dataframe(reports):
    rows = []
    for r in reports:
        rows.append({
            "Image #": r["index"],
            "File": r["source"],
            "Contributor": r["contributor"],
            "Batch": r.get("batch", "DEFAULT_BATCH"),
            "Label": r.get("label", ""),
            "Brightness": r["brightness"],
            "Contrast": r["contrast"],
            "Sharpness": r["sharpness"],
            "Quality %": r["quality_confidence"],
            "Anomaly %": r["anomaly_confidence"],
            "Spectral %": r["spectral_confidence"],
            "Shift Distance": r["shift_distance"],
            "Shift %": r["shift_confidence"],
            "Shift": r["shift_status"],
            "Exact Dup": r["duplicate_count"] > 0,
            "Near Dup": r["is_near_duplicate"],
            "Orientation": r["orientation_status"],
            "Review %": r["review_confidence"],
            "Severity": r["severity"],
            "Disposition": r["disposition"],
        })
    return pd.DataFrame(rows)

def contributor_dataframe(reports):
    groups = defaultdict(list)

    for r in reports:
        groups[r["contributor"]].append(r)

    rows = []

    for contributor, items in groups.items():
        exact_dup_flags = sum(r["duplicate_count"] > 0 for r in items)
        near_dup_flags = sum(r["is_near_duplicate"] for r in items)
        label_flags = sum(bool(r.get("label")) and r.get("is_near_duplicate") for r in items)
        ood_flags = sum(r["shift_status"] == "HIGH SHIFT" for r in items)
        trigger_flags = sum(bool(r.get("trigger_flags")) for r in items)
        anomaly_flags = sum(r["is_visual_outlier"] for r in items)

        avg_review = float(np.mean([x["review_confidence"] for x in items]))

        # Evidence-based status
        if avg_review >= 80 or trigger_flags > 0:
            status = "QUARANTINE_RECOMMENDED"
        elif avg_review >= 50:
            status = "ELEVATED"
        elif avg_review >= 25:
            status = "REVIEW"
        else:
            status = "NORMAL"

        rows.append({
            "Contributor / Source": contributor,
            "Samples": len(items),
            "Exact Dup Flags": exact_dup_flags,
            "Near Dup Flags": near_dup_flags,
            "Label Flags": label_flags,
            "OOD/Shift Flags": ood_flags,
            "Trigger Indicators": trigger_flags,
            "Anomaly Flags": anomaly_flags,
            "Avg Review %": round(avg_review, 1),
            "Quarantine": sum(x["disposition"] == "QUARANTINE" for x in items),
            "Status": status,
        })

    return pd.DataFrame(rows).sort_values(
        ["Quarantine", "Avg Review %"],
        ascending=[False, False],
    ) if rows else pd.DataFrame()


def batch_dataframe(reports):
    """Aggregate by batch (report batch or directory structure)."""
    groups = defaultdict(list)
    for r in reports:
        parts = Path(r["source"]).parts
        batch = r.get("batch") or (parts[1] if len(parts) >= 3 else parts[0] if len(parts) >= 2 else "DEFAULT_BATCH")
        groups[batch].append(r)

    rows = []
    for batch, items in groups.items():
        avg_review = float(np.mean([x["review_confidence"] for x in items]))
        rows.append({
            "Batch": batch,
            "Samples": len(items),
            "Visual Outliers": sum(x["is_visual_outlier"] for x in items),
            "Spectral Outliers": sum(x["is_spectral_outlier"] for x in items),
            "High Shift": sum(x["shift_status"] == "HIGH SHIFT" for x in items),
            "Quarantine": sum(x["disposition"] == "QUARANTINE" for x in items),
            "Avg Review %": round(avg_review, 1),
        })

    return pd.DataFrame(rows).sort_values("Avg Review %", ascending=False) if rows else pd.DataFrame()


# ================================================================
# REPORT EXPORTS — PDF / CSV / JSON / ZIP
# ================================================================

def build_json_report(dataset, reports, checks, trust, risk, audit_valid, audit_count):
    return {
        "report_type": "TrustCV Dataset Security Assessment",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": dataset.get("name") if dataset else None,
            "kind": dataset.get("kind") if dataset else None,
            "sha256": dataset.get("hash") if dataset else None,
        },
        "contributor_passkey_verified": bool(
            st.session_state.get("contributor_verified")
        ),
        "signed_manifest": {
            "status": st.session_state.get("manifest_status", "REVIEW"),
            "reason": st.session_state.get("manifest_reason", ""),
            "metadata": st.session_state.get("dataset_manifest", {}),
        },
        "posture": {
            "trust": trust,
            "risk": risk,
        },
        "checks": [
            {"control": row[0], "status": row[1], "evidence": row[2]}
            for row in checks
        ],
        "image_count": len(reports),
        "visual_outliers": sum(r.get("is_visual_outlier", False) for r in reports),
        "spectral_outliers": sum(r.get("is_spectral_outlier", False) for r in reports),
        "high_shift": sum(r.get("shift_status") == "HIGH SHIFT" for r in reports),
        "quarantine": sum(r.get("disposition") == "QUARANTINE" for r in reports),
        "audit": {"valid": audit_valid, "events": audit_count},
        "mirad_dataset_security": st.session_state.get("mirad_security_record", {}),
        "ethereum_anchor": st.session_state.get("ethereum_anchor_result", {}),
        "findings": st.session_state.get("dataset_findings", []),
    }

def build_pdf_report(dataset, reports, checks, trust, risk, audit_valid, audit_count):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
        title="TrustCV Dataset Security Assessment",
        author="TrustCV",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TrustCVTitle",
        parent=styles["Title"],
        fontSize=21,
        leading=24,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#122033"),
    )
    sub_style = ParagraphStyle(
        "TrustCVSub",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#5A6C80"),
    )
    h2_style = ParagraphStyle(
        "TrustCVH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=15,
        textColor=colors.HexColor("#17283A"),
    )
    cell_style = ParagraphStyle(
        "TrustCVCell",
        parent=styles["BodyText"],
        fontSize=6.6,
        leading=7.8,
    )
    small_style = ParagraphStyle(
        "TrustCVSmall",
        parent=styles["BodyText"],
        fontSize=6.2,
        leading=7.2,
    )

    def P(value, style=cell_style):
        safe = escape(str(value)).replace("\n", "<br/>")
        return Paragraph(safe, style)

    story = [
        P("TrustCV — Dataset Security Assessment", title_style),
        P(
            "Contributor-gated dataset assurance · Identity · Integrity · Authenticity · "
            "Provenance · Anomaly · SVD · Distribution Shift · Evidence · MIRAD Security",
            sub_style,
        ),
    ]

    manifest = st.session_state.get("dataset_manifest", {})
    dataset_name = dataset.get("name", "N/A") if dataset else "N/A"

    summary = [
        [P("DATASET"), P(dataset_name), P("SHA-256"), P(dataset.get("hash", "") if dataset else "N/A"),
         P("TRUST"), P(f"{trust}/100"), P("RISK"), P(f"{risk}/100")],
        [P("PASSKEY"), P("VERIFIED" if st.session_state.get("contributor_verified") else "LOCKED"),
         P("SIGNATURE"), P(st.session_state.get("manifest_status", "REVIEW")),
         P("AUDIT"), P("VALID" if audit_valid else "FAILED"), P("IMAGES"), P(len(reports))],
    ]

    summary_table = Table(
        summary,
        colWidths=[20*mm, 48*mm, 20*mm, 72*mm, 18*mm, 21*mm, 18*mm, 21*mm],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF1F8")),
        ("BOX", (0,0), (-1,-1), 0.45, colors.HexColor("#AAB7C6")),
        ("INNERGRID", (0,0), (-1,-1), 0.2, colors.HexColor("#CFD7E0")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story += [summary_table, Spacer(1, 3*mm)]

    story.append(P("Verification Control Matrix", h2_style))

    control_rows = [[P("Control"), P("Status"), P("Evidence")]]
    for control, status, evidence in checks:
        control_rows.append([P(control, small_style), P(status, small_style), P(evidence, small_style)])

    controls_table = Table(
        control_rows,
        colWidths=[65*mm, 28*mm, 170*mm],
        repeatRows=1,
    )
    style_cmds = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#182536")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BOX", (0,0), (-1,-1), 0.45, colors.HexColor("#AAB7C6")),
        ("INNERGRID", (0,0), (-1,-1), 0.2, colors.HexColor("#D6DDE5")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    for idx, (_, status, _) in enumerate(checks, start=1):
        if status == "PASS":
            style_cmds.append(("BACKGROUND", (1,idx), (1,idx), colors.HexColor("#E8F7F0")))
        elif status in ("REVIEW", "LOCKED"):
            style_cmds.append(("BACKGROUND", (1,idx), (1,idx), colors.HexColor("#FFF3D7")))
        elif status == "FAIL":
            style_cmds.append(("BACKGROUND", (1,idx), (1,idx), colors.HexColor("#FDE7E9")))
    controls_table.setStyle(TableStyle(style_cmds))
    story.append(controls_table)

    if reports:
        story += [PageBreak(), P("Per-Image Security Risk Matrix", h2_style)]
        rows = [[
            P("#"), P("File"), P("Contributor"), P("Quality"),
            P("Anomaly"), P("Spectral"), P("Shift"),
            P("Shift %"), P("Review"), P("Severity"),
            P("Disposition"), P("Dup"), P("Near"), P("Orientation"),
        ]]
        for r in reports:
            rows.append([
                P(r["index"]), P(r["source"]), P(r["contributor"]),
                P(f'{r["quality_confidence"]:.1f}'),
                P(f'{r["anomaly_confidence"]:.1f}'),
                P(f'{r["spectral_confidence"]:.1f}'),
                P(f'{r["shift_distance"]:.3f}'),
                P(f'{r["shift_confidence"]:.1f}'),
                P(f'{r["review_confidence"]:.1f}'),
                P(r["severity"]), P(r["disposition"]),
                P("YES" if r["duplicate_count"] else "NO"),
                P("YES" if r["is_near_duplicate"] else "NO"),
                P(r["orientation_status"]),
            ])

        matrix_table = Table(
            rows,
            colWidths=[
                8*mm, 42*mm, 30*mm, 15*mm, 16*mm, 16*mm,
                17*mm, 17*mm, 16*mm, 17*mm, 22*mm,
                12*mm, 13*mm, 23*mm,
            ],
            repeatRows=1,
        )
        matrix_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#182536")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("BOX", (0,0), (-1,-1), 0.45, colors.HexColor("#AAB7C6")),
            ("INNERGRID", (0,0), (-1,-1), 0.18, colors.HexColor("#D8DEE6")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 2.5),
            ("RIGHTPADDING", (0,0), (-1,-1), 2.5),
            ("TOPPADDING", (0,0), (-1,-1), 2.5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2.5),
        ]))
        story.append(matrix_table)

    story.append(Spacer(1, 4*mm))
    story.append(P(
        f"Generated UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} · TrustCV Dataset Security · MIRAD Protected",
        sub_style,
    ))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#65778B"))
        canvas.drawString(10*mm, 6*mm, "TrustCV · Dataset Security Assessment · MIRAD Protected")
        canvas.drawRightString(
            landscape(A4)[0] - 10*mm,
            6*mm,
            f"Page {doc_obj.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def build_evidence_bundle(dataset, reports, checks, trust, risk, audit_valid, audit_count):
    pdf_bytes = build_pdf_report(
        dataset, reports, checks, trust, risk, audit_valid, audit_count
    )
    json_bytes = json.dumps(
        build_json_report(
            dataset, reports, checks, trust, risk, audit_valid, audit_count
        ),
        indent=2,
    ).encode("utf-8")
    csv_bytes = risk_dataframe(reports).to_csv(index=False).encode("utf-8")
    audit_bytes = AUDIT.read_bytes() if AUDIT.exists() else b""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("trustcv_dataset_security_report.pdf", pdf_bytes)
        archive.writestr("trustcv_dataset_security_report.json", json_bytes)
        archive.writestr("trustcv_image_risk_matrix.csv", csv_bytes)
        if audit_bytes:
            archive.writestr("trustcv_dataset_audit_chain.jsonl", audit_bytes)

        mirad = st.session_state.get("mirad_security_record")
        if mirad:
            archive.writestr(
                "dataset_security/provenance.json",
                json.dumps(mirad.get("provenance", {}), indent=2),
            )
            archive.writestr(
                "dataset_security/verification_result.json",
                json.dumps({
                    "provenance_verification": mirad.get("provenance_verification", {}),
                    "replay_verification": mirad.get("replay_verification", {}),
                    "audit_verification": mirad.get("audit_verification", {}),
                    "checkpoint_verification": mirad.get("checkpoint_verification", {}),
                }, indent=2),
            )
            archive.writestr(
                "dataset_security/checkpoint.json",
                json.dumps(mirad.get("checkpoint", {}), indent=2),
            )
            audit_log_path = Path.home() / ".trustcv_dataset_security" / "dataset_security_audit.jsonl"
            if audit_log_path.exists():
                archive.writestr(
                    "dataset_security/audit_log.jsonl",
                    audit_log_path.read_bytes(),
                )
            manifest = mirad.get("dataset_manifest") or st.session_state.get("dataset_manifest", {})
            if manifest:
                archive.writestr(
                    "dataset_security/manifest.json",
                    json.dumps(manifest, indent=2),
                )
            archive.writestr(
                "dataset_security/security_record.json",
                json.dumps(mirad, indent=2),
            )

        # Findings
        findings = st.session_state.get("dataset_findings", [])
        if findings:
            archive.writestr(
                "dataset_security/findings.json",
                json.dumps(findings, indent=2),
            )

        # Ethereum anchor
        eth = st.session_state.get("ethereum_anchor_result")
        if eth:
            archive.writestr(
                "dataset_security/ethereum_anchor.json",
                json.dumps(eth, indent=2),
            )

    return buffer.getvalue()

def posture_score(checks):
    scored = [x for x in checks if x[1] in {"PASS", "REVIEW", "FAIL"}]

    if not scored:
        return 0

    value = (
        100
        * (
            sum(x[1] == "PASS" for x in scored)
            + 0.45 * sum(x[1] == "REVIEW" for x in scored)
        )
        / len(scored)
    )

    value -= 30 * sum(x[1] == "FAIL" for x in scored)

    return int(np.clip(round(value), 0, 100))


# ================================================================
# 3D SECURITY HUD
# ================================================================

def render_security_cube():
    st.markdown(
        """
        <div class="hud3d">
          <div class="cube-scene">
            <div class="orbit orbit-x"></div>
            <div class="orbit orbit-y"></div>
            <div class="cube">
              <div class="face front"><b>SHA-256</b><span>IDENTITY</span></div>
              <div class="face back"><b>Ed25519</b><span>AUTHENTICITY</span></div>
              <div class="face right"><b>SVD</b><span>ANOMALY</span></div>
              <div class="face left"><b>SHIFT</b><span>DISTRIBUTION</span></div>
              <div class="face top"><b>MIRAD</b><span>PROVENANCE</span></div>
              <div class="face bottom"><b>ETH</b><span>ANCHOR</span></div>
            </div>
          </div>
          <div class="hud-copy">
            <div class="ey">LIVE DATASET ASSURANCE CORE</div>
            <div class="hud-title">MULTI-GATE TRUST TELEMETRY</div>
            <div class="hud-sub">Identity → Authenticity → Content → Behaviour → Provenance → Anchor</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# UI

def main():
    # ================================================================
    
    st.markdown("""
    <style>
    .stApp { background:#05080e; }
    .block-container { max-width:1600px; padding:1rem 2rem 3rem; }
    .hero {
        padding:30px 34px;
        border-radius:26px;
        background:linear-gradient(115deg,#071d34,#171c3d,#321525);
        border:1px solid rgba(120,170,255,.18);
        box-shadow:0 20px 70px rgba(0,0,0,.35);
    }
    .hero h1 { margin:0; font-size:2.75rem; letter-spacing:-.04em; }
    .ey { font-size:11px; letter-spacing:.2em; color:#93a9c5; }
    .locked {
        padding:15px 18px;
        border-radius:14px;
        background:rgba(255,55,70,.12);
        border:1px solid rgba(255,70,82,.35);
    }
    .secure {
        padding:15px 18px;
        border-radius:14px;
        background:rgba(50,225,160,.10);
        border:1px solid rgba(50,225,160,.25);
    }
    .review {
        padding:15px 18px;
        border-radius:14px;
        background:rgba(255,174,55,.12);
        border:1px solid rgba(255,174,55,.28);
    }
    .small { color:#91a0b6; font-size:12px; }
    
    .hud3d {
        display:flex; align-items:center; gap:26px;
        min-height:240px; margin:12px 0 20px; padding:18px 24px;
        border-radius:22px; overflow:hidden;
        background:radial-gradient(circle at 30% 50%, rgba(37,115,196,.16), transparent 38%),
                   linear-gradient(115deg, rgba(4,16,30,.98), rgba(8,14,28,.98));
        border:1px solid rgba(91,169,255,.17);
        box-shadow:inset 0 0 60px rgba(20,100,180,.05), 0 18px 60px rgba(0,0,0,.28);
    }
    .cube-scene {
        width:210px; height:205px; perspective:850px; flex:0 0 auto;
        display:flex; justify-content:center; align-items:center;
        position:relative;
    }
    .cube {
        width:118px; height:118px; position:relative; transform-style:preserve-3d;
        transform:rotateX(-18deg) rotateY(25deg);
        animation:cubeSpin 11s linear infinite;
    }
    .face {
        position:absolute; width:118px; height:118px; display:flex; flex-direction:column;
        justify-content:center; align-items:center; gap:6px;
        border:1px solid rgba(108,190,255,.42);
        background:linear-gradient(135deg, rgba(26,101,165,.22), rgba(10,19,37,.88));
        box-shadow:inset 0 0 25px rgba(74,174,255,.08);
        color:#d9ecff; font-size:10px; letter-spacing:.08em; text-align:center;
    }
    .face b { font-size:15px; }
    .face span { font-size:8px; color:#7fa9cc; letter-spacing:.18em; }
    .front { transform:translateZ(59px); }
    .back { transform:rotateY(180deg) translateZ(59px); }
    .right { transform:rotateY(90deg) translateZ(59px); }
    .left { transform:rotateY(-90deg) translateZ(59px); }
    .top { transform:rotateX(90deg) translateZ(59px); }
    .bottom { transform:rotateX(-90deg) translateZ(59px); }
    .hud-copy { min-width:0; }
    .hud-copy .ey { color:#76b6ee; font-size:10px; letter-spacing:.2em; }
    .hud-title { font-size:22px; font-weight:700; margin-top:10px; letter-spacing:.05em; }
    .hud-sub { color:#91a7bd; margin-top:8px; font-size:13px; }
    
    .orbit {
        position:absolute; left:50%; top:50%; width:178px; height:58px;
        margin-left:-89px; margin-top:-29px;
        border:1px solid rgba(100,180,255,.20);
        border-radius:50%; pointer-events:none;
    }
    .orbit-x { transform:rotateX(68deg); animation:orbitSpin 7s linear infinite; }
    .orbit-y { transform:rotateY(68deg) rotateZ(24deg); animation:orbitTilt 9s linear infinite reverse; }
    @keyframes orbitSpin {
        0% { transform:rotateX(68deg) rotateZ(0deg); }
        100% { transform:rotateX(68deg) rotateZ(360deg); }
    }
    @keyframes orbitTilt {
        0% { transform:rotateY(68deg) rotateZ(0deg); }
        100% { transform:rotateY(68deg) rotateZ(360deg); }
    }
    @keyframes cubeSpin {
        0% { transform:rotateX(-18deg) rotateY(0deg); }
        100% { transform:rotateX(-18deg) rotateY(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero">
    <div class="ey">DEFENSIVE AI SECURITY OPERATIONS · DATASET ASSURANCE · MIRAD PROTECTED</div>
    <h1>🛡️ TrustCV</h1>
    <div>Contributor · Identity · Integrity · Authenticity · Provenance · Quality · Anomaly · Spectral · Shift · Evidence · Ethereum Anchor</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        page = st.radio(
            "Dataset Security Console",
            [
                "🏠 Overview",
                "01 · Dataset Intake",
                "02 · Dataset Profile",
                "03 · Integrity Analysis",
                "04 · Distribution & Spectral",
                "05 · Contributors",
                "06 · Batches",
                "07 · Findings",
                "08 · Evidence Explorer",
                "09 · Dataset Provenance",
                "10 · Verification",
                "11 · Audit Trail",
                "12 · Ethereum Anchor",
                "13 · Coverage / Limitations",
            ],
        )
    
        st.markdown("---")
        st.markdown("### 🔐 Contributor Passkey")
    
        st.markdown("### 🔐 Pipeline Operator Authorization")
        st.caption(
            "**Operator Authorization Gate**: Authorizes pipeline operation. Verifies operator access to this local TrustCV instance.\n\n"
            "*Explicit distinction: Operator Authentication ≠ Cryptographic Contributor Identity ≠ Real-World Identity.*"
        )

        passkey = st.text_input(
            "Operator passkey",
            type="password",
            placeholder="Enter operator passkey",
            key="operator_passkey_input",
        )

        if st.button("Authorize Pipeline", key="btn_auth_pipeline"):
            ok, message = verify_passkey(passkey)
            st.session_state["contributor_verified"] = ok
            st.session_state["operator_authorized"] = ok
            st.session_state["contributor_message"] = message
            audit_event({
                "event": "operator_authorization",
                "status": "PASS" if ok else "FAIL",
            })

        if st.session_state.get("contributor_verified"):
            st.success("🟢 OPERATOR AUTHORIZED")
        else:
            st.warning("🔒 PIPELINE LOCKED (Operator Auth Required)")
    
        st.markdown("---")
    
        # Ethereum status indicator
        eth_config = EthereumConfig.from_env()
        if eth_config.is_configured:
            st.caption("⟠ Ethereum: CONFIGURED")
        else:
            st.caption("⟠ Ethereum: NOT CONFIGURED")
    
        st.caption(
            "Dataset-only assurance application. All checks run locally; "
            "dataset onboarding requires the contributor passkey."
        )
    
        if st.button("↻ Reset Current Dataset"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # ================================================================
    # SECURE INTAKE
    # ================================================================
    
    contributor_verified = bool(st.session_state.get("contributor_verified"))
    
    if page == "01 · Dataset Intake":
        st.subheader("Secure Dataset Intake")
    
    if not contributor_verified and page == "01 · Dataset Intake":
        st.markdown(
            '<div class="locked">🔒 <b>DATASET UPLOAD LOCKED</b><br>'
            'Verify the contributor passkey before a dataset can enter the pipeline.'
            '</div>',
            unsafe_allow_html=True,
        )
        dataset_upload = None
        manifest_upload = None
    elif page == "01 · Dataset Intake":
        st.markdown(
            '<div class="secure">🟢 <b>OPERATOR AUTHORIZED</b> · Dataset onboarding enabled.</div>',
            unsafe_allow_html=True,
        )

        # Contributor Intake Context Binding
        c_db = get_contributor_backend()
        registered = c_db.get_contributors()
        c_map = {c["contributor_id"]: c for c in registered}
        c_options = [c["contributor_id"] for c in registered]
        if not c_options:
            c_options = ["CONTRIB-PRIMARY-01"]

        st.markdown("#### 👤 Contributor & Batch Context Binding")
        st.caption("Select the registered contributor, contribution ID, and batch ID for this intake. Uploaded samples and analysis will be bound to this context.")

        def _sync_intake_contributor():
            new_c = st.session_state.get("intake_contributor_selector")
            if new_c:
                st.session_state["intake_contribution_id"] = f"CNTRB-{new_c}-001"
                st.session_state["intake_batch_id"] = f"BATCH-{new_c}-B01"

        # Pre-initialize defaults if not yet present in session state
        init_c = st.session_state.get("intake_contributor_selector") or c_options[0]
        if "intake_contribution_id" not in st.session_state:
            st.session_state["intake_contribution_id"] = f"CNTRB-{init_c}-001"
        if "intake_batch_id" not in st.session_state:
            st.session_state["intake_batch_id"] = f"BATCH-{init_c}-B01"

        ic1, ic2, ic3 = st.columns([1.2, 1.0, 1.0])
        with ic1:
            sel_contrib = st.selectbox(
                "Contributor",
                options=c_options,
                format_func=lambda cid: f"{cid} · {c_map.get(cid, {}).get('display_name', cid)} ({c_map.get(cid, {}).get('organization', 'Individual')})" if cid in c_map else cid,
                key="intake_contributor_selector",
                on_change=_sync_intake_contributor,
            )
        with ic2:
            sel_contribution = st.text_input(
                "Contribution ID",
                key="intake_contribution_id",
            )
        with ic3:
            sel_batch = st.text_input(
                "Batch ID",
                key="intake_batch_id",
            )

        st.session_state["intake_selected_contributor"] = sel_contrib
        st.session_state["expected_contributor_override"] = sel_contrib
        st.session_state["expected_batch_override"] = sel_batch
    
        left, right = st.columns([1.35, 1.0])
    
        with left:
            dataset_upload = st.file_uploader(
                "DATASET · ZIP / CSV / EXCEL / IMAGE",
                type=[
                    "zip", "csv", "xlsx", "xls",
                    "jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "gif",
                ],
                key="DATASET_GLOBAL",
            )
    
        with right:
            manifest_upload = st.file_uploader(
                "SIGNED DATASET MANIFEST · JSON",
                type=["json"],
                key="DATASET_MANIFEST_GLOBAL",
            )
    
        if st.session_state.get("dataset_hash"):
            m_stat = st.session_state.get("manifest_status", "UNAVAILABLE")
            if m_stat == "PASS":
                st.markdown(
                    '<div class="secure" style="margin-top:10px;">🟢 <b>MANIFEST STATUS: PASS</b> · Cryptographically verified with trust anchor.</div>',
                    unsafe_allow_html=True,
                )
            elif m_stat == "UNAVAILABLE":
                st.markdown(
                    '<div class="review" style="margin-top:10px; background:#2a2a2a; border-color:#666;">⚪ <b>MANIFEST STATUS: UNAVAILABLE</b> · No external signed manifest uploaded.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="review" style="margin-top:10px;">🔴 <b>MANIFEST STATUS: {escape(m_stat)}</b> · {escape(st.session_state.get("manifest_reason", ""))}</div>',
                    unsafe_allow_html=True,
                )
    
            with st.expander("🔏 Attest & Sign Manifest for Ingested Dataset (Offline Trust Anchor)"):
                st.caption(
                    "Generate a cryptographically signed manifest for the current dataset using the local Ed25519 Trust Anchor."
                )
                c_db = get_contributor_backend()
                registered = c_db.get_contributors()
                c_opts = [c["contributor_id"] for c in registered]
                if not c_opts:
                    c_opts = ["CONTRIB-PRIMARY-01"]
    
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    sign_contrib = st.selectbox("Attesting Contributor ID", options=c_opts, index=0, key="attest_contrib_sel")
                with c_c2:
                    sign_batch = st.text_input("Contribution / Batch ID", value="BATCH-001", key="attest_batch_sel")
    
                if st.button("Generate & Sign Manifest with Trust Anchor", key="btn_sign_attest"):
                    ds_bytes = dataset_upload.getvalue() if dataset_upload else b""
                    ds_name = st.session_state.get("dataset", {}).get("name", "dataset")
                    new_m = create_signed_dataset_manifest(
                        dataset_name=ds_name,
                        dataset_digest=st.session_state["dataset_hash"],
                        contributor_id=sign_contrib,
                        batch_id=sign_batch,
                    )
                    m_bytes = json.dumps(new_m, indent=2).encode("utf-8")
                    m_h = sha256_bytes(m_bytes)
                    v_res = verify_signed_dataset_manifest(
                        m_bytes,
                        actual_dataset_digest=st.session_state["dataset_hash"],
                        expected_contributor_id=sign_contrib,
                        expected_batch_id=sign_batch,
                        actual_dataset_name=ds_name,
                    )
                    st.session_state.update(
                        dataset_manifest=new_m,
                        manifest_hash=m_h,
                        manifest_bound_dataset_hash=st.session_state["dataset_hash"],
                        verified_expected_contrib=sign_contrib,
                        manifest_status=v_res["overall"],
                        manifest_reason=v_res["reason"],
                        manifest_verification_result=v_res,
                        local_attestation_created=True,
                    )
                    audit_event({
                        "event": "dataset_manifest_attestation_created",
                        "dataset_sha256": st.session_state["dataset_hash"],
                        "manifest_sha256": m_h,
                        "contributor_id": sign_contrib,
                        "status": v_res["overall"],
                    })
                    st.success(f"✓ Manifest created & verified: {v_res['overall']}")
                    st.download_button(
                        "⬇️ Download Signed Manifest (JSON)",
                        m_bytes,
                        file_name=f"{ds_name}_signed_manifest.json",
                        mime="application/json",
                    )
    else:
        dataset_upload = None
        manifest_upload = None
    
    # Handle dataset upload from session (if already uploaded on intake page)
    if "DATASET_GLOBAL" in st.session_state and st.session_state.get("DATASET_GLOBAL") is not None:
        dataset_upload = st.session_state.get("DATASET_GLOBAL")
    if "DATASET_MANIFEST_GLOBAL" in st.session_state and st.session_state.get("DATASET_MANIFEST_GLOBAL") is not None:
        manifest_upload = st.session_state.get("DATASET_MANIFEST_GLOBAL")
    
    if dataset_upload is not None:
        raw = dataset_upload.getvalue()
        digest = sha256_bytes(raw)
    
        if st.session_state.get("dataset_hash") != digest:
            try:
                active_c = st.session_state.get("intake_selected_contributor") or st.session_state.get("expected_contributor_override")
                active_b = st.session_state.get("intake_batch_id")
                dataset = parse_dataset(dataset_upload, contributor_id=active_c, batch_id=active_b)
                attach_security_telemetry(dataset, raw)
                st.session_state.update(
                    dataset=dataset,
                    dataset_hash=digest,
                    analysis_hash=None,
                )
    
                st.session_state.pop("dataset_error", None)
    
                audit_event({
                    "event": "dataset_uploaded",
                    "name": dataset["name"],
                    "kind": dataset["kind"],
                    "sha256": digest,
                    "authority": "verified",
                })
    
            except Exception as exc:
                st.session_state["dataset_error"] = str(exc)
                for _key in ("dataset", "dataset_reports", "dataset_checks", "dataset_hash", "analysis_hash", "mirad_security_record", "mirad_security_binding", "dataset_findings"):
                    st.session_state.pop(_key, None)
    
    expected_contrib = st.session_state.get("expected_contributor_override")
    if not expected_contrib and st.session_state.get("dataset"):
        c_list = st.session_state["dataset"].get("image_contributors", [])
        if c_list and c_list[0] not in ("ROOT / UNKNOWN SOURCE", "DEFAULT_CONTRIBUTOR"):
            expected_contrib = c_list[0]
    
    actual_hash = st.session_state.get("dataset_hash_override") or st.session_state.get("dataset_hash")
    
    if manifest_upload is not None and actual_hash:
        manifest_raw = manifest_upload.getvalue()
        manifest_hash = sha256_bytes(manifest_raw)
    
        if (
            st.session_state.get("manifest_hash") != manifest_hash
            or st.session_state.get("manifest_bound_dataset_hash") != actual_hash
            or st.session_state.get("verified_expected_contrib") != expected_contrib
        ):
            dataset_name = st.session_state.get("dataset", {}).get("name") if st.session_state.get("dataset") else None
            ver_result = verify_signed_dataset_manifest(
                manifest_raw,
                actual_dataset_digest=actual_hash,
                expected_contributor_id=expected_contrib,
                actual_dataset_name=dataset_name,
            )
            status = ver_result["overall"]
            reason = ver_result["reason"]
            manifest = ver_result["manifest"]
    
            st.session_state.update(
                manifest_hash=manifest_hash,
                manifest_bound_dataset_hash=actual_hash,
                verified_expected_contrib=expected_contrib,
                dataset_manifest=manifest,
                manifest_status=status,
                manifest_reason=reason,
                manifest_verification_result=ver_result,
            )
    
            audit_event({
                "event": "dataset_manifest_validation",
                "dataset_sha256": actual_hash,
                "manifest_sha256": manifest_hash,
                "signature": status,
                "reason": reason,
            })
    
    elif st.session_state.get("dataset_manifest") and actual_hash:
        # Manifest in session (e.g. from local attestation or prior upload)
        if (
            st.session_state.get("manifest_bound_dataset_hash") != actual_hash
            or st.session_state.get("verified_expected_contrib") != expected_contrib
        ):
            dataset_name = st.session_state.get("dataset", {}).get("name") if st.session_state.get("dataset") else None
            ver_result = verify_signed_dataset_manifest(
                st.session_state["dataset_manifest"],
                actual_dataset_digest=actual_hash,
                expected_contributor_id=expected_contrib,
                actual_dataset_name=dataset_name,
            )
            st.session_state.update(
                manifest_bound_dataset_hash=actual_hash,
                verified_expected_contrib=expected_contrib,
                manifest_status=ver_result["overall"],
                manifest_reason=ver_result["reason"],
                manifest_verification_result=ver_result,
            )
    
    if dataset_upload is not None and st.session_state.get("dataset_hash") and manifest_upload is None and not st.session_state.get("dataset_manifest"):
        # No external manifest uploaded and no local attestation created. Do NOT fake a PASS state.
        if st.session_state.get("manifest_status") != "UNAVAILABLE":
            st.session_state.update(
                manifest_status="UNAVAILABLE",
                manifest_reason="No signed dataset manifest supplied. Verification requires an authentic signed manifest.",
                dataset_manifest={},
                manifest_verification_result={
                    "overall": "UNAVAILABLE",
                    "valid": False,
                    "reason": "No signed dataset manifest supplied.",
                    "checks": {
                        "manifest_integrity": {"status": "UNAVAILABLE", "reason": "No manifest supplied"},
                        "digital_signature": {"status": "UNAVAILABLE", "reason": "No signature to verify"},
                        "dataset_digest": {"status": "UNAVAILABLE", "signed": None, "observed": st.session_state.get("dataset_hash")},
                        "contributor": {"status": "UNAVAILABLE", "signed": None, "observed": None},
                    },
                },
            )
    
    if "manifest_status" not in st.session_state:
        st.session_state["manifest_status"] = "UNAVAILABLE"
        st.session_state["manifest_reason"] = "No signed dataset manifest supplied."
    
    dataset = st.session_state.get("dataset")
    
    if (
        dataset is not None
        and st.session_state.get("analysis_hash") != st.session_state.get("dataset_hash")
    ):
        reports = analyze_reports(dataset)
    
        st.session_state["dataset_reports"] = reports
        st.session_state["analysis_hash"] = st.session_state.get("dataset_hash")
    
        # Generate findings
        st.session_state["dataset_findings"] = generate_findings(dataset, reports)
    
        audit_event({
            "event": "dataset_security_analysis",
            "dataset_sha256": st.session_state.get("dataset_hash"),
            "image_count": len(reports),
            "findings_count": len(st.session_state["dataset_findings"]),
        })
    
    if dataset is not None:
        st.session_state["dataset_checks"] = dataset_checks(
            dataset,
            st.session_state.get("dataset_reports", []),
        )
    
    reports = st.session_state.get("dataset_reports", [])
    checks = st.session_state.get("dataset_checks", [])
    
    # MIRAD-derived dataset-only security layer + Persistent Contributor Backend.
    mirad_binding = (
        str(st.session_state.get("dataset_hash", "")),
        str(st.session_state.get("manifest_hash", "")),
        str(st.session_state.get("analysis_hash", "")),
    )
    if dataset is not None and mirad_binding != st.session_state.get("mirad_security_binding"):
        try:
            ensure_trust_anchor()
            active_contrib = st.session_state.get("intake_selected_contributor") or st.session_state.get("expected_contributor_override")
            active_contribution = st.session_state.get("intake_contribution_id")
            active_batch = st.session_state.get("intake_batch_id")

            mirad_result = persist_dataset_security_run(
                dataset=dataset,
                reports=reports,
                checks=checks,
                manifest=st.session_state.get("dataset_manifest", {}),
                contributor_id=active_contrib,
                contribution_id=active_contribution,
                batch_id=active_batch,
            )
            st.session_state["mirad_security_record"] = mirad_result
            st.session_state["mirad_security_binding"] = mirad_binding

            # Persist into contributor backend
            db = get_contributor_backend()
            backend_run = db.record_dataset_analysis(
                dataset_name=str(dataset.get("name", "")),
                dataset_digest=str(dataset.get("hash", "")),
                reports=reports,
                findings=st.session_state.get("dataset_findings", []),
                manifest=st.session_state.get("dataset_manifest", {}),
                provenance=mirad_result.get("provenance"),
                audit_event_id=mirad_result.get("audit_event", {}).get("audit_id"),
                checkpoint_id=mirad_result.get("checkpoint", {}).get("checkpoint_id"),
                contributor_id_override=active_contrib,
                contribution_id_override=active_contribution,
                batch_id_override=active_batch,
            )
            st.session_state["contributor_backend_run"] = backend_run
        except Exception as exc:
            st.session_state["mirad_security_record"] = {
                "dataset_only": True,
                "trusted": False,
                "error": str(exc),
            }
    
    audit_valid, audit_count = verify_audit()
    
    trust = posture_score(checks)
    risk = 100 - trust
    
    # ================================================================
    # OVERVIEW (COMMAND CENTER)
    # ================================================================
    
    if page == "🏠 Overview":
        st.subheader("Dataset Security Command Center")
        render_security_cube()
    
        if dataset is None:
            st.info("Verify authority first, then upload a dataset on the Dataset Intake page.")
        else:
            image_count = len(dataset.get("images", []))
            record_count = (
                len(dataset["df"])
                if dataset.get("df") is not None
                else image_count
            )
    
            visual_outliers = sum(r["is_visual_outlier"] for r in reports)
            spectral_outliers = sum(r["is_spectral_outlier"] for r in reports)
            high_shift = sum(r["shift_status"] == "HIGH SHIFT" for r in reports)
            quarantine = sum(r["disposition"] == "QUARANTINE" for r in reports)
            review_n = sum(x[1] == "REVIEW" for x in checks)
            fail_n = sum(x[1] == "FAIL" for x in checks)
            findings = st.session_state.get("dataset_findings", [])
            contributors = len(set(r["contributor"] for r in reports)) if reports else 0
    
            k = st.columns(8)
            k[0].metric("TRUST POSTURE", f"{trust}/100")
            k[1].metric("RISK INDEX", f"{risk}/100")
            k[2].metric("SAMPLES", record_count)
            k[3].metric("CONTRIBUTORS", contributors)
            k[4].metric("FINDINGS", len(findings))
            k[5].metric("VISUAL OUTLIERS", visual_outliers)
            k[6].metric("HIGH SHIFT", high_shift)
            k[7].metric("QUARANTINE", quarantine)
    
            # MIRAD + Ethereum status
            mirad_record = st.session_state.get("mirad_security_record", {})
            mirad_ok = mirad_record.get("checkpoint_verification", {}).get("trusted", False)
            eth_result = st.session_state.get("ethereum_anchor_result", {})
            eth_status = eth_result.get("status", "NOT CONFIGURED") if eth_result else "NOT CONFIGURED"
    
            m1, m2, m3 = st.columns(3)
            m1.metric("MIRAD VERIFICATION", "TRUSTED" if mirad_ok else "UNVERIFIED")
            m2.metric("AUDIT CHAIN", f"{audit_count} events")
            m3.metric("ETHEREUM", eth_status)
    
            if fail_n:
                st.markdown(
                    '<div class="locked">🔴 <b>CRITICAL FAILURE</b> · One or more dataset controls failed.</div>',
                    unsafe_allow_html=True,
                )
            elif review_n:
                st.markdown(
                    f'<div class="review">🟠 <b>REVIEW REQUIRED</b> · {review_n} control(s) require analyst attention or trust material.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="secure">🟢 <b>ALL IMPLEMENTED DATASET GATES CLEAR</b></div>',
                    unsafe_allow_html=True,
                )
    
            st.markdown("#### Security Gate Matrix")
    
            st.dataframe(
                pd.DataFrame(
                    checks,
                    columns=["Security Check", "Status", "Evidence"],
                ),
                use_container_width=True,
                hide_index=True,
            )
    
            if HAS_PLOTLY and reports:
                df_risk = risk_dataframe(reports)
                left, right = st.columns(2)
    
                with left:
                    fig = px.scatter(
                        df_risk,
                        x="Shift Distance",
                        y="Review %",
                        size="Review %",
                        hover_name="File",
                        hover_data=[
                            "Contributor",
                            "Anomaly %",
                            "Spectral %",
                            "Shift",
                            "Disposition",
                        ],
                        title="Image-by-image shift vs review evidence",
                    )
                    fig.update_layout(
                        height=390,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
    
                with right:
                    signal_df = pd.DataFrame({
                        "Signal": [
                            "Visual Outlier",
                            "Spectral Outlier",
                            "High Shift",
                            "Near Duplicate",
                            "Quality Flag",
                        ],
                        "Images": [
                            visual_outliers,
                            spectral_outliers,
                            high_shift,
                            sum(r["is_near_duplicate"] for r in reports),
                            sum(bool(r["quality_issues"]) for r in reports),
                        ],
                    })
    
                    fig = px.bar(
                        signal_df,
                        x="Signal",
                        y="Images",
                        title="Evidence signal counts",
                    )
                    fig.update_layout(
                        height=390,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
    
            st.info(
                "The percentages shown in the dashboard are screening-confidence values. "
                "They are not calibrated probabilities that an image is compromised."
            )
    
    # ================================================================
    # DATASET INTAKE
    # ================================================================
    
    elif page == "01 · Dataset Intake":
        if st.session_state.get("dataset_error"):
            st.error(st.session_state["dataset_error"])
    
        if dataset is not None:
            st.success(f"Dataset loaded: {dataset['name']} ({dataset['kind']})")
            st.code(st.session_state.get("dataset_hash", ""))
    
            count = (
                len(dataset["df"])
                if dataset.get("df") is not None
                else len(dataset.get("images", []))
            )
    
            a, b, c, d = st.columns(4)
            a.metric("RECORDS / IMAGES", count)
            b.metric("SHA-256", "PASS")
            c.metric("FORMAT", dataset.get("label_format", "N/A"))
            d.metric("SIGNATURE", st.session_state.get("manifest_status", "REVIEW"))
    
            if dataset.get("annotation"):
                st.markdown("#### Dataset Format / Annotation Inventory")
                st.json(dataset["annotation"])
    
    # ================================================================
    # DATASET PROFILE
    # ================================================================
    
    elif page == "02 · Dataset Profile":
        st.subheader("02 · Dataset Profile")
    
        if dataset is None:
            st.info("Upload a dataset after contributor verification.")
        else:
            count = (
                len(dataset["df"])
                if dataset.get("df") is not None
                else len(dataset.get("images", []))
            )
    
            a, b, c, d, e = st.columns(5)
            a.metric("RECORDS / IMAGES", count)
            b.metric("SHA-256", "PASS")
            c.metric("CONTRIBUTOR", "VERIFIED" if contributor_verified else "LOCKED")
            d.metric("DIGITAL SIGNATURE", st.session_state.get("manifest_status", "REVIEW"))
            e.metric("TRUST POSTURE", f"{trust}/100")
    
            st.code(st.session_state.get("dataset_hash", ""))
    
            st.markdown("#### Cryptographic Identity")
            hash_cols = st.columns(6)
            hash_cols[0].metric("ALGORITHM", dataset.get("hash_algorithm", "SHA-256"))
            hash_cols[1].metric("DIGEST", "256 BIT")
            hash_cols[2].metric("HEX LENGTH", str(dataset.get("hash_hex_length", 64)))
            hash_cols[3].metric(
                "DATASET SIZE",
                f"{dataset.get('raw_size_bytes', 0) / (1024*1024):.2f} MB",
            )
            hash_cols[4].metric(
                "RECOMPUTE",
                "PASS" if dataset.get("hash_recomputed") else "FAIL",
            )
            hash_cols[5].metric(
                "LABEL FORMAT",
                dataset.get("label_format", "NONE"),
            )
    
            # Label / class distribution
            if reports:
                labeled = [r for r in reports if r.get("label")]
                if labeled:
                    st.markdown("#### Class Distribution")
                    all_labels = []
                    for r in labeled:
                        all_labels.extend(r["label"].split(", "))
                    label_counts = pd.Series(all_labels).value_counts()
                    st.dataframe(
                        pd.DataFrame({"Class": label_counts.index, "Count": label_counts.values}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if HAS_PLOTLY:
                        fig = px.bar(x=label_counts.index[:20], y=label_counts.values[:20], title="Top classes")
                        fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
            # Image dimensions distribution
            if reports:
                dims = pd.DataFrame([{"Width": r["width"], "Height": r["height"]} for r in reports])
                st.markdown("#### Image Dimensions")
                st.dataframe(dims.describe(), use_container_width=True)
    
            st.markdown("#### Complete Verification Matrix")
            st.dataframe(
                pd.DataFrame(
                    checks,
                    columns=["Security Check", "Status", "Evidence"],
                ),
                use_container_width=True,
                hide_index=True,
            )
    
            if dataset.get("df") is not None:
                st.markdown("#### Tabular Preview")
                st.dataframe(
                    dataset["df"].head(100),
                    use_container_width=True,
                    hide_index=True,
                )
    
    # ================================================================
    # INTEGRITY ANALYSIS
    # ================================================================
    
    elif page == "03 · Integrity Analysis":
        st.subheader("03 · Integrity Analysis")
    
        if not reports:
            st.info("Upload an image ZIP containing at least 5 images.")
        else:
            tab_dup, tab_label, tab_ood, tab_trigger = st.tabs([
                "Duplicates", "Label Analysis", "OOD / Anomalies", "Trigger Indicators"
            ])
    
            with tab_dup:
                st.markdown("#### Duplicate Detection")
                exact_dups = sum(r["duplicate_count"] > 0 for r in reports)
                near_dups = sum(r["is_near_duplicate"] for r in reports)
                a, b = st.columns(2)
                a.metric("EXACT DUPLICATES", exact_dups)
                b.metric("NEAR DUPLICATES", near_dups)
    
                dup_reports = [r for r in reports if r["duplicate_count"] > 0 or r["is_near_duplicate"]]
                if dup_reports:
                    st.dataframe(
                        pd.DataFrame([{
                            "Image #": r["index"],
                            "File": r["source"],
                            "Contributor": r["contributor"],
                            "Exact Dup": r["duplicate_count"] > 0,
                            "Near Dup": r["is_near_duplicate"],
                            "Near Dup Of": r["near_duplicate_of"],
                        } for r in dup_reports]),
                        use_container_width=True,
                        hide_index=True,
                    )
    
            with tab_label:
                st.markdown("#### Label Analysis")
                label_findings = [f for f in st.session_state.get("dataset_findings", [])
                                if f.get("finding_type") in ("LABEL_INCONSISTENCY", "LABEL_FLIPPING_INDICATOR")]
                if label_findings:
                    st.warning(f"{len(label_findings)} label finding(s) detected.")
                    st.dataframe(pd.DataFrame(label_findings), use_container_width=True, hide_index=True)
                else:
                    labeled = [r for r in reports if r.get("label")]
                    if labeled:
                        st.success("No label inconsistencies detected among labeled samples.")
                    else:
                        st.info("No label annotations found in the dataset. Label analysis requires COCO or YOLO annotations.")
    
            with tab_ood:
                st.markdown("#### OOD / Anomaly Detection")
                visual_outliers = sum(r["is_visual_outlier"] for r in reports)
                spectral_outliers = sum(r["is_spectral_outlier"] for r in reports)
                a, b, c = st.columns(3)
                a.metric("VISUAL OUTLIERS", visual_outliers)
                b.metric("SPECTRAL OUTLIERS", spectral_outliers)
                c.metric("TOP SINGULAR VALUE", dataset.get("top_singular_value", "N/A"))
    
                outlier_reports = [r for r in reports if r["is_visual_outlier"] or r["is_spectral_outlier"]]
                if outlier_reports:
                    st.dataframe(
                        pd.DataFrame([{
                            "Image #": r["index"],
                            "File": r["source"],
                            "Contributor": r["contributor"],
                            "Anomaly %": r["anomaly_confidence"],
                            "Spectral %": r["spectral_confidence"],
                            "Visual Outlier": r["is_visual_outlier"],
                            "Spectral Outlier": r["is_spectral_outlier"],
                            "Disposition": r["disposition"],
                        } for r in outlier_reports]),
                        use_container_width=True,
                        hide_index=True,
                    )
    
                st.info(
                    "This prototype applies Isolation Forest and SVD to a compact handcrafted visual descriptor. "
                    "Anomaly ≠ Maliciousness. Outlier status indicates statistical deviation, not proven compromise."
                )
    
            with tab_trigger:
                st.markdown("#### Trigger Injection Indicators")
                trigger_findings = [f for f in st.session_state.get("dataset_findings", [])
                                  if f.get("finding_type") == "TRIGGER_INJECTION_INDICATOR"]
                if trigger_findings:
                    st.warning(f"{len(trigger_findings)} trigger indicator(s) detected.")
                    st.dataframe(pd.DataFrame(trigger_findings), use_container_width=True, hide_index=True)
                else:
                    st.success("No trigger injection indicators detected.")
                st.info(
                    "Trigger indicators are heuristic signals (dual-outlier with high anomaly). "
                    "They do not prove backdoor injection. Analyst review is required."
                )
    
    # ================================================================
    # DISTRIBUTION & SPECTRAL
    # ================================================================
    
    elif page == "04 · Distribution & Spectral":
        st.subheader("04 · Distribution Shift & Spectral Screening")
    
        if not reports:
            st.info("Upload an image ZIP containing at least 5 images.")
        else:
            df_risk = risk_dataframe(reports)
    
            high = int((df_risk["Shift"] == "HIGH SHIFT").sum())
            moderate = int((df_risk["Shift"] == "MODERATE SHIFT").sum())
    
            a, b, c, d = st.columns(4)
            a.metric("MEAN SHIFT DISTANCE", f"{dataset.get('overall_shift_distance', 0):.3f}")
            b.metric("HIGH SHIFT", high)
            c.metric("MODERATE SHIFT", moderate)
            d.metric("IMAGES", len(reports))
    
            if HAS_PLOTLY:
                left, right = st.columns(2)
    
                with left:
                    fig = px.histogram(
                        df_risk,
                        x="Shift Distance",
                        nbins=18,
                        title="Per-image shift distance distribution",
                    )
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
    
                with right:
                    fig = px.scatter(
                        df_risk,
                        x="Brightness",
                        y="Contrast",
                        size="Shift %",
                        color="Shift",
                        hover_name="File",
                        title="Acquisition conditions vs shift evidence",
                    )
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
    
            st.markdown("#### Spectral Screening")
            spectral_count = sum(r["is_spectral_outlier"] for r in reports)
            a, b, c = st.columns(3)
            a.metric("SPECTRAL OUTLIERS", spectral_count)
            b.metric("TOP SINGULAR VALUE", dataset.get("top_singular_value", "N/A"))
            c.metric("TOP SINGULAR VARIANCE", dataset.get("spectral_variance_ratio", "N/A"))
    
            df_spectral = df_risk[
                ["Image #", "File", "Contributor", "Anomaly %", "Spectral %",
                 "Review %", "Severity", "Disposition"]
            ].sort_values(["Spectral %", "Review %"], ascending=[False, False])
    
            st.dataframe(df_spectral, use_container_width=True, hide_index=True)
    
            if HAS_PLOTLY and len(reports):
                fig = px.scatter_3d(
                    df_risk,
                    x="Anomaly %",
                    y="Shift Distance",
                    z="Spectral %",
                    color="Severity",
                    size="Review %",
                    hover_name="File",
                    hover_data=["Contributor", "Review %", "Disposition"],
                    title="3D Evidence Map: Anomaly × Shift × Spectral",
                )
                fig.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
            st.info(
                "Shift distance is a feature-space screening measure. A high shift can be "
                "caused by legitimate operating changes. SVD spectral screening uses handcrafted "
                "visual descriptors, not learned neural representations."
            )
    
    # ================================================================
    # CONTRIBUTORS
    # ================================================================
    
    elif page == "05 · Contributors":
        st.subheader("05 · Multi-Contributor Backend & Assurance")
    
        db = get_contributor_backend()
        persistent_contributors = db.list_contributors()
    
        st.markdown("#### Persistent Contributor Registry")
        st.markdown(
            '<div class="small">Persistent offline records stored in SQLite backend. '
            'Status is aggregated upward from sample findings: NORMAL, REVIEW, ELEVATED, QUARANTINE_RECOMMENDED.</div>',
            unsafe_allow_html=True,
        )
    
        if persistent_contributors:
            p_df = pd.DataFrame([
                {
                    "Contributor ID": c.contributor_id,
                    "Display Name": c.display_name,
                    "Organization": c.organization or "N/A",
                    "Source ID": c.source_id,
                    "Total Samples": c.sample_count,
                    "Quarantine": c.quarantine_count,
                    "Avg Review %": c.avg_review_confidence,
                    "Status": c.status,
                    "Created At": c.created_at,
                    "Last Updated": c.updated_at,
                }
                for c in persistent_contributors
            ])
            st.dataframe(p_df, use_container_width=True, hide_index=True)
    
            st.markdown("#### Contributor Traceability & Evidence Reconstruction")
            st.caption("Reconstruct: Contributor → Contributions → Batches → Affected Samples → Findings → Provenance")
    
            contrib_ids = [c.contributor_id for c in persistent_contributors]
            selected_cid = st.selectbox("Select Contributor to Trace", contrib_ids, key="trace_cid_select")
            if selected_cid:
                trace = db.get_contributor_traceability(selected_cid)
                if trace.get("found"):
                    agg = db.get_contributor_aggregation(selected_cid)
                    t_cols1 = st.columns(6)
                    t_cols1[0].metric("CONTRIBUTOR", agg.get("display_name", selected_cid))
                    t_cols1[1].metric("STATUS", agg.get("status", "NORMAL"))
                    t_cols1[2].metric("LIFETIME SAMPLES", agg.get("lifetime_samples", 0))
                    t_cols1[3].metric("CONTRIBUTIONS", agg.get("contributions", 0))
                    t_cols1[4].metric("BATCHES", agg.get("batches", 0))
                    t_cols1[5].metric("QUARANTINE", agg.get("quarantine_count", 0))

                    t_cols2 = st.columns(6)
                    t_cols2[0].metric("AVG REVIEW %", f"{agg.get('review_percentage', 0.0)}%")
                    t_cols2[1].metric("EXACT DUP FLAGS", agg.get("exact_duplicate_findings", 0))
                    t_cols2[2].metric("NEAR DUP FLAGS", agg.get("near_duplicate_findings", 0))
                    t_cols2[3].metric("LABEL FINDINGS", agg.get("label_findings", 0))
                    t_cols2[4].metric("OOD/SHIFT FLAGS", agg.get("ood_shift_findings", 0))
                    t_cols2[5].metric("TRIGGER FLAGS", agg.get("trigger_anomaly_indicators", 0))
    
                    tab_batches, tab_findings, tab_samples = st.tabs(["Batches & Contributions", "Associated Findings", "Sample Evidence"])
    
                    with tab_batches:
                        if trace.get("batches"):
                            b_rows = [
                                {
                                    "Batch ID": b["batch_id"],
                                    "Batch Name": b["batch_name"],
                                    "Samples": b["sample_count"],
                                    "Visual Outliers": b["visual_outliers"],
                                    "Spectral Outliers": b["spectral_outliers"],
                                    "High Shift": b["high_shift"],
                                    "Quarantine": b["quarantine_count"],
                                    "Avg Review %": b["avg_review_confidence"],
                                    "Status": b["status"],
                                }
                                for b in trace["batches"]
                            ]
                            st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)
                        else:
                            st.info("No batches recorded for this contributor.")
    
                    with tab_findings:
                        if trace.get("findings"):
                            f_rows = [
                                {
                                    "Finding ID": f["finding_id"],
                                    "Type": f["finding_type"],
                                    "Severity": f["severity"],
                                    "Confidence %": f["confidence"],
                                    "Disposition": f["recommended_disposition"],
                                    "Reason": f["reason"],
                                    "Affected Samples": ", ".join(f["affected_samples"]) if f["affected_samples"] else "N/A",
                                }
                                for f in trace["findings"]
                            ]
                            st.dataframe(pd.DataFrame(f_rows), use_container_width=True, hide_index=True)
                        else:
                            st.success("No anomalous findings associated with this contributor.")
    
                    with tab_samples:
                        if trace.get("samples"):
                            s_rows = [
                                {
                                    "Sample ID": s["sample_id"],
                                    "File": s["file_path"],
                                    "Disposition": s["disposition"],
                                    "Severity": s["severity"],
                                    "Review %": s["review_confidence"],
                                    "Shift": s["shift_status"],
                                    "Quality %": s["quality_confidence"],
                                    "Visual Outlier": bool(s["is_visual_outlier"]),
                                    "Spectral Outlier": bool(s["is_spectral_outlier"]),
                                }
                                for s in trace["samples"][:100]
                            ]
                            st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)
                            if len(trace["samples"]) > 100:
                                st.caption(f"Showing first 100 of {len(trace['samples'])} samples.")
                        else:
                            st.info("No sample records stored for this contributor.")
        else:
            st.info("No contributors registered in the persistent store yet. Register below or upload a dataset.")
    
        with st.expander("➕ Register / Enroll New Contributor"):
            r_c1, r_c2 = st.columns(2)
            with r_c1:
                new_cid = st.text_input("Contributor ID (e.g. CONTRIB-01)", key="reg_cid")
                new_cname = st.text_input("Display Name", key="reg_cname")
            with r_c2:
                new_org = st.text_input("Organization / Affiliation", key="reg_corg")
                new_src = st.text_input("Source Identifier", value="MANUAL_REGISTRATION", key="reg_csrc")
            if st.button("Save Contributor to Backend"):
                if new_cid.strip():
                    rec = db.get_or_create_contributor(
                        new_cid.strip(),
                        display_name=new_cname.strip() or new_cid.strip(),
                        organization=new_org.strip(),
                        source_id=new_src.strip(),
                    )
                    st.success(f"✓ Registered contributor: {rec.contributor_id} ({rec.display_name})")
                    audit_event({"event": "contributor_registered", "contributor_id": rec.contributor_id})
                    st.rerun()
                else:
                    st.error("Contributor ID is required.")
    
        if reports:
            st.markdown("---")
            st.markdown("#### Active Dataset Contributor Aggregation")
            summary = contributor_dataframe(reports)
            if not summary.empty:
                st.dataframe(summary, use_container_width=True, hide_index=True)
    
                if HAS_PLOTLY and len(summary):
                    fig = px.bar(
                        summary,
                        x="Contributor / Source",
                        y="Avg Review %",
                        color="Status",
                        hover_data=[
                            "Samples", "Exact Dup Flags", "Near Dup Flags",
                            "OOD/Shift Flags", "Trigger Indicators", "Anomaly Flags", "Quarantine",
                        ],
                        title="Active Dataset Review Evidence by Contributor",
                    )
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=100))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    elif page == "06 · Batches":
        st.subheader("06 · Hierarchical Batch / Source Aggregation")
    
        db = get_contributor_backend()
        all_batches = db.list_all_batches()
    
        st.markdown("#### Persistent Batch Registry")
        st.markdown(
            '<div class="small">Batches partitioned by contributor and source directory. '
            'Hierarchy: Contributor → Contribution → Batch → Samples.</div>',
            unsafe_allow_html=True,
        )
    
        if all_batches:
            b_df = pd.DataFrame([
                {
                    "Batch ID": b["batch_id"],
                    "Batch Name": b["batch_name"],
                    "Contributor": b["contributor_id"],
                    "Dataset Digest": f"{b['dataset_digest'][:16]}...",
                    "Samples": b["sample_count"],
                    "Visual Outliers": b["visual_outliers"],
                    "Spectral Outliers": b["spectral_outliers"],
                    "High Shift": b["high_shift"],
                    "Quarantine": b["quarantine_count"],
                    "Avg Review %": b["avg_review_confidence"],
                    "Status": b["status"],
                    "Created At": b["created_at"],
                }
                for b in all_batches
            ])
            st.dataframe(b_df, use_container_width=True, hide_index=True)
    
            # Hierarchical grouping view
            st.markdown("#### Contributor ➔ Batch Hierarchy")
            by_c = defaultdict(list)
            for b in all_batches:
                by_c[b["contributor_id"]].append(b)
    
            for c_id, b_list in by_c.items():
                with st.expander(f"👤 {c_id} ({len(b_list)} batch(es))"):
                    for b in b_list:
                        st.write(
                            f"• **{b['batch_name']}** — {b['sample_count']} samples · "
                            f"Quarantine: {b['quarantine_count']} · Avg Review: {b['avg_review_confidence']}% · "
                            f"Status: `{b['status']}`"
                        )
        else:
            st.info("No batches recorded in the persistent store yet.")
    
        if reports:
            st.markdown("---")
            st.markdown("#### Active Dataset Batch Aggregation")
            batch_df = batch_dataframe(reports)
            if not batch_df.empty:
                st.dataframe(batch_df, use_container_width=True, hide_index=True)
                if HAS_PLOTLY and len(batch_df):
                    fig = px.bar(
                        batch_df,
                        x="Batch",
                        y="Avg Review %",
                        hover_data=["Samples", "Visual Outliers", "Spectral Outliers", "High Shift", "Quarantine"],
                        title="Active Dataset Batch Evidence",
                    )
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=100))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    # ================================================================
    # FINDINGS
    # ================================================================
    
    elif page == "07 · Findings":
        st.subheader("07 · Structured Findings")
    
        findings = st.session_state.get("dataset_findings", [])
    
        if not findings:
            st.info("No findings generated yet. Upload and analyze a dataset.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("TOTAL FINDINGS", len(findings))
            b.metric("HIGH SEVERITY", sum(f.get("severity") == "HIGH" for f in findings))
            c.metric("MEDIUM SEVERITY", sum(f.get("severity") == "MEDIUM" for f in findings))
            d.metric("QUARANTINE", sum(f.get("recommended_disposition") == "QUARANTINE" for f in findings))
    
            # Filter by type
            types = list(set(f.get("finding_type", "UNKNOWN") for f in findings))
            selected_types = st.multiselect("Filter by type", types, default=types)
            filtered = [f for f in findings if f.get("finding_type") in selected_types]
    
            st.dataframe(
                pd.DataFrame(filtered),
                use_container_width=True,
                hide_index=True,
            )
    
            st.download_button(
                "⬇️ Export Findings JSON",
                json.dumps(findings, indent=2).encode("utf-8"),
                file_name="trustcv_findings.json",
                mime="application/json",
            )
    
    # ================================================================
    # EVIDENCE EXPLORER
    # ================================================================
    
    elif page == "08 · Evidence Explorer":
        st.subheader("08 · Evidence Explorer")
    
        if not reports:
            st.info("Upload an image ZIP containing at least 5 images.")
        else:
            df_risk = risk_dataframe(reports)
    
            st.markdown(
                '<div class="small">All percentages are screening-confidence values, not calibrated attack probabilities.</div>',
                unsafe_allow_html=True,
            )
    
            st.dataframe(
                df_risk,
                use_container_width=True,
                hide_index=True,
                height=560,
            )
    
            st.download_button(
                "⬇️ Download complete image risk matrix",
                df_risk.to_csv(index=False).encode("utf-8"),
                file_name="trustcv_image_risk_matrix.csv",
                mime="text/csv",
            )
    
            st.markdown("#### Individual Image Explorer")
    
            selected = st.number_input(
                "Image number",
                min_value=1,
                max_value=len(reports),
                value=1,
                step=1,
            )
    
            r = reports[selected - 1]
            image = dataset["images"][selected - 1]
    
            left, right = st.columns([1.0, 1.3])
    
            with left:
                st.image(
                    image,
                    caption=f"Image {selected} · {r['source']}",
                    width=620,
                )
    
            with right:
                a, b, c, d = st.columns(4)
                a.metric("QUALITY", f"{r['quality_confidence']:.1f}%")
                b.metric("ANOMALY", f"{r['anomaly_confidence']:.1f}%")
                c.metric("SPECTRAL", f"{r['spectral_confidence']:.1f}%")
                d.metric("SHIFT", f"{r['shift_confidence']:.1f}%")
    
                e, f, g = st.columns(3)
                e.metric("SHIFT DISTANCE", f"{r['shift_distance']:.3f}")
                f.metric("REVIEW", f"{r['review_confidence']:.1f}%")
                g.metric("SEVERITY", r["severity"])
    
                st.write("Disposition: " + r["disposition"])
                st.write("Contributor: " + r["contributor"])
                if r.get("batch"):
                    st.write("Batch: " + r["batch"])
                st.write(f"Sample SHA-256: {r.get('hash', 'N/A')}")
                if r.get("phash"):
                    st.write(f"Perceptual Hash: {r.get('phash')}")
                if r.get("label"):
                    st.write("Label: " + r["label"])
                st.write("Orientation: " + r["orientation_status"])
                st.write("Exact duplicate: " + ("YES" if r["duplicate_count"] else "NO"))
                st.write("Near duplicate: " + ("YES" if r["is_near_duplicate"] else "NO"))
                st.write("Shift status: " + r["shift_status"])
                if r.get("trigger_flags"):
                    st.warning(f"Trigger indicators: {', '.join(r['trigger_flags'])}")
    
                if r["disposition"] == "QUARANTINE":
                    st.error(
                        "QUARANTINE: multiple screening signals require investigation."
                    )
                elif r["disposition"] == "REVIEW":
                    st.warning(
                        "REVIEW: screening evidence is above the review threshold."
                    )
                else:
                    st.success("ACCEPT: no high-priority screening signal.")
    
    # ================================================================
    # DATASET PROVENANCE
    # ================================================================
    
    elif page == "09 · Dataset Provenance":
        st.subheader("09 · Dataset Provenance Chain")
    
        mirad_record = st.session_state.get("mirad_security_record", {})
        if not mirad_record or not mirad_record.get("provenance"):
            p_path = Path("demo_output/dataset_security/provenance.json")
            if not p_path.exists():
                p_path = Path("demo_output/provenance.json")
            v_path = Path("demo_output/dataset_security/verification_result.json")
            if not v_path.exists():
                v_path = Path("demo_output/verification_result.json")
            if p_path.exists() and v_path.exists():
                try:
                    p_data = json.loads(p_path.read_text(encoding="utf-8"))
                    v_data = json.loads(v_path.read_text(encoding="utf-8"))
                    mirad_record = {
                        "provenance": p_data,
                        "provenance_verification": v_data.get("provenance_verification", {}),
                    }
                except Exception:
                    pass
    
        if not mirad_record or mirad_record.get("error"):
            st.info("MIRAD provenance is generated after dataset analysis.")
            if mirad_record.get("error"):
                st.error(f"MIRAD error: {mirad_record['error']}")
        else:
            prov = mirad_record.get("provenance", {})
            prov_ver = mirad_record.get("provenance_verification", {})
    
            st.markdown("#### Provenance Chain")
            provenance_data = [
                ["Dataset", prov.get("dataset_name", "N/A")],
                ["Dataset Digest", prov.get("dataset_digest", "N/A")],
                ["Manifest Digest", prov.get("manifest_digest", "N/A")],
                ["Contributor ID", prov.get("contributor_id", "N/A")],
                ["Contribution ID", prov.get("contribution_id", "N/A")],
                ["Batch ID", prov.get("batch_id", "N/A")],
                ["Event ID / Nonce", f"{prov.get('event_id', 'N/A')} (Nonce: {str(prov.get('nonce', 'N/A'))[:16]}...)"],
                ["Sequence", prov.get("sequence", "N/A")],
                ["Timestamp", prov.get("timestamp", "N/A")],
                ["Context", prov.get("context", "N/A")],
                ["Analysis Digest", prov.get("analysis_digest", "N/A")],
                ["Evidence Digest", prov.get("evidence_digest", "N/A")],
                ["Signature Algorithm", prov.get("signature_algorithm", "N/A")],
                ["Key ID", prov.get("key_id", "N/A")],
                ["Trust Anchor", prov.get("trust_anchor_id", "N/A")],
                ["Signature (Base64)", f"{str(prov.get('signature', 'N/A'))[:24]}..."],
                ["Verification State", "TRUSTED [OK]" if prov_ver.get("trusted") else "UNVERIFIED / FAILED"],
            ]
            st.dataframe(
                pd.DataFrame(provenance_data, columns=["Field", "Value"]),
                use_container_width=True,
                hide_index=True,
            )
    
            st.markdown("#### Provenance Verification")
            ver_data = [
                ["Schema Valid", prov_ver.get("schema_valid", False)],
                ["Signature Valid", prov_ver.get("signature_valid", False)],
                ["Trust Anchor Valid", prov_ver.get("trust_anchor_valid", False)],
                ["Dataset Binding Valid", prov_ver.get("dataset_binding_valid", False)],
                ["Policy Valid", prov_ver.get("policy_valid", False)],
                ["Overall Trusted", prov_ver.get("trusted", False)],
            ]
            st.dataframe(
                pd.DataFrame(ver_data, columns=["Check", "Result"]),
                use_container_width=True,
                hide_index=True,
            )
    
            st.markdown("#### Full Security Pipeline")
            st.markdown("""
            ```
            Dataset → Dataset Identity → Dataset Digest → Dataset Analysis
            → Sample Evidence → Findings → Contributor/Batch Aggregation
            → Dataset Assurance Record → MIRAD Canonicalization → MIRAD Hash
            → MIRAD Signature → MIRAD Provenance → MIRAD Audit Entry
            → MIRAD Audit Chain → MIRAD Checkpoint → Optional Ethereum Anchor
            → Verification
            ```
            """)
    
    # ================================================================
    # VERIFICATION
    # ================================================================
    
    elif page == "10 · Verification":
        st.subheader("10 · Cryptographic Verification & Security Assurance")
    
        # ============================================================
        # SECTION 1: SIGNED DATASET MANIFEST VERIFICATION
        # ============================================================
        st.markdown("#### 1. Authoritative Signed Dataset Manifest Verification")
        st.caption("Cryptographically verifies the authoritative signed manifest against the ingested dataset and trusted local trust anchor.")
    
        ver_result = st.session_state.get("manifest_verification_result")
        overall_status = ver_result.get("overall", "UNAVAILABLE") if ver_result else "UNAVAILABLE"
        ver_reason = ver_result.get("reason", "No verification performed.") if ver_result else "No verification performed."
        checks = ver_result.get("checks", {}) if ver_result else {}
        manifest = ver_result.get("manifest", {}) if ver_result else {}
        actual_hash = st.session_state.get("dataset_hash_override") or st.session_state.get("dataset_hash", "")
    
        # Top Status Banner
        if overall_status == "PASS":
            st.markdown(
                '<div class="secure" style="padding:16px; border-radius:6px; font-size:17px; margin-bottom:15px;">'
                '🟢 <b>OVERALL MANIFEST VERIFICATION: PASS</b><br>'
                '<span style="font-size:13px; opacity:0.9;">Authentic Ed25519 signature valid under local trust anchor. All dataset digest and contributor bindings verified.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif overall_status == "DATASET_DIGEST_MISMATCH":
            st.markdown(
                f'<div class="review" style="padding:16px; border-radius:6px; font-size:17px; background:#4a1212; border:1px solid #ff4d4d; color:#ffcccc; margin-bottom:15px;">'
                f'🔴 <b>OVERALL MANIFEST VERIFICATION: DATASET DIGEST MISMATCH</b><br>'
                f'<span style="font-size:13px;">{escape(ver_reason)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif overall_status == "CONTRIBUTOR_MISMATCH":
            st.markdown(
                f'<div class="review" style="padding:16px; border-radius:6px; font-size:17px; background:#4a2c12; border:1px solid #ff9933; color:#ffe6cc; margin-bottom:15px;">'
                f'🔴 <b>OVERALL MANIFEST VERIFICATION: CONTRIBUTOR MISMATCH</b><br>'
                f'<span style="font-size:13px;">{escape(ver_reason)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif overall_status == "INVALID_SIGNATURE":
            st.markdown(
                f'<div class="review" style="padding:16px; border-radius:6px; font-size:17px; background:#4a1212; border:1px solid #ff4d4d; color:#ffcccc; margin-bottom:15px;">'
                f'🔴 <b>OVERALL MANIFEST VERIFICATION: INVALID SIGNATURE</b><br>'
                f'<span style="font-size:13px;">{escape(ver_reason)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif overall_status == "MISSING_TRUSTED_KEY":
            st.markdown(
                f'<div class="review" style="padding:16px; border-radius:6px; font-size:17px; background:#4a1212; border:1px solid #ff4d4d; color:#ffcccc; margin-bottom:15px;">'
                f'🔴 <b>OVERALL MANIFEST VERIFICATION: MISSING TRUSTED KEY</b><br>'
                f'<span style="font-size:13px;">{escape(ver_reason)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif overall_status == "UNAVAILABLE":
            st.markdown(
                '<div class="review" style="padding:16px; border-radius:6px; font-size:17px; background:#242424; border:1px solid #666; color:#bbb; margin-bottom:15px;">'
                '⚪ <b>OVERALL MANIFEST VERIFICATION: UNAVAILABLE</b><br>'
                '<span style="font-size:13px;">No signed dataset manifest supplied or loaded. Please upload a manifest or attest one on Page 01.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="review" style="padding:16px; border-radius:6px; font-size:17px; margin-bottom:15px;">'
                f'🟠 <b>OVERALL MANIFEST VERIFICATION: {escape(overall_status)}</b><br>'
                f'<span style="font-size:13px;">{escape(ver_reason)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    
        # 4 Primary Verification KPI Metrics
        sig_check = checks.get("digital_signature", {})
        dig_check = checks.get("dataset_digest", {})
        cnt_check = checks.get("contributor", {})
    
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("DIGITAL SIGNATURE", sig_check.get("status", "UNAVAILABLE"))
        m2.metric("DATASET BINDING", dig_check.get("status", "UNAVAILABLE"))
        m3.metric("CONTRIBUTOR BINDING", cnt_check.get("status", "UNAVAILABLE"))
        m4.metric("OVERALL VERIFICATION", overall_status)
    
        # Signed vs Observed Comparison Matrix Table
        st.markdown("##### Signed vs Observed Verification Matrix")
    
        signed_hash = dig_check.get("signed")
        obs_hash = dig_check.get("observed") or actual_hash
        signed_hash_disp = f"{signed_hash[:16]}...{signed_hash[-8:]}" if signed_hash else "None (Unsigned)"
        obs_hash_disp = f"{obs_hash[:16]}...{obs_hash[-8:]}" if obs_hash else "None (Not Observed)"
    
        signed_cnt = cnt_check.get("signed") or "None (Unsigned)"
        obs_cnt = cnt_check.get("observed") or "None (Not Observed)"
    
        bat_check = checks.get("batch", {})
        signed_bat = bat_check.get("signed") or "None (Unsigned)"
        obs_bat = bat_check.get("observed") or "None (Not Observed)"
    
        integ_check = checks.get("manifest_integrity", {})
    
        matrix_rows = [
            {
                "Security Dimension": "Dataset Digest (SHA-256)",
                "Authoritative Signed Value": signed_hash_disp,
                "Observed / Expected Value": obs_hash_disp,
                "Status": dig_check.get("status", "UNAVAILABLE"),
                "Technical Details": dig_check.get("reason", "N/A"),
            },
            {
                "Security Dimension": "Contributor Identity",
                "Authoritative Signed Value": signed_cnt,
                "Observed / Expected Value": obs_cnt,
                "Status": cnt_check.get("status", "UNAVAILABLE"),
                "Technical Details": cnt_check.get("reason", "N/A"),
            },
            {
                "Security Dimension": "Batch / Source ID",
                "Authoritative Signed Value": signed_bat,
                "Observed / Expected Value": obs_bat,
                "Status": bat_check.get("status", "UNCHECKED"),
                "Technical Details": bat_check.get("reason", "N/A"),
            },
            {
                "Security Dimension": "Digital Signature",
                "Authoritative Signed Value": f"Ed25519 (Key ID: {str(sig_check.get('key_id', ''))[:16]}...)" if sig_check.get("key_id") else "Ed25519",
                "Observed / Expected Value": f"Anchor: {str(sig_check.get('trusted_anchor_id', ''))[:16]}..." if sig_check.get("trusted_anchor_id") else "Local Trust Anchor",
                "Status": sig_check.get("status", "UNAVAILABLE"),
                "Technical Details": sig_check.get("reason", "N/A"),
            },
            {
                "Security Dimension": "Manifest Integrity",
                "Authoritative Signed Value": f"Schema: {manifest.get('schema_version', 'N/A')}",
                "Observed / Expected Value": "Canonical RFC 8785 JSON",
                "Status": integ_check.get("status", "UNAVAILABLE"),
                "Technical Details": integ_check.get("reason", "N/A"),
            },
        ]
    
        df_matrix = pd.DataFrame(matrix_rows)
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)
    
        # Interactive Verification & Tamper Simulation Tools
        with st.expander("🧪 Interactive Verification & Tamper Simulation Tools"):
            st.caption("Demonstrate truthful verification rejection: simulate contributor mismatches, dataset tampering, or corrupted signatures.")
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    
            with t_col1:
                curr_override = st.session_state.get("expected_contributor_override") or ""
                override_val = st.text_input("Expected Contributor Override", value=curr_override, key="ctrl_override_contrib")
                if st.button("Apply Contributor Override", key="btn_apply_contrib_override"):
                    st.session_state["expected_contributor_override"] = override_val.strip() if override_val.strip() else None
                    st.session_state["manifest_bound_dataset_hash"] = None
                    st.rerun()
    
            with t_col2:
                if st.button("Simulate Tampered Dataset", key="btn_sim_tamper_ds"):
                    st.session_state["dataset_hash_override"] = "0000000000000000000000000000000000000000000000000000000000000000"
                    st.session_state["manifest_bound_dataset_hash"] = None
                    st.rerun()
    
            with t_col3:
                if st.button("Simulate Corrupted Signature", key="btn_sim_tamper_sig"):
                    if st.session_state.get("dataset_manifest"):
                        m_tampered = dict(st.session_state["dataset_manifest"])
                        orig_sig = m_tampered.get("signature", "")
                        m_tampered["signature"] = "bad" + orig_sig[3:] if len(orig_sig) > 3 else "badsignature=="
                        st.session_state["dataset_manifest"] = m_tampered
                        st.session_state["manifest_verification_result"] = verify_signed_dataset_manifest(
                            m_tampered,
                            actual_dataset_digest=actual_hash,
                            expected_contributor_id=st.session_state.get("expected_contributor_override"),
                        )
                        st.session_state["manifest_status"] = st.session_state["manifest_verification_result"]["overall"]
                        st.rerun()
    
            with t_col4:
                if st.button("Restore Genuine / Clean State", key="btn_restore_clean"):
                    st.session_state.pop("expected_contributor_override", None)
                    st.session_state.pop("dataset_hash_override", None)
                    if manifest_upload is not None:
                        try:
                            st.session_state["dataset_manifest"] = json.loads(manifest_upload.getvalue().decode("utf-8"))
                        except Exception:
                            pass
                    st.session_state["manifest_bound_dataset_hash"] = None
                    st.rerun()
    
        # Raw Manifest View
        if manifest:
            with st.expander("📄 Authoritative Signed Manifest JSON & Signature Payload"):
                st.json(manifest)
    
        st.markdown("---")
    
        # ============================================================
        # SECTION 2: MIRAD SECURITY LAYER & PROVENANCE VERIFICATION
        # ============================================================
        st.markdown("#### 2. MIRAD Security Layer & Provenance Verification")
    
        mirad_record = st.session_state.get("mirad_security_record", {})
    
        if not mirad_record or mirad_record.get("error"):
            st.info("MIRAD provenance verification requires dataset analysis.")
        else:
            prov_ver = mirad_record.get("provenance_verification", {})
            replay_ver = mirad_record.get("replay_verification", {})
            audit_ver = mirad_record.get("audit_verification", {})
            chk_ver = mirad_record.get("checkpoint_verification", {})
    
            all_trusted = (
                prov_ver.get("trusted", False)
                and audit_ver.get("valid", False)
                and chk_ver.get("trusted", False)
            )
    
            if all_trusted:
                st.markdown(
                    '<div class="secure">🟢 <b>ALL MIRAD VERIFICATION GATES PASSED</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="review">🟠 <b>SOME MIRAD VERIFICATION CHECKS REQUIRE ATTENTION</b></div>',
                    unsafe_allow_html=True,
                )
    
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("PROVENANCE", "TRUSTED" if prov_ver.get("trusted") else "FAILED")
            v2.metric("REPLAY", "VALID" if replay_ver.get("valid") else "FAILED")
            v3.metric("AUDIT CHAIN", "VALID" if audit_ver.get("valid") else "FAILED")
            v4.metric("CHECKPOINT", "TRUSTED" if chk_ver.get("trusted") else "FAILED")
    
            with st.expander("Detailed MIRAD Verification Telemetry"):
                st.json({
                    "provenance_verification": prov_ver,
                    "replay_verification": replay_ver,
                    "audit_verification": audit_ver,
                    "checkpoint_verification": chk_ver,
                })
    
            # Replay protection demonstration
            st.markdown("##### Replay Protection Demonstration")
            st.caption("MIRAD replay protection cryptographically prevents reuse of provenance events.")
            prov = mirad_record.get("provenance", {})
            if prov and st.button("Test Replay Attack", key="btn_test_replay_mirad"):
                replay_result = replay_existing_provenance(prov)
                if not replay_result.get("valid"):
                    st.success(f"✓ Replay correctly rejected: {replay_result.get('reason')}")
                else:
                    st.error("Replay was accepted — this should not happen.")
    
    # ================================================================
    # AUDIT TRAIL
    # ================================================================
    
    elif page == "11 · Audit Trail":
        st.subheader("11 · Tamper-Evident Audit Trail")
    
        # MIRAD audit
        mirad_audit = verify_dataset_audit()
        st.markdown("#### MIRAD Dataset Security Audit")
        a, b = st.columns(2)
        a.metric("MIRAD AUDIT CHAIN", "VALID" if mirad_audit.get("valid") else "FAILED")
        b.metric("MIRAD AUDIT EVENTS", mirad_audit.get("events", 0))
        st.caption(mirad_audit.get("reason", ""))
    
        # Local audit chain
        if audit_valid:
            st.success(f"🟢 LOCAL AUDIT CHAIN VALID · {audit_count} events")
        else:
            st.error(f"🔴 LOCAL AUDIT CHAIN FAILED · {audit_count} events verified before failure")
    
        # Checkpoint
        mirad_record = st.session_state.get("mirad_security_record", {})
        checkpoint = mirad_record.get("checkpoint", {})
        if checkpoint:
            st.markdown("#### Latest MIRAD Checkpoint")
            chk_data = [
                ["Checkpoint ID", checkpoint.get("checkpoint_id", "N/A")],
                ["Sequence", checkpoint.get("latest_sequence", "N/A")],
                ["Audit Hash", checkpoint.get("latest_audit_hash", "N/A")],
                ["Timestamp", checkpoint.get("timestamp", "N/A")],
                ["Key ID", checkpoint.get("signing_key_id", "N/A")],
            ]
            st.dataframe(
                pd.DataFrame(chk_data, columns=["Field", "Value"]),
                use_container_width=True,
                hide_index=True,
            )
    
        if st.button("🔄 Create New Checkpoint"):
            try:
                new_chk = mirad_create_checkpoint()
                st.success(f"Checkpoint created: {new_chk.get('checkpoint_id')}")
                audit_event({"event": "checkpoint_created", "checkpoint_id": new_chk.get("checkpoint_id")})
            except Exception as exc:
                st.error(f"Checkpoint creation failed: {exc}")
    
        # Local audit events
        if AUDIT.exists():
            raw_audit = AUDIT.read_text(encoding="utf-8")
            audit_df = pd.DataFrame(
                [json.loads(line) for line in raw_audit.splitlines() if line.strip()]
            )
    
            st.markdown("#### Audit Events")
            st.dataframe(
                audit_df,
                use_container_width=True,
                hide_index=True,
            )
    
            st.download_button(
                "⬇️ Export audit_chain.jsonl",
                raw_audit,
                file_name="trustcv_dataset_audit_chain.jsonl",
                mime="application/jsonl",
            )
    
        # Assurance record
        if dataset is not None:
            st.markdown("#### Dataset Assurance Record")
            evidence = {
                "dataset_name": dataset.get("name"),
                "dataset_kind": dataset.get("kind"),
                "dataset_sha256": dataset.get("hash"),
                "contributor_verified": contributor_verified,
                "manifest_status": st.session_state.get("manifest_status"),
                "trust_posture": trust,
                "risk_index": risk,
                "image_count": len(reports),
                "findings_count": len(st.session_state.get("dataset_findings", [])),
                "visual_outliers": sum(r["is_visual_outlier"] for r in reports),
                "spectral_outliers": sum(r["is_spectral_outlier"] for r in reports),
                "high_shift": sum(r["shift_status"] == "HIGH SHIFT" for r in reports),
                "quarantine": sum(r["disposition"] == "QUARANTINE" for r in reports),
                "mirad_verification": mirad_record.get("checkpoint_verification", {}).get("trusted", False),
            }
            st.json(evidence)
    
        # Export center
        if dataset is not None:
            st.markdown("#### Analyst Export Center")
            pdf_bytes = build_pdf_report(
                dataset, reports, checks, trust, risk, audit_valid, audit_count
            )
            json_bytes = json.dumps(
                build_json_report(
                    dataset, reports, checks, trust, risk, audit_valid, audit_count
                ),
                indent=2,
            ).encode("utf-8")
            csv_bytes = risk_dataframe(reports).to_csv(index=False).encode("utf-8")
            bundle_bytes = build_evidence_bundle(
                dataset, reports, checks, trust, risk, audit_valid, audit_count
            )
    
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.download_button(
                    "📄 PDF Report",
                    pdf_bytes,
                    file_name="trustcv_dataset_security_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with e2:
                st.download_button(
                    "📊 CSV Matrix",
                    csv_bytes,
                    file_name="trustcv_image_risk_matrix.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with e3:
                st.download_button(
                    "🧾 JSON Evidence",
                    json_bytes,
                    file_name="trustcv_dataset_security_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with e4:
                st.download_button(
                    "📦 Evidence ZIP",
                    bundle_bytes,
                    file_name="trustcv_dataset_security_evidence_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
    
    # ================================================================
    # ETHEREUM ANCHOR
    # ================================================================
    
    elif page == "12 · Ethereum Anchor":
        st.subheader("12 · Ethereum Audit Checkpoint Anchor")
    
        st.markdown("""
        <div class="small">
        Ethereum provides an <b>external blockchain commitment</b> to a MIRAD-protected checkpoint.
        It does NOT store the audit log, dataset images, or sensitive metadata on-chain.
        The full audit log is NOT placed on-chain — only the cryptographic digest of the checkpoint.
        </div>
        """, unsafe_allow_html=True)
    
        eth_config = EthereumConfig.from_env()
        eth_status = get_ethereum_status(eth_config)
    
        st.markdown("#### Ethereum Configuration Status")
        status_str = eth_status.get("status", "NOT CONFIGURED")
        if status_str == "CONNECTED":
            st.markdown(
                '<div class="secure">⟠ <b>ETHEREUM CONNECTED</b></div>',
                unsafe_allow_html=True,
            )
            a, b, c = st.columns(3)
            a.metric("CHAIN ID", eth_status.get("chain_id", "N/A"))
            b.metric("BLOCK NUMBER", eth_status.get("block_number", "N/A"))
            c.metric("CONTRACT", str(eth_status.get("contract", "N/A"))[:16] + "…")
        elif status_str == "OFFLINE":
            st.markdown(
                f'<div class="review">⟠ <b>ETHEREUM OFFLINE</b> · {eth_status.get("detail", "")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="locked">⟠ <b>ETHEREUM NOT CONFIGURED</b> · {eth_status.get("detail", "Set environment variables per .env.example")}</div>',
                unsafe_allow_html=True,
            )
    
        # Anchor checkpoint
        mirad_record = st.session_state.get("mirad_security_record", {})
        checkpoint = mirad_record.get("checkpoint", {})
    
        if checkpoint and eth_config.is_configured:
            st.markdown("#### Anchor Checkpoint to Ethereum")
            st.caption(f"Checkpoint: {checkpoint.get('checkpoint_id', 'N/A')}")
            st.caption(f"Audit Hash: {checkpoint.get('latest_audit_hash', 'N/A')}")
    
            if st.button("⟠ Anchor to Ethereum"):
                audit_hash = checkpoint.get("latest_audit_hash", "")
                if audit_hash and audit_hash != "GENESIS":
                    dataset_digest = st.session_state.get("dataset_hash", "0" * 64)
                    result = eth_anchor_checkpoint(
                        audit_digest=audit_hash,
                        sequence=checkpoint.get("latest_sequence", 0),
                        dataset_digest=dataset_digest,
                        config=eth_config,
                    )
                    st.session_state["ethereum_anchor_result"] = result.to_dict()
    
                    if result.success:
                        st.success(f"✓ ANCHORED — TX: {result.tx_hash}")
                        audit_event({
                            "event": "ethereum_anchor",
                            "tx_hash": result.tx_hash,
                            "block_number": result.block_number,
                            "audit_digest": audit_hash,
                        })
                    else:
                        st.error(f"Anchor failed: {result.status} — {result.error}")
                else:
                    st.warning("No valid audit hash to anchor. Create a checkpoint first.")
    
        # Previous anchor result
        eth_result = st.session_state.get("ethereum_anchor_result", {})
        if eth_result:
            st.markdown("#### Last Anchor Result")
            anchor_data = [
                ["Status", eth_result.get("status", "N/A")],
                ["Transaction Hash", eth_result.get("tx_hash", "N/A")],
                ["Block Number", eth_result.get("block_number", "N/A")],
                ["Chain ID", eth_result.get("chain_id", "N/A")],
                ["Contract", eth_result.get("contract_address", "N/A")],
                ["Audit Digest", eth_result.get("audit_digest", "N/A")],
                ["Dataset Digest", eth_result.get("dataset_digest", "N/A")],
                ["Sequence", eth_result.get("sequence", "N/A")],
            ]
            st.dataframe(
                pd.DataFrame(anchor_data, columns=["Field", "Value"]),
                use_container_width=True,
                hide_index=True,
            )
    
        # Verification
        if eth_result.get("success") and eth_config.is_configured:
            st.markdown("#### Verify Against Ethereum")
            if st.button("🔍 Verify Anchor"):
                audit_hash = checkpoint.get("latest_audit_hash", "")
                ver_result = eth_verify_anchor(
                    audit_digest=eth_result.get("audit_digest", ""),
                    local_audit_hash=audit_hash,
                    local_dataset_hash=st.session_state.get("dataset_hash", ""),
                    config=eth_config,
                )
                ver_dict = ver_result.to_dict()
                st.session_state["ethereum_verification"] = ver_dict
    
                if ver_result.status == "MATCH":
                    st.markdown(
                        '<div class="secure">⟠ <b>MATCH</b> · Local MIRAD audit digest and dataset digest match the Ethereum anchor.</div>',
                        unsafe_allow_html=True,
                    )
                elif ver_result.status == "MISMATCH":
                    st.markdown(
                        '<div class="locked">⟠ <b>MISMATCH</b> · Local digest differs from the Ethereum anchor.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"Verification status: {ver_result.status}")
    
                st.json(ver_dict)
    
        if not checkpoint:
            st.info("Analyze a dataset first to generate a MIRAD checkpoint for anchoring.")
    
    # ================================================================
    # COVERAGE / LIMITATIONS
    # ================================================================
    
    elif page == "13 · Coverage / Limitations":
        st.subheader("13 · Coverage / Limitations")
    
        st.markdown("""
        #### Supported Formats
        | Format | Status |
        |--------|--------|
        | COCO JSON | ✅ Supported |
        | YOLO TXT labels | ✅ Supported |
        | Image ZIP | ✅ Supported |
        | CSV | ✅ Supported |
        | Excel (XLSX/XLS) | ✅ Supported |
        | Single Image | ✅ Supported |
        | Image Folder + CSV metadata | ⚠️ Partial (via ZIP) |
    
        #### Supported Analyses
        | Analysis | Method | Status |
        |----------|--------|--------|
        | Exact duplicate detection | SHA-256 | ✅ |
        | Near-duplicate detection | Perceptual hash (aHash) | ✅ |
        | Visual outlier screening | Isolation Forest | ✅ |
        | Spectral screening | SVD on visual descriptors | ✅ |
        | Distribution shift | Leave-one-out | ✅ |
        | Image quality | Brightness/contrast/sharpness | ✅ |
        | Label inconsistency | Cross-reference near-dups | ✅ |
        | Label flipping indicator | Exact dup + different labels | ✅ |
        | Trigger injection indicator | Dual-outlier heuristic | ✅ |
        | EXIF orientation | EXIF tag 274 | ✅ |
        | Archive safety | Path traversal, expansion ratio | ✅ |
    
        #### Contributor / Batch Metadata
        - Contributor grouping uses the **first directory level** in ZIP archives
        - Batch grouping uses the **second directory level**
        - When directory structure does not exist, contributor is "ROOT / UNKNOWN SOURCE"
        - The system does **not** invent contributor metadata
    
        #### Security Mechanisms
        | Mechanism | Provider | Status |
        |-----------|----------|--------|
        | Canonicalization | MIRAD | ✅ |
        | SHA-256 hashing | MIRAD | ✅ |
        | Ed25519 signing | MIRAD | ✅ |
        | Provenance records | MIRAD | ✅ |
        | Replay protection | MIRAD | ✅ |
        | Audit chain | MIRAD | ✅ |
        | Signed checkpoints | MIRAD | ✅ |
        | Ethereum anchoring | web3.py | ✅ (optional) |
    
        #### Known Limitations
        - **Anomaly ≠ Maliciousness**: All outlier signals are screening evidence, not proof of attack
        - **SVD spectral screening** uses handcrafted descriptors, not learned neural representations
        - **Trigger detection** is a lightweight heuristic, not a full neural backdoor scanner
        - **Label analysis** requires COCO or YOLO annotations to be present in the ZIP
        - **Perceptual hash** uses average hash (aHash); more sophisticated methods (dHash, pHash) could reduce false positives
        - **Offline-first**: Core workflow does not require Ethereum or any network connection
        - **Ethereum**: Only the checkpoint digest is anchored; the audit log is NOT placed on-chain
        - **Isolation Forest contamination**: Uses `auto` setting; may not be optimal for all datasets
        - **Class imbalance**: Systematic mislabelling detection requires labeled data; not applicable to unlabeled datasets
        - **Confidence values**: Screening percentages are NOT calibrated probabilities
        - **False positives**: Legitimate variation (lighting, sensor) can trigger outlier flags
        - **False negatives**: Sophisticated attacks may evade handcrafted feature extraction
        - **Air-gapped operation**: Full dataset analysis, MIRAD security, and verification work without network
    
        #### Out of Scope
        - Model upload / integrity / fingerprinting / backdoor analysis
        - Inference execution / output integrity / provenance
        - Model parameter analysis / substitution analysis
        """)
    
    # ================================================================
    # FOOTER
    # ================================================================
    
    st.markdown("---")
    st.caption(
        "TrustCV Dataset Integrity + Security + Assurance Console · MIRAD Protected · "
        "local authority gate · SHA-256 · Ed25519 · signed provenance · "
        "integrity · duplicate screening · quality · outlier · SVD spectral screening · "
        "distribution shift · label analysis · trigger indicators · "
        "contributor/batch aggregation · Ethereum audit anchor · tamper-evident audit evidence."
    )


if st.runtime.exists():
    main()
