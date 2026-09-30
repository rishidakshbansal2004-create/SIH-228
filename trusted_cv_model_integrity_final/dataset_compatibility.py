from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _safe_json(zf: zipfile.ZipFile, name: str) -> dict[str, Any] | None:
    try:
        value = json.loads(zf.read(name).decode('utf-8'))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _safe_yaml_names(zf: zipfile.ZipFile, name: str) -> list[str]:
    try:
        import yaml
        obj = yaml.safe_load(zf.read(name).decode('utf-8')) or {}
        names = obj.get('names') if isinstance(obj, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
        if isinstance(names, (list, tuple)):
            return [str(x) for x in names]
    except Exception:
        pass
    return []


def inspect_yolo_model(path: str | Path) -> dict[str, Any]:
    from ultralytics import YOLO
    model = YOLO(str(path))
    names = getattr(model, 'names', None)
    if names is None and hasattr(model, 'model'):
        names = getattr(model.model, 'names', None)
    if isinstance(names, dict):
        class_names = [str(names[k]) for k in sorted(names)]
    elif isinstance(names, (list, tuple)):
        class_names = [str(x) for x in names]
    else:
        class_names = []
    task = str(getattr(model, 'task', 'unknown'))
    imgsz = getattr(model, 'overrides', {}).get('imgsz') if hasattr(model, 'overrides') else None
    if imgsz is None and hasattr(model, 'model'):
        imgsz = getattr(model.model, 'imgsz', None)
    if isinstance(imgsz, tuple):
        imgsz = list(imgsz)
    return {'framework': 'ultralytics', 'task': task, 'num_classes': len(class_names), 'class_names': class_names, 'input_size': imgsz}


def _zip_images(zf: zipfile.ZipFile) -> list[str]:
    return [n for n in zf.namelist() if not n.endswith('/') and Path(n).suffix.lower() in IMAGE_EXTENSIONS]


def _looks_like_cifar10(zf: zipfile.ZipFile) -> bool:
    names = {Path(n).name for n in zf.namelist()}
    train = {f'data_batch_{i}' for i in range(1, 6)}
    return train.issubset(names) or 'test_batch' in names


def _find_coco_json(zf: zipfile.ZipFile) -> tuple[str | None, dict[str, Any] | None]:
    candidates = [n for n in zf.namelist() if n.lower().endswith('.json')]
    for name in candidates:
        obj = _safe_json(zf, name)
        if isinstance(obj, dict) and {'images', 'categories'}.issubset(obj.keys()):
            return name, obj
    return None, None


def _find_yaml(zf: zipfile.ZipFile) -> tuple[str | None, list[str]]:
    for n in zf.namelist():
        if Path(n).name.lower() in {'data.yaml', 'dataset.yaml'} or n.lower().endswith(('/data.yaml', '/dataset.yaml')):
            return n, _safe_yaml_names(zf, n)
    return None, []


def _validate_sample_images(zf: zipfile.ZipFile, names: list[str], limit: int = 32) -> dict[str, Any]:
    bad, shapes = [], []
    for name in names[:limit]:
        try:
            raw = zf.read(name)
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            with Image.open(io.BytesIO(raw)) as im:
                shapes.append([im.width, im.height, len(im.getbands())])
        except Exception as exc:
            bad.append({'file': name, 'error': str(exc)})
    return {'checked': min(len(names), limit), 'invalid': bad, 'valid': not bad, 'sample_shapes': shapes[:10]}


def _validate_yolo_labels(zf: zipfile.ZipFile, image_names: list[str], label_names: list[str], num_classes: int | None, limit: int = 1000) -> dict[str, Any]:
    labels = {Path(n).as_posix(): n for n in label_names}
    image_stems = {}
    for im in image_names:
        p = Path(im)
        image_stems[p.stem] = p
    matched, missing = 0, 0
    malformed, out_of_range, unique_ids, annotation_types = [], [], set(), set()
    COORD_EPS = 1e-3
    for im_name in image_names[:limit]:
        p = Path(im_name)
        candidates = [str(p.with_suffix('.txt')), str(Path(*p.parts[:-1], 'labels', p.stem + '.txt'))]
        label_file = next((c for c in candidates if c in labels), None)
        if label_file is None:
            # Common images/train -> labels/train mirror.
            parts = list(p.parts)
            if 'images' in parts:
                parts[parts.index('images')] = 'labels'
                c = str(Path(*parts).with_suffix('.txt'))
                label_file = labels.get(c)
        if label_file is None:
            missing += 1
            continue
        matched += 1
        try:
            content = zf.read(label_file).decode('utf-8').strip()
        except Exception as exc:
            malformed.append({'file': label_file, 'error': str(exc)})
            continue
        if not content:
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            parts = line.split()
            if len(parts) < 5:
                malformed.append({'file': label_file, 'line': line_no, 'error': 'YOLO annotation has too few fields'})
                continue
            try:
                cid = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except Exception:
                malformed.append({'file': label_file, 'line': line_no, 'error': 'Non-numeric YOLO annotation'})
                continue

            # Standard YOLO detection: class + 4 normalized bbox values.
            # YOLO segmentation: class + an even-length sequence of normalized
            # polygon x/y coordinates. Roboflow exports commonly use this form.
            if len(parts) == 5:
                annotation_type = 'bbox'
            elif len(parts) >= 7 and len(vals) % 2 == 0:
                annotation_type = 'polygon'
            else:
                malformed.append({
                    'file': label_file, 'line': line_no,
                    'error': 'YOLO annotation must be either class x_center y_center width height or class followed by an even number of polygon x/y coordinates'
                })
                continue

            unique_ids.add(cid)
            if num_classes is not None and not (0 <= cid < num_classes):
                out_of_range.append({'file': label_file, 'line': line_no, 'class_id': cid})
            

            if any(v < -COORD_EPS or v > 1 + COORD_EPS for v in vals):
                malformed.append({
               'file': label_file,
               'line': line_no,
               'error': 'YOLO coordinates are outside the valid normalized range'
            })
                continue

            # For polygon labels, derive a normalized bounding box for downstream
            # detector compatibility without modifying the user's source labels.
            if annotation_type == 'polygon':
                xs = vals[0::2]
                ys = vals[1::2]
                if len(xs) < 3:
                    malformed.append({'file': label_file, 'line': line_no, 'error': 'Polygon must contain at least 3 points'})
                    continue
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                if x_max <= x_min or y_max <= y_min:
                    malformed.append({'file': label_file, 'line': line_no, 'error': 'Polygon has zero-area bounding box'})
                    continue
                annotation_types.add('polygon')
            else:
                annotation_types.add('bbox')
    checked = min(len(image_names), limit)
    return {
        'present': bool(label_names),
        'checked_images': checked,
        'images_with_labels': matched,
        'images_without_labels': missing,
        'coverage': (matched / checked) if checked else 0.0,
        'unique_class_ids': sorted(unique_ids),
        'annotation_types': sorted(annotation_types),
        'malformed': malformed[:20],
        'out_of_range_class_ids': out_of_range[:20],
        'valid': not malformed and not out_of_range,
    }


def inspect_reference_zip(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return {'format': 'invalid', 'errors': ['Reference dataset must be a ZIP file.']}
    with zipfile.ZipFile(path, 'r') as zf:
        if _looks_like_cifar10(zf):
            names = {Path(n).name for n in zf.namelist()}
            return {'format': 'cifar10', 'task': 'image_classification', 'num_classes': 10,
                    'class_names': ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
                    'num_images': 10000 if 'test_batch' in names else None, 'has_annotations': False,
                    'label_status': 'not_applicable',
                    'image_validation': {'checked': 0, 'invalid': [], 'valid': True, 'sample_shapes': [[32,32,3]]}}

        image_names = _zip_images(zf)
        coco_name, coco = _find_coco_json(zf)
        yaml_name, yaml_names = _find_yaml(zf)
        image_validation = _validate_sample_images(zf, image_names)

        # Detect a YOLO detection dataset when labels are present, or when a data.yaml is present.
        label_names = [n for n in zf.namelist() if not n.endswith('/') and Path(n).suffix.lower() == '.txt' and ('labels' in Path(n).parts or Path(n).name.lower().startswith('label'))]
        if yaml_name or label_names:
            label_validation = _validate_yolo_labels(zf, image_names, label_names, len(yaml_names) if yaml_names else None)
            class_names = yaml_names
            class_source = 'data.yaml' if yaml_names else None
            return {
                'format': 'yolo_detection', 'task': 'object_detection', 'num_images': len(image_names),
                'num_classes': len(class_names) if class_names else (max(label_validation['unique_class_ids']) + 1 if label_validation['unique_class_ids'] else None),
                'class_names': class_names, 'class_names_source': class_source,
                'has_annotations': bool(label_names),
                'label_status': 'present' if label_names else 'absent',
                'yaml_file': yaml_name,
                'annotation_validation': label_validation,
                'image_validation': image_validation,
            }

        if coco is not None:
            categories = coco.get('categories') or []
            cat_names = [str(c.get('name')) for c in categories if isinstance(c, dict) and c.get('name') is not None]
            image_ids = {x.get('id') for x in coco.get('images', []) if isinstance(x, dict)}
            ann = coco.get('annotations') or []
            bad_boxes = 0; missing_image_refs = 0; bad_category_refs = 0
            category_ids = {c.get('id') for c in categories if isinstance(c, dict)}
            for a in ann[:1000]:
                if not isinstance(a, dict): continue
                if a.get('image_id') not in image_ids: missing_image_refs += 1
                if a.get('category_id') not in category_ids: bad_category_refs += 1
                bbox = a.get('bbox')
                if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] < 0 or bbox[3] < 0): bad_boxes += 1
            return {'format':'coco','task':'object_detection','annotation_file':coco_name,
                    'num_images':len(coco.get('images',[])),'num_annotations':len(ann),'num_classes':len(cat_names),
                    'class_names':cat_names,'has_annotations':True,'label_status':'present',
                    'annotation_validation':{'checked_annotations':min(len(ann),1000),'missing_image_refs':missing_image_refs,
                                             'invalid_bboxes':bad_boxes,'invalid_category_refs':bad_category_refs,
                                             'valid':missing_image_refs==0 and bad_boxes==0 and bad_category_refs==0},
                    'image_validation':image_validation}

        return {'format':'image_zip','task':'object_detection_reference_images','num_images':len(image_names),
                'num_classes':None,'class_names':[],'has_annotations':False,'label_status':'absent',
                'image_validation':image_validation}


def validate_yolo_reference(model_info: dict[str, Any], dataset_info: dict[str, Any]) -> dict[str, Any]:
    errors=[]; warnings=[]; checks=[]
    task=str(model_info.get('task','unknown')).lower()
    model_classes=[str(x).strip().lower() for x in model_info.get('class_names',[]) if str(x).strip()]
    fmt=str(dataset_info.get('format','unknown')); dtask=str(dataset_info.get('task','unknown'))
    dataset_classes=[str(x).strip().lower() for x in dataset_info.get('class_names',[]) if str(x).strip()]
    task_ok=task in {'detect','detection','object_detection'}
    checks.append({'id':'task','passed':task_ok,'detail':f"Model task: {model_info.get('task')}"})
    if not task_ok: errors.append(f"The uploaded YOLO model exposes task '{model_info.get('task')}', not object detection.")

    is_classification = fmt == 'cifar10' or dtask == 'image_classification'
    checks.append({'id':'dataset_task','passed':not is_classification,'detail':'Classification data is not object-detection reference data.' if is_classification else 'Reference data is suitable for object-detection analysis.'})
    if is_classification: errors.append('The supplied reference dataset is classification data and is not compatible with the uploaded YOLO object detector.')

    has_images=bool(dataset_info.get('num_images'))
    image_ok=bool(dataset_info.get('image_validation',{}).get('valid',True))
    checks.append({'id':'images','passed':has_images and image_ok,'detail':f"Reference images: {dataset_info.get('num_images',0)}"})
    if not has_images: errors.append('The reference dataset contains no usable images.')
    elif not image_ok: errors.append('One or more sampled reference images could not be decoded.')

    if fmt in {'coco','yolo_detection'}:
        av=dataset_info.get('annotation_validation',{})
        ann_ok=bool(av.get('valid',True))
        checks.append({'id':'annotations','passed':ann_ok,'detail':f"Annotations: {dataset_info.get('label_status','unknown')}"})
        if not ann_ok: errors.append('The supplied detection annotations contain malformed records, invalid boxes, or invalid class references.')
        if fmt=='yolo_detection' and not dataset_info.get('has_annotations'):
            warnings.append('No YOLO label files were supplied; image-only analysis can still run, but annotation-dependent checks are unavailable.')

    if model_classes and dataset_classes:
        ms=set(model_classes); ds=set(dataset_classes); missing=sorted(ms-ds); extra=sorted(ds-ms)
        if missing:
            checks.append({'id':'class_space','passed':False,'detail':f'Missing model classes in dataset: {missing}'})
            errors.append('The reference dataset does not contain all deployed model classes: '+', '.join(missing)+'.')
        else:
            checks.append({'id':'class_space','passed':True,'detail':'Model class names are represented in the reference dataset.'})
            if extra: warnings.append('The reference dataset contains additional categories not exposed by the uploaded model: '+', '.join(extra)+'.')
    elif model_classes and not dataset_classes:
        checks.append({'id':'class_space','passed':None,'detail':'No verified reference class mapping is available.'})
        warnings.append('Class-space compatibility could not be fully verified because the reference data has no category mapping.')
    else:
        checks.append({'id':'class_space','passed':None,'detail':'Model class names were not available for semantic comparison.'})
        warnings.append('The model did not expose class names, so semantic class-space verification is limited.')

    # Labels are explicitly informational for image-only analysis.
    label_status=dataset_info.get('label_status')
    checks.append({'id':'labels','passed':True if label_status=='present' else None,
                   'detail':'Detection labels present.' if label_status=='present' else 'Detection labels not present; image-only analysis remains supported.'})

    # Missing category metadata is a limitation, not an incompatibility.
    # Images (and valid annotations when supplied) are sufficient for
    # image-based behavioral analysis. Semantic class comparison is simply
    # marked as unavailable when the dataset has no category mapping.
    if errors:
        compatible = False
        disposition = 'blocked'
    elif warnings:
        compatible = True
        disposition = 'review'
    else:
        compatible = True
        disposition = 'compatible'

    return {
        'compatible': compatible,
        'status': 'review' if disposition == 'review' else ('blocked' if disposition == 'blocked' else 'compatible'),
        'disposition': disposition,
        'checks': checks,
        'errors': errors,
        'warnings': warnings,
    }


def preprocess_reference_images(
    source: str | Path | list[Any],
    imgsz: int = 640,
    max_samples: int = 16,
) -> list[Any]:
    """
    Standardized Preprocessor for Phase 4 Reference Images.
    Loads, validates, decodes, and standardizes reference frames to (imgsz, imgsz) RGB arrays.
    
    Supports:
    1. ZIP archive containing images (.zip)
    2. Directory containing image files
    3. Single image file or list of image files / numpy arrays
    
    Returns:
    List of uint8 RGB numpy arrays shaped [imgsz, imgsz, 3].
    """
    import numpy as np
    from PIL import Image

    processed_images: list[np.ndarray] = []

    # Case 1: ZIP Archive
    if isinstance(source, (str, Path)) and Path(source).is_file() and zipfile.is_zipfile(Path(source)):
        with zipfile.ZipFile(Path(source), 'r') as zf:
            img_members = _zip_images(zf)[:max_samples]
            for name in img_members:
                try:
                    data = zf.read(name)
                    with Image.open(io.BytesIO(data)) as im:
                        rgb_im = im.convert('RGB')
                        if rgb_im.size != (imgsz, imgsz):
                            rgb_im = rgb_im.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
                        processed_images.append(np.array(rgb_im, dtype=np.uint8))
                except Exception:
                    continue

    # Case 2: Directory Path
    elif isinstance(source, (str, Path)) and Path(source).is_dir():
        files = [p for p in Path(source).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS][:max_samples]
        for p in files:
            try:
                with Image.open(p) as im:
                    rgb_im = im.convert('RGB')
                    if rgb_im.size != (imgsz, imgsz):
                        rgb_im = rgb_im.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
                    processed_images.append(np.array(rgb_im, dtype=np.uint8))
            except Exception:
                continue

    # Case 3: List of items (paths, filenames, or numpy arrays)
    elif isinstance(source, (list, tuple)):
        for item in source[:max_samples]:
            if isinstance(item, np.ndarray):
                arr = item.copy()
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 1:
                    arr = np.concatenate([arr] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                
                h, w = arr.shape[:2]
                if (w, h) != (imgsz, imgsz):
                    im = Image.fromarray(arr).resize((imgsz, imgsz), Image.Resampling.BILINEAR)
                    arr = np.array(im, dtype=np.uint8)
                processed_images.append(arr)
            elif isinstance(item, (str, Path)) and Path(item).exists():
                try:
                    with Image.open(Path(item)) as im:
                        rgb_im = im.convert('RGB')
                        if rgb_im.size != (imgsz, imgsz):
                            rgb_im = rgb_im.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
                        processed_images.append(np.array(rgb_im, dtype=np.uint8))
                except Exception:
                    continue

    # Case 4: Single file path
    elif isinstance(source, (str, Path)) and Path(source).exists():
        try:
            with Image.open(Path(source)) as im:
                rgb_im = im.convert('RGB')
                if rgb_im.size != (imgsz, imgsz):
                    rgb_im = rgb_im.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
                processed_images.append(np.array(rgb_im, dtype=np.uint8))
        except Exception:
            pass

    return processed_images


def assess_dataset_tier(
    dataset_path: str | Path | None,
    model_info: dict[str, Any],
    model_type: str,
    dataset_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluates dataset compatibility into explicit operational tiers:
    - NOT_COMPATIBLE: Non-image files, corrupted archive, or empty -> BLOCKED
    - PREPROCESSED_COMPATIBLE: Valid images auto-standardized to 640x640 RGB -> B3D ENABLED, TRACE DISABLED
    - HIGHLY_COMPATIBLE: Valid images + verified ground truth labels matching class space -> B3D + TRACE ENABLED
    - INTERNAL_RESOLVED: When model uses built-in reference frames / testbeds -> AUTO-RESOLVED
    """
    policy = dataset_policy or {}
    source_policy = policy.get("source", "internal_reference_optional_user")

    # Case A: No dataset supplied
    if dataset_path is None or (isinstance(dataset_path, (str, Path)) and not Path(dataset_path).exists()):
        if source_policy in ("internal_auto", "internal_reference_optional_user"):
            return {
                "status": "INTERNAL_RESOLVED",
                "compatible": True,
                "disposition": "internal_resolved",
                "tier": "INTERNAL_RESOLVED",
                "trace_eligible": False,
                "detail": "TrustCV internal reference dataset auto-configured for B3D backdoor scanning.",
                "num_images": 4,
                "preprocessed_count": 4,
                "checks": [
                    {"id": "source", "passed": True, "detail": "Internal verified baseline canvas/frames"},
                    {"id": "b3d_trigger_inversion", "passed": True, "detail": "Active (Zero-Knowledge Black-Box)"},
                    {"id": "trace_behavioral", "passed": False, "detail": "Standby (requires specific user dataset)"},
                ],
                "errors": [],
                "warnings": ["Using internal reference frames. Upload custom in-distribution dataset if TRACE behavioral evidence is also desired."],
            }
        else:
            return {
                "status": "NOT_COMPATIBLE",
                "compatible": False,
                "disposition": "blocked",
                "tier": "NOT_COMPATIBLE",
                "trace_eligible": False,
                "detail": "This model requires a clean calibration dataset matching its class distribution before proceeding.",
                "num_images": 0,
                "preprocessed_count": 0,
                "checks": [
                    {"id": "dataset_presence", "passed": False, "detail": "No dataset supplied"},
                ],
                "errors": ["Clean calibration dataset is mandatory for Neural Cleanse and Activation Ablation."],
                "warnings": [],
            }

    dpath = Path(dataset_path)

    # Case B: Inspect uploaded file/archive
    if dpath.is_file() and not zipfile.is_zipfile(dpath) and dpath.suffix.lower() not in IMAGE_EXTENSIONS:
        return {
            "status": "NOT_COMPATIBLE",
            "compatible": False,
            "disposition": "blocked",
            "tier": "NOT_COMPATIBLE",
            "trace_eligible": False,
            "detail": f"Uploaded file format '{dpath.suffix}' is not a supported image dataset. Please upload a .zip archive, folder, or image files.",
            "num_images": 0,
            "preprocessed_count": 0,
            "checks": [
                {"id": "format_validation", "passed": False, "detail": f"Invalid extension {dpath.suffix}"},
            ],
            "errors": ["Only image datasets (.zip containing images or raw images) are supported."],
            "warnings": [],
        }

    # Inspect images and labels
    if dpath.is_file() and zipfile.is_zipfile(dpath):
        dataset_info = inspect_reference_zip(dpath)
        img_count = dataset_info.get("num_images") or 0
        if img_count == 0:
            return {
                "status": "NOT_COMPATIBLE",
                "compatible": False,
                "disposition": "blocked",
                "tier": "NOT_COMPATIBLE",
                "trace_eligible": False,
                "detail": "The uploaded ZIP archive contains 0 valid image files (.jpg, .png, etc.). Please upload a valid image dataset.",
                "num_images": 0,
                "preprocessed_count": 0,
                "checks": [
                    {"id": "image_presence", "passed": False, "detail": "0 images detected in archive"},
                ],
                "errors": ["Archive contains no supported image files."],
                "warnings": [],
            }

        # Test preprocessing standardization
        sample_pre = preprocess_reference_images(dpath, imgsz=640, max_samples=8)
        preprocessed_count = len(sample_pre)

        # Check YOLO compatibility
        if model_type in ("yolo", "onnx"):
            compat = validate_yolo_reference(model_info, dataset_info)
            has_labels = dataset_info.get("label_status") == "present"
            class_match = any(c.get("id") == "class_space" and c.get("passed") is True for c in compat.get("checks", []))

            if has_labels and class_match and compat.get("compatible"):
                return {
                    "status": "HIGHLY_COMPATIBLE",
                    "compatible": True,
                    "disposition": "highly_compatible",
                    "tier": "HIGHLY_COMPATIBLE",
                    "trace_eligible": True,
                    "detail": "Verified reference images with ground-truth labels matching model class space. Unlocks both B3D Trigger Inversion and TRACE Behavioral Assurance.",
                    "num_images": img_count,
                    "preprocessed_count": preprocessed_count,
                    "dataset_info": dataset_info,
                    "checks": [
                        *compat.get("checks", []),
                        {"id": "preprocessing", "passed": True, "detail": f"Sample standardized to 640×640 RGB ({preprocessed_count} frames verified)"},
                        {"id": "trace_readiness", "passed": True, "detail": "Eligible for TRACE behavioral anomaly calibration"},
                    ],
                    "errors": compat.get("errors", []),
                    "warnings": compat.get("warnings", []),
                }
            else:
                return {
                    "status": "PREPROCESSED_COMPATIBLE",
                    "compatible": True,
                    "disposition": "preprocessed",
                    "tier": "PREPROCESSED_COMPATIBLE",
                    "trace_eligible": False,
                    "detail": f"Reference images ({img_count} found) successfully standardized to 640×640 RGB. Compatible for B3D Black-Box Backdoor Inversion.",
                    "num_images": img_count,
                    "preprocessed_count": preprocessed_count,
                    "dataset_info": dataset_info,
                    "checks": [
                        {"id": "image_presence", "passed": True, "detail": f"{img_count} images found in dataset"},
                        {"id": "preprocessing", "passed": True, "detail": f"Auto-standardized to 640×640 RGB ({preprocessed_count} sample frames verified)"},
                        {"id": "b3d_inversion", "passed": True, "detail": "Active for spatial backdoor trigger search"},
                        {"id": "trace_behavioral", "passed": False, "detail": "Disabled (ground-truth bounding box labels matching class space not found)"},
                    ],
                    "errors": [],
                    "warnings": ["Labels are absent or do not fully match model classes. B3D trigger search will run; TRACE behavioral analysis is bypassed."],
                }

        # Check SmallCNN / Classification compatibility
        elif model_type == "smallcnn":
            fmt = dataset_info.get("format")
            if fmt == "cifar10":
                return {
                    "status": "HIGHLY_COMPATIBLE",
                    "compatible": True,
                    "disposition": "highly_compatible",
                    "tier": "HIGHLY_COMPATIBLE",
                    "trace_eligible": True,
                    "detail": "Verified 10-class CIFAR-10 reference dataset. Fully compatible with Neural Cleanse and Activation Ablation.",
                    "num_images": img_count,
                    "preprocessed_count": preprocessed_count,
                    "dataset_info": dataset_info,
                    "checks": [
                        {"id": "format", "passed": True, "detail": "CIFAR-10 clean binary reference"},
                        {"id": "class_space", "passed": True, "detail": "10-class alignment verified"},
                        {"id": "neural_cleanse", "passed": True, "detail": "Ready for multi-class trigger reverse engineering"},
                    ],
                    "errors": [],
                    "warnings": [],
                }
            else:
                return {
                    "status": "PREPROCESSED_COMPATIBLE",
                    "compatible": True,
                    "disposition": "preprocessed",
                    "tier": "PREPROCESSED_COMPATIBLE",
                    "trace_eligible": False,
                    "detail": "Classification images preprocessed. Compatible for general query evaluation.",
                    "num_images": img_count,
                    "preprocessed_count": preprocessed_count,
                    "dataset_info": dataset_info,
                    "checks": [
                        {"id": "image_presence", "passed": True, "detail": f"{img_count} images found"},
                    ],
                    "errors": [],
                    "warnings": ["Non-standard CIFAR-10 distribution; verify class mapping before full ablation."],
                }

    # Raw image file or directory fallback
    sample_pre = preprocess_reference_images(dpath, imgsz=640, max_samples=8)
    if not sample_pre:
        return {
            "status": "NOT_COMPATIBLE",
            "compatible": False,
            "disposition": "blocked",
            "tier": "NOT_COMPATIBLE",
            "trace_eligible": False,
            "detail": "Could not decode any valid images from the supplied input.",
            "num_images": 0,
            "preprocessed_count": 0,
            "checks": [{"id": "image_decoding", "passed": False, "detail": "Failed to decode image data"}],
            "errors": ["Failed to load image samples from source."],
            "warnings": [],
        }

    return {
        "status": "PREPROCESSED_COMPATIBLE",
        "compatible": True,
        "disposition": "preprocessed",
        "tier": "PREPROCESSED_COMPATIBLE",
        "trace_eligible": False,
        "detail": f"{len(sample_pre)} reference images standardized to 640×640 RGB. Compatible for B3D backdoor scanning.",
        "num_images": len(sample_pre),
        "preprocessed_count": len(sample_pre),
        "checks": [
            {"id": "image_presence", "passed": True, "detail": f"{len(sample_pre)} images standardized"},
            {"id": "b3d_inversion", "passed": True, "detail": "Active for spatial backdoor trigger search"},
        ],
        "errors": [],
        "warnings": ["Unlabeled images: B3D trigger inversion enabled; TRACE behavioral bypassed."],
    }

