import io
import os
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CIFAR10_FILES = {
    "data_batch_1", "data_batch_2", "data_batch_3",
    "data_batch_4", "data_batch_5", "test_batch", "batches.meta"
}


@dataclass
class DatasetInfo:
    format: str
    source: str
    num_samples: int
    channels: int
    height: int
    width: int
    num_classes: int | None
    class_names: list[str]
    has_labels: bool


@dataclass
class DatasetResult:
    images: torch.Tensor
    labels: torch.Tensor | None
    info: DatasetInfo


def _image_to_tensor(image):
    image = image.convert("RGB")
    return torch.from_numpy(
        np.asarray(image, dtype=np.float32)
    ).permute(2, 0, 1) / 255.0


def _read_cifar_batch(raw_bytes):
    obj = pickle.loads(raw_bytes, encoding="latin1")
    data = np.asarray(obj["data"], dtype=np.uint8)
    labels = obj.get("labels", obj.get("fine_labels"))
    images = torch.from_numpy(data).reshape(-1, 3, 32, 32).float() / 255.0
    labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
    return images, labels


def _detect_cifar10_zip(zf):
    names = {Path(name).name for name in zf.namelist()}

    required_train = {
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    }

    return required_train.issubset(names) or "test_batch" in names


def _load_cifar10(zf, max_samples=None, include_cifar_test=False):
    names_by_base = {Path(name).name: name for name in zf.namelist()}

    train_batches = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]

    has_train = all(name in names_by_base for name in train_batches)

    if not has_train and "test_batch" in names_by_base:
        batch_names = ["test_batch"]
    else:
        batch_names = train_batches

        if include_cifar_test and "test_batch" in names_by_base:
            batch_names.append("test_batch")

    image_parts = []
    label_parts = []

    for batch_name in batch_names:
        x, y = _read_cifar_batch(zf.read(names_by_base[batch_name]))
        image_parts.append(x)
        if y is not None:
            label_parts.append(y)

    images = torch.cat(image_parts, dim=0)

    labels = None
    if label_parts:
        labels = torch.cat(label_parts, dim=0)

    if max_samples is not None:
        images = images[:max_samples]
        if labels is not None:
            labels = labels[:max_samples]

    return DatasetResult(
        images=images,
        labels=labels,
        info=DatasetInfo(
            format="cifar10",
            source="user_supplied",
            num_samples=len(images),
            channels=images.shape[1],
            height=images.shape[2],
            width=images.shape[3],
            num_classes=10,
            class_names=[
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
            has_labels=labels is not None,
        ),
    )


def _find_image_members(zf):
    return [
        name for name in zf.namelist()
        if not name.endswith("/") and Path(name).suffix.lower() in IMAGE_EXTENSIONS
    ]


def _load_image_zip(zf, max_samples=None):
    members = _find_image_members(zf)
    if not members:
        raise ValueError("ZIP does not contain supported image files.")

    if max_samples is not None:
        members = members[:max_samples]

    images = []
    labels = []
    class_names = []
    class_to_index = {}
    candidate_classes = []

    for member in members:
        parts = Path(member).parts
        if len(parts) >= 2:
            candidate_classes.append(parts[-2])

    has_class_structure = len(set(candidate_classes)) > 1

    for member in members:
        try:
            image = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
            tensor = _image_to_tensor(image)
            images.append(tensor)

            if has_class_structure:
                class_name = Path(member).parts[-2]
                if class_name not in class_to_index:
                    class_to_index[class_name] = len(class_names)
                    class_names.append(class_name)
                labels.append(class_to_index[class_name])
        except Exception as exc:
            raise ValueError(f"Could not read image '{member}': {exc}") from exc

    if not images:
        raise ValueError("No valid images found.")

    shapes = {tuple(x.shape) for x in images}
    if len(shapes) != 1:
        raise ValueError("Reference images have different dimensions. Use a dataset with consistent image dimensions.")

    tensor = torch.stack(images)
    label_tensor = torch.tensor(labels, dtype=torch.long) if has_class_structure else None

    return DatasetResult(
        images=tensor, labels=label_tensor,
        info=DatasetInfo(
            format="imagefolder_zip" if has_class_structure else "image_zip",
            source="user_supplied", num_samples=len(tensor),
            channels=tensor.shape[1], height=tensor.shape[2], width=tensor.shape[3],
            num_classes=len(class_names) if has_class_structure else None,
            class_names=class_names, has_labels=label_tensor is not None
        )
    )


def load_dataset_from_zip(
    zip_path,
    max_samples=None,
    include_cifar_test=False,
):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Reference dataset must be a ZIP file.")

    with zipfile.ZipFile(zip_path, "r") as zf:
        if _detect_cifar10_zip(zf):
            return _load_cifar10(
                zf,
                max_samples=max_samples,
                include_cifar_test=include_cifar_test
            )
        return _load_image_zip(zf, max_samples=max_samples)


def validate_dataset_for_model(dataset_result, expected_channels=None, expected_height=None, expected_width=None, expected_classes=None):
    info = dataset_result.info
    errors = []
    if expected_channels is not None and info.channels != expected_channels:
        errors.append(f"Model expects {expected_channels} channels, but dataset provides {info.channels}.")
    if expected_height is not None and info.height != expected_height:
        errors.append(f"Model expects height {expected_height}, but dataset provides {info.height}.")
    if expected_width is not None and info.width != expected_width:
        errors.append(f"Model expects width {expected_width}, but dataset provides {info.width}.")
    if expected_classes is not None and info.num_classes is not None and info.num_classes != expected_classes:
        errors.append(f"Model expects {expected_classes} classes, but dataset contains {info.num_classes}.")
    return {"compatible": not errors, "errors": errors}


def dataset_summary(dataset_result):
    info = dataset_result.info
    return {
        "format": info.format, "source": info.source, "num_samples": info.num_samples,
        "shape": [info.channels, info.height, info.width],
        "num_classes": info.num_classes, "class_names": info.class_names,
        "has_labels": info.has_labels
    }
