"""
Phase 4 — Model Integrity, White-Box Stage 1: Neural Cleanse

Neural Cleanse reverse-engineers a candidate trigger for every target class
using clean reference images supplied by the user.

The reference images are NOT downloaded automatically.
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from poison_and_train import SmallCNN, add_trigger


DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

IMG_SHAPE = (3, 32, 32)
NUM_CLASSES = 10

NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)

STEPS = 800
LR = 0.05
LAMBDA_INIT = 1e-4
LAMBDA_MAX = 5.0
ASR_TARGET = 0.90
MAD_ANOMALY_THRESHOLD = 2.0
MASK_INIT_BIAS = -4.0


def _validate_reference_images(reference_images):
    if reference_images is None:
        raise ValueError(
            "Clean reference images are required for Neural Cleanse."
        )

    if not isinstance(reference_images, torch.Tensor):
        raise TypeError(
            "reference_images must be a torch.Tensor."
        )

    if reference_images.ndim != 4:
        raise ValueError(
            "reference_images must have shape [N,C,H,W], "
            f"got {tuple(reference_images.shape)}"
        )

    if reference_images.shape[1] != 3:
        raise ValueError(
            "Neural Cleanse currently expects RGB images with 3 channels."
        )

    if reference_images.shape[2] != 32 or reference_images.shape[3] != 32:
        raise ValueError(
            "The current SmallCNN testbed expects 32x32 reference images. "
            f"Received {reference_images.shape[2]}x{reference_images.shape[3]}."
        )

    reference_images = reference_images.float()

    if reference_images.max().item() > 1.0:
        reference_images = reference_images / 255.0

    reference_images = torch.clamp(reference_images, 0.0, 1.0)

    return reference_images


def _normalize(x, mean=NORM_MEAN, std=NORM_STD):
    mean = torch.tensor(
        mean,
        device=x.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        std,
        device=x.device
    ).view(1, 3, 1, 1)

    return (x - mean) / std


def reverse_engineer_trigger(
    model,
    target_class,
    calib_imgs,
    steps=STEPS,
    lr=LR,
    device=DEVICE
):
    """
    Optimize a mask and pattern that forces target_class on clean images
    while minimizing the mask size.
    """

    model.eval()

    calib_imgs = calib_imgs.to(device)

    n, c, h, w = calib_imgs.shape

    mask_param = torch.full(
        (1, 1, h, w),
        MASK_INIT_BIAS,
        device=device,
        requires_grad=True
    )

    pattern_param = torch.zeros(
        1,
        c,
        h,
        w,
        device=device,
        requires_grad=True
    )

    opt = torch.optim.Adam(
        [mask_param, pattern_param],
        lr=lr
    )

    lam = LAMBDA_INIT

    target = torch.full(
        (n,),
        target_class,
        dtype=torch.long,
        device=device
    )

    for step in range(steps):

        mask = torch.sigmoid(mask_param)
        pattern = torch.sigmoid(pattern_param)

        blended = (
            (1 - mask) * calib_imgs
            + mask * pattern
        )

        logits = model(_normalize(blended))

        cls_loss = F.cross_entropy(
            logits,
            target
        )

        mask_size = mask.mean()

        loss = cls_loss + lam * mask_size

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == steps - 1:

            with torch.no_grad():
                asr = (
                    logits.argmax(1) == target
                ).float().mean().item()

            if asr > ASR_TARGET:
                lam = min(
                    lam * 1.5,
                    LAMBDA_MAX
                )
            else:
                lam = max(
                    lam * 0.8,
                    LAMBDA_INIT
                )

    with torch.no_grad():

        mask = torch.sigmoid(mask_param)
        pattern = torch.sigmoid(pattern_param)

        blended = (
            (1 - mask) * calib_imgs
            + mask * pattern
        )

        logits = model(_normalize(blended))

        final_asr = (
            logits.argmax(1) == target
        ).float().mean().item()

        mask_size = mask.mean().item()

    return {
        "target_class": target_class,
        "mask": mask.detach().cpu(),
        "pattern": pattern.detach().cpu(),
        "mask_size": mask_size,
        "optimization_asr": final_asr
    }


def mad_anomaly_scores(mask_sizes):
    sizes = np.array(
        mask_sizes,
        dtype=float
    )

    median = np.median(sizes)

    mad = np.median(
        np.abs(sizes - median)
    ) + 1e-12

    anomaly = np.where(
        sizes < median,
        (median - sizes) / mad,
        0.0
    )

    return anomaly, median, mad


@torch.no_grad()
def verify_trigger(
    model,
    mask,
    pattern,
    target_class,
    held_out_imgs,
    device=DEVICE
):
    """
    Verify the recovered trigger on clean images that were not used
    during optimization.
    """

    model.eval()

    held_out_imgs = held_out_imgs.to(device)
    mask = mask.to(device)
    pattern = pattern.to(device)

    blended = (
        (1 - mask) * held_out_imgs
        + mask * pattern
    )

    logits = model(
        _normalize(blended)
    )

    preds = logits.argmax(1)

    asr = (
        preds == target_class
    ).float().mean().item()

    return asr


def _trigger_geometry(mask):
    mask_2d = mask.detach().cpu().squeeze()
    threshold = float(mask_2d.mean().item())
    active = mask_2d > threshold
    coords = torch.nonzero(active, as_tuple=False)

    if coords.numel() == 0:
        return {
            "threshold": threshold,
            "bbox_xyxy": None,
            "active_pixels": 0,
            "image_area": int(mask_2d.numel()),
        }

    y0 = int(coords[:, 0].min().item())
    x0 = int(coords[:, 1].min().item())
    y1 = int(coords[:, 0].max().item())
    x1 = int(coords[:, 1].max().item())

    return {
        "threshold": threshold,
        "bbox_xyxy": [x0, y0, x1, y1],
        "active_pixels": int(active.sum().item()),
        "image_area": int(mask_2d.numel()),
    }


def run_neural_cleanse(
    model,
    reference_images,
    num_classes=NUM_CLASSES,
    n_calib=512,
    n_holdout=512,
    device=DEVICE
):
    """
    Full Neural Cleanse pipeline.

    reference_images:
        Clean images supplied by the user.
        Expected shape: [N, 3, 32, 32]
        Expected range: [0, 1]

    Calibration and holdout images are independently sampled from the
    full supplied reference dataset using seeds 0 and 1.
    """

    reference_images = _validate_reference_images(
        reference_images
    )

    required = n_calib + n_holdout

    if len(reference_images) < required:
        raise ValueError(
            f"Neural Cleanse requires at least {required} clean "
            f"reference images, but received {len(reference_images)}."
        )

    # Reproduce the original Neural Cleanse testbed sampling:
    # calibration and holdout are independently sampled from the full
    # user-supplied CIFAR-10 test reference using seeds 0 and 1.
    generator_calib = torch.Generator().manual_seed(0)
    generator_holdout = torch.Generator().manual_seed(1)

    calib_idx = torch.randperm(
        len(reference_images),
        generator=generator_calib
    )[:n_calib]

    holdout_idx = torch.randperm(
        len(reference_images),
        generator=generator_holdout
    )[:n_holdout]

    calib_imgs = reference_images[calib_idx]
    holdout_imgs = reference_images[holdout_idx]

    print(
        f"[Neural Cleanse] Using {n_calib} calibration images "
        f"and {n_holdout} held-out images.",
        flush=True
    )

    per_class = []

    for c in range(num_classes):

        print(
            f"[Neural Cleanse] optimizing candidate trigger "
            f"for class {c}/{num_classes - 1}...",
            flush=True
        )

        result = reverse_engineer_trigger(
            model,
            c,
            calib_imgs,
            device=device
        )

        per_class.append(result)

    mask_sizes = [
        r["mask_size"]
        for r in per_class
    ]

    anomaly_scores, median, mad = (
        mad_anomaly_scores(mask_sizes)
    )

    flags = []

    for r, score in zip(
        per_class,
        anomaly_scores
    ):

        flags.append({
            "target_class": r["target_class"],
            "mask_size": r["mask_size"],
            "optimization_asr": r["optimization_asr"],
            "anomaly_index": float(score)
        })

    suspect_idx = int(
        np.argmax(anomaly_scores)
    )

    suspect = per_class[
        suspect_idx
    ]

    suspect_flagged = (
        anomaly_scores[suspect_idx]
        > MAD_ANOMALY_THRESHOLD
    )

    verification_asr = None

    if suspect_flagged:

        verification_asr = verify_trigger(
            model,
            suspect["mask"],
            suspect["pattern"],
            suspect["target_class"],
            holdout_imgs,
            device=device
        )

    verdict_flagged = bool(
        suspect_flagged
        and verification_asr is not None
        and verification_asr > 0.80
    )

    if verdict_flagged:

        disposition = "quarantine"

        reason = (
            f"Class {suspect['target_class']} required an "
            f"anomalously small mask "
            f"(anomaly_index="
            f"{anomaly_scores[suspect_idx]:.2f}, "
            f"threshold={MAD_ANOMALY_THRESHOLD}) "
            f"to force misclassification, and the recovered "
            f"trigger achieved "
            f"{verification_asr:.1%} attack success rate "
            f"on held-out clean images."
        )

    elif suspect_flagged:

        disposition = "review"

        reason = (
            f"Class {suspect['target_class']} showed a mild "
            f"mask-size anomaly "
            f"(anomaly_index="
            f"{anomaly_scores[suspect_idx]:.2f}) "
            f"but did not clear held-out verification."
        )

    else:

        disposition = "accept"

        reason = (
            "No class showed an anomalously small trigger "
            "mask relative to the others."
        )

    confidence = (
        float(
            min(
                anomaly_scores[suspect_idx]
                / (2 * MAD_ANOMALY_THRESHOLD),
                1.0
            )
        )
        if suspect_flagged
        else
        1.0 - float(
            min(
                max(anomaly_scores),
                1.0
            )
            / MAD_ANOMALY_THRESHOLD
        )
    )

    report = {

        "method": (
            "Neural Cleanse "
            "(white-box, requires gradient access)"
        ),

        "access_level": "white_box",

        "disposition": disposition,

        "reason": reason,

        "evidence": {

            "reference_data": {
                "source": "user_supplied",
                "total_images": len(reference_images),
                "calibration_images": n_calib,
                "holdout_images": n_holdout,
                "split_method": "independent_random",
                "calibration_seed": 0,
                "holdout_seed": 1
            },

            "per_class_flags": flags,

            "median_mask_size": float(median),

            "mad": float(mad),

            "suspect_class": (
                suspect["target_class"]
                if suspect_flagged
                else None
            ),

            "verification_asr_on_holdout":
                verification_asr,

            "recovered_trigger": (
                {
                    "target_class": suspect["target_class"],
                    "mask_size": suspect["mask_size"],
                    "optimization_asr": suspect["optimization_asr"],
                    **_trigger_geometry(suspect["mask"])
                }
                if suspect_flagged
                else None
            )
        },

        "confidence": confidence,

        "limitations": [

            "Does not prove the recovered pattern is the "
            "attacker's exact trigger — it is a candidate "
            "reverse-engineered pattern that behaves like one.",

            "Requires white-box gradient access.",

            "Assumes a patch-style trigger reasonably "
            "localized in input space; may miss distributed, "
            "semantic, or non-additive triggers.",

            "Optimization ASR is measured on the same batch "
            "used during optimization and is optimistic; "
            "only held-out verification ASR should be treated "
            "as validation evidence.",

            "The supplied reference dataset must be representative "
            "of the legitimate model input distribution.",

            "Current SmallCNN integration expects RGB 32x32 "
            "images with CIFAR-10 normalization."
        ]
    }

    return report, per_class


if __name__ == "__main__":

    import argparse
    from PIL import Image
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True
    )

    parser.add_argument(
        "--reference_dir",
        required=True,
        help="Directory containing clean reference images"
    )

    parser.add_argument(
        "--out",
        default="neural_cleanse_report.json"
    )

    args = parser.parse_args()

    model = SmallCNN(
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            args.weights,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    image_paths = sorted(
        [
            p for p in Path(
                args.reference_dir
            ).iterdir()
            if p.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            }
        ]
    )

    if not image_paths:
        raise ValueError(
            "No supported images found in reference_dir."
        )

    images = []

    for path in image_paths:

        image = Image.open(path).convert("RGB")

        image = image.resize(
            (32, 32)
        )

        array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        tensor = torch.from_numpy(
            array
        ).permute(
            2, 0, 1
        )

        images.append(tensor)

    reference_images = torch.stack(
        images
    )

    report, per_class = run_neural_cleanse(
        model,
        reference_images
    )

    print(
        "\n=== Neural Cleanse report ==="
    )

    print(
        json.dumps(
            report,
            indent=2,
            default=str
        )
    )

    with open(
        args.out,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str
        )

    print(
        f"\nSaved report to {args.out}"
    )

    suspect_class = report[
        "evidence"
    ][
        "suspect_class"
    ]

    if suspect_class is not None:

        suspect = next(
            r for r in per_class
            if r["target_class"]
            == suspect_class
        )

        mask = suspect["mask"]
        pattern = suspect["pattern"]

        recovered_trigger = (
            mask * pattern
        )[0]

        mask_visual = (
            mask[0]
            .repeat(3, 1, 1)
        )

        out_dir = (
            os.path.dirname(args.out)
            or "."
        )

        save_image(
            recovered_trigger,
            os.path.join(
                out_dir,
                "recovered_trigger.png"
            )
        )

        save_image(
            mask_visual,
            os.path.join(
                out_dir,
                "recovered_trigger_mask.png"
            )
        )

        torch.save(
            {
                "mask": mask,
                "pattern": pattern,
                "target_class": suspect_class
            },
            os.path.join(
                out_dir,
                "recovered_trigger.pt"
            )
        )

        print(
            "Saved recovered trigger image to "
            "recovered_trigger.png "
            f"(class {suspect_class}, "
            f"mask covers "
            f"{suspect['mask_size']:.1%} "
            "of the image)"
        )

        print(
            "Saved raw mask/pattern tensors to "
            "recovered_trigger.pt"
        )

    else:

        print(
            "No class was flagged — "
            "no trigger image to save."
        )