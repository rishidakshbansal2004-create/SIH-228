"""
Phase 4 — Model Integrity: unified entry point.

Combines:

- STRIP
- Neural Cleanse
- Stage 2 ablation validation

Clean reference images must be supplied by the user.
No dataset is downloaded automatically.
"""

import json

import torch

from poison_and_train import SmallCNN
import neural_cleanse as nc
import strip_cifar as strip_mod
import ablation_stage2 as ablation_mod


DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def _to_flag(sub_report, check_name):
    """
    Convert one sub-module report into a Phase-8-style flag entry.
    """

    disposition = sub_report.get(
        "disposition",
        "review"
    )

    normalized = {
        "accept": "accept",
        "review": "review",
        "quarantine": "quarantine",
        "confirmed_localized": "quarantine",
        "confirmed_localized_group": "quarantine",
        "inconclusive_pathway": "review",
    }.get(
        disposition,
        "review"
    )

    return {
        "check": check_name,
        "disposition": normalized,
        "raw_disposition": disposition,
        "confidence": sub_report.get(
            "confidence"
        ),
        "reason": sub_report.get(
            "reason"
        ),
        "evidence": sub_report.get(
            "evidence"
        ),
        "limitations": sub_report.get(
            "limitations",
            []
        )
    }


def _validate_reference_images(reference_images):
    if reference_images is None:
        raise ValueError(
            "Clean reference images are required for Phase 4. "
            "Please upload a clean reference dataset."
        )

    if not isinstance(
        reference_images,
        torch.Tensor
    ):
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
            "The current SmallCNN testbed requires RGB "
            "reference images."
        )

    reference_images = (
        reference_images
        .float()
    )

    if reference_images.max().item() > 1.0:
        reference_images = (
            reference_images / 255.0
        )

    reference_images = torch.clamp(
        reference_images,
        0.0,
        1.0
    )

    return reference_images


def run_phase4(
    weights_path,
    access_level,
    reference_images=None,
    n_calib=512,
    n_test=200,
    run_strip_in_white_box=True,
    device=DEVICE,
    reference_data=None
):
    """
    Full Phase 4 entry point.

    weights_path:
        Path to the SmallCNN state_dict.

    access_level:
        "white_box" or "black_box".

    reference_images:
        Clean images supplied by the user.
        Expected shape [N,3,H,W] and values [0,1].

    n_calib:
        Number of clean images used by Neural Cleanse
        for optimization.

    n_test:
        Number of clean images used for STRIP testing.
    """

    reference_images = _validate_reference_images(
        reference_images
    )

    required_images = max(
        n_calib * 2,
        400 + n_test
    )

    if len(reference_images) < required_images:
        raise ValueError(
            f"Phase 4 requires at least "
            f"{required_images} clean reference images, "
            f"but only {len(reference_images)} were supplied."
        )

    model = SmallCNN(
        num_classes=10
    )

    model.load_state_dict(
        torch.load(
            weights_path,
            map_location=device
        )
    )

    model.to(device)

    flags = []
    limitations = set()

    if access_level == "black_box":

        reference_pool = (
            reference_images[:300]
        )

        calib_images = (
            reference_images[300:400]
        )

        test_images = (
            reference_images[
                400:400 + n_test
            ]
        )

        threshold, _ = (
            strip_mod.calibrate_threshold(
                model,
                calib_images,
                reference_pool,
                device=device
            )
        )

        strip_report, _ = (
            strip_mod.evaluate_batch(
                model,
                test_images,
                reference_pool,
                threshold,
                device=device,
                label="phase4_black_box_test_batch"
            )
        )

        flags.append(
            _to_flag(
                strip_report,
                "strip_black_box"
            )
        )

        limitations.update(
            strip_report.get(
                "limitations",
                []
            )
        )

        limitations.add(
            "Only black-box access was available. "
            "Neural Cleanse and Stage 2 ablation require "
            "gradient/weight access."
        )

    elif access_level == "white_box":

        print(
            "[Phase 4] Starting Neural Cleanse "
            "with user-supplied clean reference data...",
            flush=True
        )

        nc_report, nc_per_class = (
            nc.run_neural_cleanse(
                model,
                reference_images=reference_images,
                n_calib=n_calib,
                n_holdout=n_calib,
                device=device
            )
        )

        flags.append(
            _to_flag(
                nc_report,
                "neural_cleanse_white_box"
            )
        )

        limitations.update(
            nc_report.get(
                "limitations",
                []
            )
        )

        suspect_class = (
            nc_report[
                "evidence"
            ].get(
                "suspect_class"
            )
        )

        if suspect_class is not None:

            suspect = next(
                r for r in nc_per_class
                if r["target_class"]
                == suspect_class
            )

            mask = suspect["mask"]
            pattern = suspect["pattern"]

            calib_images = (
                reference_images[:64]
            )

            triggered_images = (
                ablation_mod._make_triggered_batch(
                    mask,
                    pattern,
                    calib_images
                )
            )

            ablation_report = (
                ablation_mod.run_stage2(
                    model,
                    triggered_images,
                    suspect_class,
                    calib_images,
                    trigger_label="recovered",
                    device=device
                )
            )

            flags.append(
                _to_flag(
                    ablation_report,
                    "ablation_validation_stage2"
                )
            )

            limitations.update(
                ablation_report.get(
                    "limitations",
                    []
                )
            )

        else:

            limitations.add(
                "Stage 2 ablation validation was not run "
                "because Neural Cleanse did not flag a "
                "candidate target class."
            )

        if run_strip_in_white_box:

            reference_pool = (
                reference_images[:300]
            )

            calib_images_strip = (
                reference_images[300:400]
            )

            test_images = (
                reference_images[
                    400:400 + n_test
                ]
            )

            print(
                "[Phase 4] Running secondary STRIP signal...",
                flush=True
            )

            threshold, _ = (
                strip_mod.calibrate_threshold(
                    model,
                    calib_images_strip,
                    reference_pool,
                    device=device
                )
            )

            strip_report, _ = (
                strip_mod.evaluate_batch(
                    model,
                    test_images,
                    reference_pool,
                    threshold,
                    device=device,
                    label="phase4_white_box_secondary_strip"
                )
            )

            flags.append(
                _to_flag(
                    strip_report,
                    "strip_secondary_signal"
                )
            )

            limitations.update(
                strip_report.get(
                    "limitations",
                    []
                )
            )

    else:

        raise ValueError(
            "access_level must be "
            "'white_box' or 'black_box', "
            f"got {access_level!r}"
        )

    if any(
        f["disposition"] == "quarantine"
        for f in flags
    ):
        overall_disposition = "quarantine"

    elif any(
        f["disposition"] == "review"
        for f in flags
    ):
        overall_disposition = "review"

    else:
        overall_disposition = "accept"

    report = {
        "phase": "phase4_model_integrity",
        "access_level": access_level,
        "disposition": overall_disposition,
        "flags": flags,
        "reference_data": {
            **(reference_data or {}),
            "source": "user_supplied",
            "total_images": len(reference_images),
            "calibration_images": n_calib,
            "strip_test_images": n_test
        },
        "limitations": sorted(
            limitations
        )
    }

    return report


if __name__ == "__main__":

    import argparse
    from pathlib import Path
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True
    )

    parser.add_argument(
        "--access_level",
        choices=[
            "white_box",
            "black_box"
        ],
        default="white_box"
    )

    parser.add_argument(
        "--reference_dir",
        required=True,
        help="Directory containing clean reference images"
    )

    parser.add_argument(
        "--out",
        default="phase4_report.json"
    )

    args = parser.parse_args()

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
            "No supported reference images found."
        )

    images = []

    for path in image_paths:

        image = Image.open(
            path
        ).convert("RGB")

        image = image.resize(
            (32, 32)
        )

        array = (
            np.asarray(
                image,
                dtype=np.float32
            ) / 255.0
        )

        tensor = (
            torch.from_numpy(
                array
            )
            .permute(2, 0, 1)
        )

        images.append(
            tensor
        )

    reference_images = torch.stack(
        images
    )

    report = run_phase4(
        args.weights,
        args.access_level,
        reference_images=reference_images
    )

    print(
        "\n=== Phase 4 — Model Integrity report ==="
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