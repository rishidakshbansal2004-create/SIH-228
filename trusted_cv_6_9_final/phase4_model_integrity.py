"""
Phase 4 — Model Integrity: unified entry point.

Wires together everything built separately so far:
    - STRIP            (strip_cifar.py)     — black-box, input-level
    - Neural Cleanse    (neural_cleanse.py)  — white-box, model-level, Stage 1
    - Ablation validation (ablation_stage2.py) — white-box, Stage 2, only
      runs if Stage 1 actually flagged a class

Output shape matches Phase 8's governance schema exactly (see
phase8_integrity.py in the team repo): a top-level disposition, a `flags`
list with one entry per check (reason/evidence/confidence/disposition),
an `access_level`, and a `limitations` list — so this plugs straight into
the same ledger (Phase 9) without any schema translation.

Routing logic (see design discussion — matches the PS's graceful-
degradation requirement):
    access_level == "white_box"  (weights available):
        run Neural Cleanse (Stage 1). If it flags a class, ALSO run
        Stage 2 ablation on that class as supplementary evidence.
        STRIP is also run as a cheap secondary signal (optional).
    access_level == "black_box"  (no weights, forward-pass only):
        run STRIP only. Report explicitly states Neural Cleanse and
        ablation were UNAVAILABLE and why.

TODO (tracked in memory, not yet done): reference/calibration images are
currently pulled from CIFAR-10 (correct for this testbed). For a real
model, swap `_load_calibration_batch` / `_load_cifar_pool` for a loader
that reads user-supplied clean reference images from the UI instead.
"""

import json
import os

import torch

from poison_and_train import SmallCNN
import neural_cleanse as nc
import strip_cifar as strip_mod
import ablation_stage2 as ablation_mod

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _to_flag(sub_report, check_name):
    """Convert one sub-module's report into a Phase-8-style flag entry."""
    disposition = sub_report.get("disposition", "review")
    # Normalize sub-module dispositions (which include extra states like
    # "inconclusive_pathway" / "confirmed_localized_group") into the
    # accept/review/quarantine vocabulary Phase 8 uses, while keeping the
    # original value visible in evidence for anyone who wants the detail.
    normalized = {
        "accept": "accept",
        "review": "review",
        "quarantine": "quarantine",
        "confirmed_localized": "quarantine",
        "confirmed_localized_group": "quarantine",
        "inconclusive_pathway": "review",
    }.get(disposition, "review")

    return {
        "check": check_name,
        "disposition": normalized,
        "raw_disposition": disposition,
        "confidence": sub_report.get("confidence"),
        "reason": sub_report.get("reason"),
        "evidence": sub_report.get("evidence"),
        "limitations": sub_report.get("limitations", []),
    }


def run_phase4(weights_path, access_level, calib_dir=None, n_calib=300, n_test=200,
               run_strip_in_white_box=True, device=DEVICE):
    """Full Phase 4 entry point.

    weights_path: path to the model's state_dict (SmallCNN architecture in
        this testbed; swap the loader for a real model's format later).
    access_level: "white_box" or "black_box" — decided upstream by the UI
        based on what the submitter actually provided (see memory: weights
        file = white-box by definition, API-only = black-box).
    calib_dir: reserved for the future user-image loader (TODO above);
        currently unused, CIFAR-10 is used internally by each sub-module.
    """
    model = SmallCNN(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    flags = []
    limitations = set()

    if access_level == "black_box":
        # Only STRIP is possible without gradient access.
        reference_pool = strip_mod._load_cifar_pool(n_images=300, seed=0)
        calib_images = strip_mod._load_cifar_pool(n_images=100, seed=1)
        test_images = strip_mod._load_cifar_pool(n_images=n_test, seed=2)
        threshold, _ = strip_mod.calibrate_threshold(model, calib_images, reference_pool, device=device)
        strip_report, _ = strip_mod.evaluate_batch(model, test_images, reference_pool, threshold,
                                                     device=device, label="phase4_black_box_test_batch")
        flags.append(_to_flag(strip_report, "strip_black_box"))
        limitations.update(strip_report.get("limitations", []))
        limitations.add(
            "Only black-box access was available — Neural Cleanse and the Stage 2 ablation "
            "validation could NOT run (both require gradient/weight access). This assessment "
            "relies solely on STRIP's input-level entropy signal, which cannot rule out a "
            "backdoor that none of the tested inputs happened to trigger."
        )

    elif access_level == "white_box":
        # Stage 1: Neural Cleanse (the primary white-box detector)
        nc_report, nc_per_class = nc.run_neural_cleanse(model, device=device)
        flags.append(_to_flag(nc_report, "neural_cleanse_white_box"))
        limitations.update(nc_report.get("limitations", []))

        # Stage 2: only run ablation if Stage 1 actually flagged a class —
        # otherwise there's no candidate trigger to validate against.
        suspect_class = nc_report["evidence"].get("suspect_class")
        if suspect_class is not None:
            suspect = next(r for r in nc_per_class if r["target_class"] == suspect_class)
            mask, pattern = suspect["mask"], suspect["pattern"]
            calib_images = nc._load_calibration_batch(n_images=64, batch_seed=3)
            triggered_images = ablation_mod._make_triggered_batch(mask, pattern, calib_images)
            ablation_report = ablation_mod.run_stage2(model, triggered_images, suspect_class, calib_images,
                                                       trigger_label="recovered", device=device)
            flags.append(_to_flag(ablation_report, "ablation_validation_stage2"))
            limitations.update(ablation_report.get("limitations", []))
        else:
            limitations.add(
                "Stage 2 ablation validation was not run — Neural Cleanse did not flag any "
                "class as anomalous, so there was no candidate trigger to validate against."
            )

        # Optional secondary signal: STRIP, even though we have white-box access.
        if run_strip_in_white_box:
            reference_pool = strip_mod._load_cifar_pool(n_images=300, seed=0)
            calib_images_strip = strip_mod._load_cifar_pool(n_images=100, seed=1)
            test_images = strip_mod._load_cifar_pool(n_images=n_test, seed=2)
            threshold, _ = strip_mod.calibrate_threshold(model, calib_images_strip, reference_pool, device=device)
            strip_report, _ = strip_mod.evaluate_batch(model, test_images, reference_pool, threshold,
                                                         device=device, label="phase4_white_box_secondary_strip")
            flags.append(_to_flag(strip_report, "strip_secondary_signal"))
            limitations.update(strip_report.get("limitations", []))
    else:
        raise ValueError(f"access_level must be 'white_box' or 'black_box', got {access_level!r}")

    # Overall disposition: worst flag wins (quarantine > review > accept),
    # same rule Phase 8 uses.
    if any(f["disposition"] == "quarantine" for f in flags):
        overall_disposition = "quarantine"
    elif any(f["disposition"] == "review" for f in flags):
        overall_disposition = "review"
    else:
        overall_disposition = "accept"

    report = {
        "phase": "phase4_model_integrity",
        "access_level": access_level,
        "disposition": overall_disposition,
        "flags": flags,
        "limitations": sorted(limitations),
    }
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--access_level", choices=["white_box", "black_box"], default="white_box")
    p.add_argument("--out", default="phase4_report.json")
    args = p.parse_args()

    report = run_phase4(args.weights, args.access_level)
    print("\n=== Phase 4 — Model Integrity report ===")
    print(json.dumps(report, indent=2, default=str))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to {args.out}")
