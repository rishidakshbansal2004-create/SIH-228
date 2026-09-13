"""
Phase 4 — Model Integrity, White-Box Stage 1: Neural Cleanse

Paper: "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in
Neural Networks" — Wang, Yao, Shan, Li, Viswanath, Zheng, Zhao,
IEEE S&P 2019. https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf

What this does, per class c:
    1. Learn a mask (where to overwrite) and a pattern (what to overwrite
       with) that reliably forces the model to predict class c on a batch
       of CLEAN, mixed-class images — while minimizing how much of the
       image needs to be overwritten (L1 penalty on the mask).
    2. Record the final mask size (L1 norm) for class c.
After running this for every class:
    3. Compare mask sizes across all classes using Median Absolute
       Deviation (MAD). A class whose mask is an anomalously small outlier
       is flagged as the suspected backdoor target.
    4. Verify the flagged class's recovered mask+pattern by applying it to
       FRESH clean images (not used during optimization) and checking it
       reliably forces the target class — this is what separates "here's a
       candidate" from "here's a validated candidate."

This does NOT prove the recovered pattern is the attacker's exact trigger
— see limitations in the returned report.

Designed to plug into poison_and_train.py's SmallCNN + CIFAR-10 testbed,
but works on any classifier: swap NUM_CLASSES / IMG_SHAPE / the model
loader as needed.
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

# Reuse the exact architecture used to train the testbed models.
from poison_and_train import SmallCNN, add_trigger

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SHAPE = (3, 32, 32)          # CIFAR-10
NUM_CLASSES = 10
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)

# Optimization hyperparameters (paper-standard ballpark, tuned down for speed)
STEPS = 800
LR = 0.05
LAMBDA_INIT = 1e-4        # weight on the mask-size penalty; paper anneals this
LAMBDA_MAX = 5.0
ASR_TARGET = 0.90         # if a candidate reaches this attack success rate on
                           # the optimization batch, we consider it "converged"
MAD_ANOMALY_THRESHOLD = 2.0   # paper's standard cutoff
MASK_INIT_BIAS = -4.0     # sigmoid(-4) ≈ 0.018 → mask starts NEAR-EMPTY, not
                           # at 50% coverage. The optimizer must earn every
                           # bit of mask it uses; a real small trigger should
                           # converge to a small mask, not shrink from a big one.


def _load_calibration_batch(n_images=512, batch_seed=0):
    """Our own clean reference images — NOT the vendor's/attacker's training
    data. Mixed across all classes, matching the threat model: we never
    assume access to the model owner's original training set."""
    g = torch.Generator().manual_seed(batch_seed)
    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    idx = torch.randperm(len(ds), generator=g)[:n_images]
    imgs = torch.stack([ds[i][0] for i in idx])
    return imgs  # raw [0,1] tensors, NOT normalized yet


def _normalize(x):
    mean = torch.tensor(NORM_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def reverse_engineer_trigger(model, target_class, calib_imgs, steps=STEPS, lr=LR, device=DEVICE):
    """Optimize a (mask, pattern) pair that forces `target_class` on
    `calib_imgs`, while minimizing mask size. Returns the final mask,
    pattern, mask L1 size, and the attack success rate achieved on this
    same optimization batch (a training-set ASR, not held-out — see
    verify_trigger() for the held-out check).
    """
    model.eval()
    calib_imgs = calib_imgs.to(device)
    n, c, h, w = calib_imgs.shape

    # Mask and pattern are optimized in an unconstrained space and squashed
    # into [0,1] via sigmoid/tanh, matching Neural Cleanse's parameterization
    # (keeps gradient descent well-behaved instead of manually clipping).
    # Mask starts NEAR-EMPTY (see MASK_INIT_BIAS) so the optimizer must
    # genuinely grow it to whatever size is required — a real small trigger
    # should converge small; a class with no shortcut will be forced to grow
    # the mask large to satisfy the classification objective.
    mask_param = torch.full((1, 1, h, w), MASK_INIT_BIAS, device=device, requires_grad=True)
    pattern_param = torch.zeros(1, c, h, w, device=device, requires_grad=True)
    mask_param.requires_grad_(True)
    pattern_param.requires_grad_(True)

    opt = torch.optim.Adam([mask_param, pattern_param], lr=lr)
    lam = LAMBDA_INIT
    target = torch.full((n,), target_class, dtype=torch.long, device=device)

    for step in range(steps):
        mask = torch.sigmoid(mask_param)         # [1,1,H,W] in (0,1)
        pattern = torch.sigmoid(pattern_param)    # [1,C,H,W] in (0,1)

        blended = (1 - mask) * calib_imgs + mask * pattern
        logits = model(_normalize(blended))
        cls_loss = F.cross_entropy(logits, target)
        mask_size = mask.mean()  # normalized (0..1 fraction of image), easier to compare across classes than raw L1
        loss = cls_loss + lam * mask_size

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                asr = (logits.argmax(1) == target).float().mean().item()
            # Anneal lambda: if we're comfortably fooling the model, push
            # harder on shrinking the mask; if we're struggling, back off
            # so the optimizer can first find ANY working solution.
            if asr > ASR_TARGET:
                lam = min(lam * 1.5, LAMBDA_MAX)
            else:
                lam = max(lam * 0.8, LAMBDA_INIT)

    with torch.no_grad():
        mask = torch.sigmoid(mask_param)
        pattern = torch.sigmoid(pattern_param)
        blended = (1 - mask) * calib_imgs + mask * pattern
        logits = model(_normalize(blended))
        final_asr = (logits.argmax(1) == target).float().mean().item()
        mask_size = mask.mean().item()

    return {
        "target_class": target_class,
        "mask": mask.detach().cpu(),
        "pattern": pattern.detach().cpu(),
        "mask_size": mask_size,          # fraction of image overwritten, 0..1
        "optimization_asr": final_asr,   # ASR on the SAME batch used to optimize (optimistic)
    }


def mad_anomaly_scores(mask_sizes):
    """Standard MAD-based outlier score, per Neural Cleanse Section IV."""
    sizes = np.array(mask_sizes, dtype=float)
    median = np.median(sizes)
    mad = np.median(np.abs(sizes - median)) + 1e-12
    # Neural Cleanse flags classes whose mask is anomalously SMALL, so we
    # only care about negative deviations (below the median).
    anomaly = np.where(sizes < median, (median - sizes) / mad, 0.0)
    return anomaly, median, mad


@torch.no_grad()
def verify_trigger(model, mask, pattern, target_class, held_out_imgs, device=DEVICE):
    """Apply the recovered mask+pattern to FRESH images not used during
    optimization, and check how often it forces target_class. This is the
    "verification" step (Section 6 of the design doc) — separates a
    candidate trigger from a validated one."""
    model.eval()
    held_out_imgs = held_out_imgs.to(device)
    mask = mask.to(device)
    pattern = pattern.to(device)
    blended = (1 - mask) * held_out_imgs + mask * pattern
    logits = model(_normalize(blended))
    preds = logits.argmax(1)
    asr = (preds == target_class).float().mean().item()
    return asr


def run_neural_cleanse(model, num_classes=NUM_CLASSES, n_calib=512, n_holdout=512, device=DEVICE):
    """Full Stage 1 pipeline: reverse-engineer a candidate trigger for every
    class, score anomalies, verify the most suspicious class on held-out
    data. Returns a governance-ready report."""
    calib_imgs = _load_calibration_batch(n_images=n_calib, batch_seed=0)
    holdout_imgs = _load_calibration_batch(n_images=n_holdout, batch_seed=1)

    per_class = []
    for c in range(num_classes):
        print(f"[Neural Cleanse] optimizing candidate trigger for class {c}/{num_classes-1}...")
        result = reverse_engineer_trigger(model, c, calib_imgs, device=device)
        per_class.append(result)

    mask_sizes = [r["mask_size"] for r in per_class]
    anomaly_scores, median, mad = mad_anomaly_scores(mask_sizes)

    flags = []
    for r, score in zip(per_class, anomaly_scores):
        flags.append({
            "target_class": r["target_class"],
            "mask_size": r["mask_size"],
            "optimization_asr": r["optimization_asr"],
            "anomaly_index": float(score),
        })

    suspect_idx = int(np.argmax(anomaly_scores))
    suspect = per_class[suspect_idx]
    suspect_flagged = anomaly_scores[suspect_idx] > MAD_ANOMALY_THRESHOLD

    verification_asr = None
    if suspect_flagged:
        verification_asr = verify_trigger(
            model, suspect["mask"], suspect["pattern"], suspect["target_class"], holdout_imgs, device=device
        )

    verdict_flagged = bool(suspect_flagged and verification_asr is not None and verification_asr > 0.80)

    report = {
        "method": "Neural Cleanse (white-box, requires gradient access)",
        "access_level": "white_box",
        "disposition": "quarantine" if verdict_flagged else ("review" if suspect_flagged else "accept"),
        "reason": (
            f"Class {suspect['target_class']} required an anomalously small mask "
            f"(anomaly_index={anomaly_scores[suspect_idx]:.2f}, threshold={MAD_ANOMALY_THRESHOLD}) "
            f"to force misclassification, and the recovered trigger achieved "
            f"{verification_asr:.1%} attack success rate on held-out clean images "
            f"never used during optimization."
            if verdict_flagged else
            (f"Class {suspect['target_class']} showed a mild mask-size anomaly "
             f"(anomaly_index={anomaly_scores[suspect_idx]:.2f}) but did not clear "
             f"held-out verification — treat as inconclusive, not confirmed."
             if suspect_flagged else
             "No class showed an anomalously small trigger mask relative to the others.")
        ),
        "evidence": {
            "per_class_flags": flags,
            "median_mask_size": float(median),
            "mad": float(mad),
            "suspect_class": suspect["target_class"] if suspect_flagged else None,
            "verification_asr_on_holdout": verification_asr,
        },
        "confidence": float(min(anomaly_scores[suspect_idx] / (2 * MAD_ANOMALY_THRESHOLD), 1.0)) if suspect_flagged else 1.0 - float(min(max(anomaly_scores), 1.0) / MAD_ANOMALY_THRESHOLD),
        "limitations": [
            "Does not prove the recovered pattern is the attacker's exact trigger — it is a candidate reverse-engineered pattern that behaves like one.",
            "Requires white-box (gradient) access — has no fallback if only an API/black-box is available (use STRIP instead in that case).",
            "Assumes a patch-style trigger reasonably localized in input space; may miss distributed, semantic, or non-additive triggers.",
            "Optimization ASR is measured on the same batch used for optimization and is optimistic; only the held-out verification ASR should be trusted as evidence.",
        ],
    }
    return report, per_class


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to a SmallCNN state_dict, e.g. backdoor_testbed/backdoored_model.pt")
    p.add_argument("--out", default="neural_cleanse_report.json")
    args = p.parse_args()

    model = SmallCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)

    report, per_class = run_neural_cleanse(model)

    print("\n=== Neural Cleanse report ===")
    print(json.dumps(report, indent=2))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {args.out}")

    # Save the recovered trigger itself as a viewable image, not just numbers.
    # This is what you'd actually put in your report/demo as "here is the
    # pattern the model was reacting to" — see report["evidence"]["suspect_class"].
    suspect_class = report["evidence"]["suspect_class"]
    if suspect_class is not None:
        suspect = next(r for r in per_class if r["target_class"] == suspect_class)
        mask, pattern = suspect["mask"], suspect["pattern"]

        recovered_trigger = (mask * pattern)[0]           # what actually gets pasted, [C,H,W]
        mask_visual = mask[0].repeat(3, 1, 1)              # mask alone, as a grayscale-looking image

        out_dir = os.path.dirname(args.out) or "."
        save_image(recovered_trigger, os.path.join(out_dir, "recovered_trigger.png"))
        save_image(mask_visual, os.path.join(out_dir, "recovered_trigger_mask.png"))
        torch.save({"mask": mask, "pattern": pattern, "target_class": suspect_class},
                   os.path.join(out_dir, "recovered_trigger.pt"))
        print(f"Saved recovered trigger image to recovered_trigger.png "
              f"(class {suspect_class}, mask covers {suspect['mask_size']:.1%} of the image)")
        print("Saved raw mask/pattern tensors to recovered_trigger.pt for reuse in Stage 2 ablation.")
        print("Compare this visually against backdoor_testbed/trigger_examples.png "
              "(the REAL planted trigger) to sanity-check the reconstruction — "
              "they are not guaranteed to look identical, only to behave similarly.")
    else:
        print("No class was flagged — no trigger image to save.")