"""
Phase 4 — Model Integrity, Black-Box: STRIP

Paper: "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks"
Gao, Xu, Wang, Chen, Ranasinghe, Nepal, ACSAC 2019.
https://arxiv.org/abs/1902.06531

Core idea: blend a candidate input with random clean reference images and
run it through the model many times. A clean input's prediction should
wobble under heavy blending (high entropy). An input carrying a backdoor
trigger tends to keep predicting the SAME class regardless of blending,
because the trigger dominates the decision (low entropy = suspicious).

Only needs forward-pass access to the model — no gradients, no weights
required. This is the path used when a vendor only exposes an API/binary
(no weights file at all), or as a second, cheaper signal alongside
Neural Cleanse when white-box access IS available.

Important limitation (stated explicitly in the report, not hidden):
STRIP only tests the inputs you actually feed it. If none of your test
images happen to contain the trigger, STRIP cannot conclude the model has
no backdoor at all — it can only say "these specific inputs looked clean."
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from poison_and_train import SmallCNN, add_trigger

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)
N_PERTURBATIONS = 40
BLEND_ALPHA = 0.6            # STRIP paper's typical operating range is 0.4-0.7
CALIBRATION_PERCENTILE = 1.0  # flag anything scoring below the 1st percentile of clean entropy


def _normalize(x):
    mean = torch.tensor(NORM_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _load_cifar_pool(n_images, seed):
    g = torch.Generator().manual_seed(seed)
    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    idx = torch.randperm(len(ds), generator=g)[:n_images]
    imgs = torch.stack([ds[i][0] for i in idx])
    return imgs  # raw [0,1] tensors


@torch.no_grad()
def strip_entropy(model, input_image, reference_pool, n_perturbations=N_PERTURBATIONS,
                   blend_alpha=BLEND_ALPHA, device=DEVICE):
    """Blend `input_image` with random images from `reference_pool`, run the
    model on each blend, and return the entropy of the AVERAGE predicted
    class distribution across all blends. Low entropy = suspicious."""
    model.eval()
    input_image = input_image.to(device)
    reference_pool = reference_pool.to(device)

    idx = np.random.choice(reference_pool.shape[0], size=n_perturbations, replace=True)
    refs = reference_pool[idx]

    blended = blend_alpha * refs + (1 - blend_alpha) * input_image.unsqueeze(0)
    logits = model(_normalize(blended))
    probs = F.softmax(logits, dim=1)

    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
    preds = probs.argmax(dim=1).tolist()
    return entropy, preds


def calibrate_threshold(model, calib_images, reference_pool, percentile=CALIBRATION_PERCENTILE, device=DEVICE):
    """Run STRIP on KNOWN CLEAN images to find a sensible entropy cutoff.
    Anything scoring below this on new inputs gets flagged."""
    entropies = []
    for img in calib_images:
        e, _ = strip_entropy(model, img, reference_pool, device=device)
        entropies.append(e)
    threshold = float(np.percentile(entropies, percentile))
    return threshold, entropies


def evaluate_batch(model, images, reference_pool, threshold, device=DEVICE, label=""):
    """Run STRIP over a batch of test images and return a governance-style
    report (matches Neural Cleanse's / Phase 8's shape)."""
    entropies = []
    flagged_count = 0
    per_image = []
    for i, img in enumerate(images):
        e, preds = strip_entropy(model, img, reference_pool, device=device)
        flagged = e < threshold
        entropies.append(e)
        if flagged:
            flagged_count += 1
        per_image.append({"index": i, "entropy": e, "flagged": flagged})

    flag_rate = flagged_count / max(len(images), 1)
    verdict_flagged = flag_rate > 0.10  # more than 10% of this batch look suspicious

    report = {
        "method": "STRIP (black-box, forward-pass only)",
        "access_level": "black_box",
        "batch_label": label,
        "disposition": "quarantine" if verdict_flagged else "accept",
        "reason": (
            f"{flagged_count}/{len(images)} tested inputs ({flag_rate:.1%}) showed unusually "
            f"low prediction entropy under perturbation (< {threshold:.3f}), consistent with a "
            f"trigger dominating the model's decision on this input set."
            if verdict_flagged else
            f"Only {flagged_count}/{len(images)} tested inputs ({flag_rate:.1%}) fell below the "
            f"clean-calibrated entropy threshold — consistent with normal behavior on this input set."
        ),
        "evidence": {
            "threshold": threshold,
            "mean_entropy": float(np.mean(entropies)),
            "min_entropy": float(np.min(entropies)),
            "flagged_count": flagged_count,
            "total": len(images),
            "flag_rate": flag_rate,
        },
        "confidence": float(min(flag_rate / 0.5, 1.0)) if verdict_flagged else float(1.0 - flag_rate),
        "limitations": [
            "Only evaluates the specific inputs tested — cannot conclude the model has no "
            "backdoor if none of the tested inputs happened to contain a trigger.",
            "Assumes the trigger, if present, dominates the prediction strongly enough to "
            "survive blending — may miss weak or low-opacity triggers.",
            "Entropy threshold is calibrated on this team's own clean reference images, not "
            "the vendor's data — distribution mismatch could shift false-positive/negative rates.",
        ],
    }
    return report, per_image


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to a SmallCNN state_dict, e.g. backdoor_testbed/backdoored_model.pt")
    p.add_argument("--out", default="strip_report.json")
    p.add_argument("--n_test", type=int, default=200)
    args = p.parse_args()

    model = SmallCNN(num_classes=10)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)

    reference_pool = _load_cifar_pool(n_images=300, seed=0)
    calib_images = _load_cifar_pool(n_images=100, seed=1)
    test_clean = _load_cifar_pool(n_images=args.n_test, seed=2)

    print("Calibrating STRIP threshold on clean images...")
    threshold, calib_entropies = calibrate_threshold(model, calib_images, reference_pool)
    print(f"Threshold: {threshold:.4f}  (clean calibration entropy range: "
          f"{min(calib_entropies):.3f} - {max(calib_entropies):.3f})")

    # Case A: fresh CLEAN test images, no trigger. STRIP should mostly clear these.
    print("\n=== Testing CLEAN images (no trigger) ===")
    clean_report, _ = evaluate_batch(model, test_clean, reference_pool, threshold, label="clean_test_images")
    print(json.dumps(clean_report, indent=2))

    # Case B: the SAME images, but with the trigger stamped on — simulates
    # "a suspicious input carrying the trigger arrives." STRIP should flag
    # most of these if the model is genuinely backdoored.
    print("\n=== Testing TRIGGERED images (same images + trigger patch) ===")
    test_triggered = torch.stack([add_trigger(img) for img in test_clean])
    triggered_report, _ = evaluate_batch(model, test_triggered, reference_pool, threshold, label="triggered_test_images")
    print(json.dumps(triggered_report, indent=2))

    with open(args.out, "w") as f:
        json.dump({"clean": clean_report, "triggered": triggered_report}, f, indent=2)
    print(f"\nSaved combined report to {args.out}")
