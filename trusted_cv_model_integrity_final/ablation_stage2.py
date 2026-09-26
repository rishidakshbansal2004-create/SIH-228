"""
Phase 4 — Model Integrity, White-Box Stage 2 (OUR OWN PROPOSED IDEA)

This is NOT the original Neural Cleanse paper's method, and is NOT
established literature reproduced exactly — it is our own experimental
validation layer, built on top of a Neural Cleanse candidate trigger.
See handoff doc sections 11 and 14 for why this distinction matters.

Question this stage answers:
    Given a candidate trigger already reverse-engineered and VERIFIED by
    Neural Cleanse (Stage 1), does the model's target-class prediction on
    triggered inputs depend disproportionately on a small number of
    internal channels/pathways, compared to how clean predictions depend
    on the network?

Method (Phase A/B/C from the design doc):
    A. Systematically ablate (zero out) each convolutional channel, one at
       a time, and record how much the target-class probability drops for
       TRIGGERED inputs vs. how much the top-class probability drops for
       CLEAN inputs.
    B. Rank channels by influence (ΔP) on triggered inputs specifically,
       controlling for channels that are just generally important (i.e.
       also influential on clean predictions — those are not "trigger
       pathways", they're just important channels).
    C. Targeted ablation: zero out the top influential "trigger-specific"
       channels together and confirm the effect concentrates there,
       rather than being explainable by removing any random channels.

Honest framing (do not overclaim):
    - This does NOT prove those channels are "the backdoor" in some
      mechanistic/causal sense beyond correlation under ablation.
    - Finding NO concentrated pathway does not mean no backdoor — Stage 1's
      finding stands on its own; a distributed/redundant backdoor is a
      valid (if less clean) outcome, and should be reported as such, not
      as a failure of detection.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from poison_and_train import SmallCNN, add_trigger

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)

# A channel is "trigger-specific" if ablating it hurts the triggered
# prediction much more than it hurts clean predictions, by at least this
# much (in probability points). Purely a starting heuristic — tune based on
# what you observe, and say so in your report rather than presenting it as
# a validated cutoff.
TRIGGER_SPECIFICITY_MARGIN = 0.15
# A pathway is considered "concentrated" (Phase C passes) if removing just
# the top-k flagged channels collapses target probability by at least this
# much, while the same removal barely touches clean predictions.
CONCENTRATION_COLLAPSE_THRESHOLD = 0.50
CONCENTRATION_CLEAN_TOLERANCE = 0.15
TOP_K_CHANNELS = 5


def _normalize(x):
    mean = torch.tensor(NORM_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class ChannelAblator:
    """Registers a forward hook on a named conv layer that can zero out a
    specific output channel (or none, when `active_channel` is None)."""

    def __init__(self, model, layer_name):
        self.model = model
        self.layer = dict(model.named_modules())[layer_name]
        self.active_channel = None
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if self.active_channel is not None:
            out = out.clone()
            out[:, self.active_channel, :, :] = 0.0
            return out
        return out

    def set_channel(self, idx):
        self.active_channel = idx

    def clear(self):
        self.active_channel = None

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def _target_prob(model, images, target_class, device=DEVICE):
    """Mean predicted probability of target_class across a batch."""
    images = images.to(device)
    logits = model(_normalize(images))
    probs = F.softmax(logits, dim=1)
    return probs[:, target_class].mean().item()


@torch.no_grad()
def _top_class_prob(model, images, device=DEVICE):
    """Mean probability the model assigns to EACH image's own top-1 class
    (i.e. how confident is the model in whatever it currently predicts).
    Used as the 'clean prediction strength' baseline, since clean images
    don't share one target class the way triggered images do."""
    images = images.to(device)
    logits = model(_normalize(images))
    probs = F.softmax(logits, dim=1)
    top = probs.max(dim=1).values
    return top.mean().item()


def _make_triggered_batch(mask, pattern, clean_images):
    mask = mask.to(clean_images.device)
    pattern = pattern.to(clean_images.device)
    return (1 - mask) * clean_images + mask * pattern


def _collect_all_channels(model, layer_names):
    """Flat list of (layer_name, channel_idx) across all given layers."""
    all_channels = []
    for layer_name in layer_names:
        layer = dict(model.named_modules())[layer_name]
        for ch in range(layer.out_channels):
            all_channels.append((layer_name, ch))
    return all_channels


@torch.no_grad()
def _eval_with_group_ablated(model, triggered_images, clean_images, target_class, group, device=DEVICE):
    """Zero out an arbitrary set of (layer_name, channel_idx) pairs
    SIMULTANEOUSLY and return (triggered_target_prob, clean_top_prob)."""
    by_layer = {}
    for layer_name, ch in group:
        by_layer.setdefault(layer_name, []).append(ch)

    handles = []
    for layer_name, channels in by_layer.items():
        layer = dict(model.named_modules())[layer_name]

        def make_hook(channels_to_zero):
            def hook(module, inp, out):
                out = out.clone()
                for c in channels_to_zero:
                    out[:, c, :, :] = 0.0
                return out
            return hook

        handles.append(layer.register_forward_hook(make_hook(channels)))

    triggered_p = _target_prob(model, triggered_images, target_class, device)
    clean_p = _top_class_prob(model, clean_images, device)

    for h in handles:
        h.remove()
    return triggered_p, clean_p


def greedy_group_ablation(model, triggered_images, clean_images, target_class,
                           layer_names=("conv1", "conv2", "conv3"),
                           max_group_size=15, device=DEVICE):
    """Answers: 'not single channel, but BATCH-wise' — does a coordinated
    group of channels, chosen together, collapse the triggered prediction
    even though no single channel did? Greedy forward selection: at each
    step, add whichever remaining channel most increases
    (triggered_prob_drop - clean_prob_drop) given the CURRENT group already
    removed — this captures interaction effects single-channel sweeps miss.
    """
    model.to(device).eval()
    triggered_images = triggered_images.to(device)
    clean_images = clean_images.to(device)

    baseline_triggered, baseline_clean = _eval_with_group_ablated(
        model, triggered_images, clean_images, target_class, group=[], device=device)

    candidates = _collect_all_channels(model, layer_names)
    current_group = []
    trajectory = []

    for step in range(max_group_size):
        best_candidate, best_score, best_result = None, -1e9, None
        # NOTE: evaluating every remaining candidate each step is O(n^2) in
        # channel count — fine for this small model (160 channels), would
        # need subsampling candidates for a much larger network.
        for cand in candidates:
            if cand in current_group:
                continue
            trial_group = current_group + [cand]
            t_p, c_p = _eval_with_group_ablated(model, triggered_images, clean_images, target_class, trial_group, device=device)
            score = (baseline_triggered - t_p) - (baseline_clean - c_p)
            if score > best_score:
                best_candidate, best_score, best_result = cand, score, (t_p, c_p)

        current_group.append(best_candidate)
        trajectory.append({
            "step": step + 1,
            "added": {"layer": best_candidate[0], "channel": best_candidate[1]},
            "group_size": len(current_group),
            "triggered_prob": best_result[0],
            "clean_prob": best_result[1],
            "triggered_collapse_from_baseline": baseline_triggered - best_result[0],
            "clean_disruption_from_baseline": baseline_clean - best_result[1],
        })

    return {
        "baseline_triggered_prob": baseline_triggered,
        "baseline_clean_prob": baseline_clean,
        "trajectory": trajectory,
        "final_group": [{"layer": l, "channel": c} for l, c in current_group],
    }


def random_subset_baseline(model, triggered_images, clean_images, target_class,
                            layer_names=("conv1", "conv2", "conv3"),
                            group_size=15, n_trials=20, device=DEVICE, seed=0):
    """Calibration check: how much does a RANDOM group of the same size
    collapse the triggered prediction, on average? If the greedy group's
    collapse is far beyond what random groups achieve, that's evidence the
    greedy group is meaningfully special, not just 'removing enough stuff
    breaks anything'."""
    model.to(device).eval()
    triggered_images = triggered_images.to(device)
    clean_images = clean_images.to(device)
    candidates = _collect_all_channels(model, layer_names)

    rng = np.random.RandomState(seed)
    triggered_collapses, clean_disruptions = [], []
    baseline_triggered, baseline_clean = _eval_with_group_ablated(
        model, triggered_images, clean_images, target_class, group=[], device=device)

    for _ in range(n_trials):
        idx = rng.choice(len(candidates), size=group_size, replace=False)
        group = [candidates[i] for i in idx]
        t_p, c_p = _eval_with_group_ablated(model, triggered_images, clean_images, target_class, group, device=device)
        triggered_collapses.append(baseline_triggered - t_p)
        clean_disruptions.append(baseline_clean - c_p)

    return {
        "n_trials": n_trials,
        "group_size": group_size,
        "mean_triggered_collapse": float(np.mean(triggered_collapses)),
        "std_triggered_collapse": float(np.std(triggered_collapses)),
        "max_triggered_collapse": float(np.max(triggered_collapses)),
        "mean_clean_disruption": float(np.mean(clean_disruptions)),
    }


def run_ablation_study(model, triggered_images, target_class, clean_images,
                        layer_names=("conv1", "conv2", "conv3"), device=DEVICE):
    """Phase A + B: sweep every channel in the given layers, one at a time,
    recording its effect on triggered-target-probability vs clean-top-
    probability. Returns per-channel records plus the ranked "trigger-
    specific" channel list.

    triggered_images: a pre-built batch of already-triggered inputs — can
    come from a Neural Cleanse recovered mask/pattern blend, OR from the
    real ground-truth trigger (e.g. add_trigger() in a testbed setting
    where the true trigger is known), so results can be compared."""
    model.to(device).eval()
    triggered_images = triggered_images.to(device)
    clean_images = clean_images.to(device)

    baseline_target_p = _target_prob(model, triggered_images, target_class, device)
    baseline_clean_p = _top_class_prob(model, clean_images, device)
    print(f"Baseline: P(target={target_class} | triggered)={baseline_target_p:.4f}  "
          f"P(top-class | clean)={baseline_clean_p:.4f}")

    records = []
    for layer_name in layer_names:
        ablator = ChannelAblator(model, layer_name)
        n_channels = ablator.layer.out_channels
        print(f"[Ablation] sweeping {n_channels} channels in {layer_name}...")
        for ch in range(n_channels):
            ablator.set_channel(ch)
            triggered_p = _target_prob(model, triggered_images, target_class, device)
            clean_p = _top_class_prob(model, clean_images, device)
            ablator.clear()

            delta_triggered = baseline_target_p - triggered_p   # how much removing this channel HURT the trigger
            delta_clean = baseline_clean_p - clean_p             # how much it hurt normal predictions
            records.append({
                "layer": layer_name, "channel": ch,
                "delta_triggered_prob": delta_triggered,
                "delta_clean_prob": delta_clean,
                "specificity": delta_triggered - delta_clean,   # high = hurts trigger much more than clean
            })
        ablator.remove()

    records.sort(key=lambda r: r["specificity"], reverse=True)
    trigger_specific = [r for r in records if r["specificity"] > TRIGGER_SPECIFICITY_MARGIN]
    return {
        "baseline_target_prob": baseline_target_p,
        "baseline_clean_prob": baseline_clean_p,
        "all_channel_records": records,
        "trigger_specific_channels": trigger_specific[:TOP_K_CHANNELS],
        "top_10_by_specificity_regardless_of_threshold": records[:10],  # always visible, even if none crossed the cutoff
    }


@torch.no_grad()
def targeted_ablation_test(model, triggered_images, target_class, clean_images,
                            top_channels, device=DEVICE):
    """Phase C: zero out ALL top-k flagged channels simultaneously and see
    whether the effect concentrates as expected — target prob should
    collapse for triggered inputs while clean predictions stay relatively
    intact."""
    model.to(device).eval()
    triggered_images = triggered_images.to(device)
    clean_images = clean_images.to(device)

    ablators = []
    by_layer = {}
    for rec in top_channels:
        by_layer.setdefault(rec["layer"], []).append(rec["channel"])

    handles = []
    for layer_name, channels in by_layer.items():
        layer = dict(model.named_modules())[layer_name]

        def make_hook(channels_to_zero):
            def hook(module, inp, out):
                out = out.clone()
                for c in channels_to_zero:
                    out[:, c, :, :] = 0.0
                return out
            return hook

        handles.append(layer.register_forward_hook(make_hook(channels)))

    triggered_p_after = _target_prob(model, triggered_images, target_class, device)
    clean_p_after = _top_class_prob(model, clean_images, device)

    for h in handles:
        h.remove()

    return {"triggered_target_prob_after": triggered_p_after, "clean_top_prob_after": clean_p_after}


def run_stage2(model, triggered_images, target_class, calib_images, trigger_label="recovered",
               run_group_search=True, device=DEVICE):
    """Full Stage 2 pipeline for ONE trigger source (recovered or real).
    Call this twice — once per trigger source — to compare them."""
    sweep = run_ablation_study(model, triggered_images, target_class, calib_images, device=device)
    top_channels = sweep["trigger_specific_channels"]

    concentrated = False
    targeted_result = None
    if top_channels:
        targeted_result = targeted_ablation_test(model, triggered_images, target_class, calib_images, top_channels, device=device)
        triggered_collapse = sweep["baseline_target_prob"] - targeted_result["triggered_target_prob_after"]
        clean_disruption = sweep["baseline_clean_prob"] - targeted_result["clean_top_prob_after"]
        concentrated = (triggered_collapse > CONCENTRATION_COLLAPSE_THRESHOLD and
                         clean_disruption < CONCENTRATION_CLEAN_TOLERANCE)

    # Batch/group-wise search: even if no SINGLE channel stood out, a
    # coordinated small group might. This directly tests the "distributed,
    # not single-channel" hypothesis, with a random-group baseline for
    # calibration so a raw collapse number isn't over-interpreted.
    group_result, random_baseline = None, None
    group_concentrated = False
    if run_group_search:
        print(f"[Ablation] running greedy group search (trigger_source={trigger_label})...")
        group_result = greedy_group_ablation(model, triggered_images, calib_images, target_class, device=device)
        random_baseline = random_subset_baseline(model, triggered_images, calib_images, target_class,
                                                   group_size=group_result["trajectory"][-1]["group_size"], device=device)
        final_step = group_result["trajectory"][-1]
        greedy_collapse = final_step["triggered_collapse_from_baseline"]
        greedy_clean_disruption = final_step["clean_disruption_from_baseline"]
        # "Meaningfully special" = greedy collapse clears both an absolute
        # bar AND beats the random-group baseline by a healthy margin,
        # while not trashing clean predictions.
        group_concentrated = (
            greedy_collapse > CONCENTRATION_COLLAPSE_THRESHOLD and
            greedy_clean_disruption < CONCENTRATION_CLEAN_TOLERANCE and
            greedy_collapse > random_baseline["mean_triggered_collapse"] + 2 * random_baseline["std_triggered_collapse"]
        )

    if group_concentrated:
        disposition = "confirmed_localized_group"
        reason = (
            f"No single channel was individually decisive, but a coordinated greedy-selected "
            f"group of {group_result['trajectory'][-1]['group_size']} channels collapsed the "
            f"triggered target probability by {group_result['trajectory'][-1]['triggered_collapse_from_baseline']:.3f} "
            f"(vs a random group of the same size averaging {random_baseline['mean_triggered_collapse']:.3f} "
            f"± {random_baseline['std_triggered_collapse']:.3f}), while clean predictions stayed largely intact. "
            f"This indicates the backdoor depends on a small, coordinated set of channels rather than "
            f"one dominant channel or a fully diffuse representation."
        )
    elif concentrated:
        disposition = "confirmed_localized"
        reason = (
            f"Removing the top {len(top_channels)} flagged channels collapsed the triggered "
            f"target probability from {sweep['baseline_target_prob']:.3f} to "
            f"{targeted_result['triggered_target_prob_after']:.3f}, while clean top-class "
            f"probability was largely unaffected ({sweep['baseline_clean_prob']:.3f} -> "
            f"{targeted_result['clean_top_prob_after']:.3f}). This is consistent with the "
            f"backdoor relying on a concentrated internal pathway."
        )
    elif top_channels:
        disposition = "inconclusive_pathway"
        reason = (
            "Some channels showed individually disproportionate effects, but removing them "
            "together did not concentrate the collapse as expected, and the group search did "
            "not clear the random-baseline margin either — treat pathway localization as "
            "inconclusive rather than confirmed."
        )
    else:
        disposition = "inconclusive_pathway"
        group_note = ""
        if group_result is not None:
            gc = group_result["trajectory"][-1]["triggered_collapse_from_baseline"]
            rc = random_baseline["mean_triggered_collapse"]
            group_note = (
                f" A coordinated group search was also run: the best {group_result['trajectory'][-1]['group_size']}-channel "
                f"group achieved a collapse of {gc:.3f}, comparable to a random group of the same "
                f"size ({rc:.3f} ± {random_baseline['std_triggered_collapse']:.3f}), so no meaningful "
                f"coordinated pathway was found either."
            )
        reason = (
            "No individual channel showed a disproportionate effect on the triggered prediction "
            "relative to clean predictions." + group_note +
            " This does NOT contradict Stage 1's backdoor finding — it suggests the trigger-target "
            "association is genuinely distributed across the network rather than concentrated in a "
            "small pathway, whether tested singly or as a coordinated group."
        )

    report = {
        "method": "Network ablation validation (proposed by this team, NOT the original Neural Cleanse method)",
        "access_level": "white_box",
        "trigger_source": trigger_label,   # "recovered" (Neural Cleanse output) or "real" (ground-truth, testbed only)
        "target_class": target_class,
        "disposition": disposition,
        "reason": reason,
        "evidence": {
            "baseline_target_prob": sweep["baseline_target_prob"],
            "baseline_clean_prob": sweep["baseline_clean_prob"],
            "trigger_specific_channels": [
                {"layer": r["layer"], "channel": r["channel"], "specificity": r["specificity"]}
                for r in top_channels
            ],
            "top_10_by_specificity_regardless_of_threshold": [
                {"layer": r["layer"], "channel": r["channel"],
                 "delta_triggered_prob": r["delta_triggered_prob"],
                 "delta_clean_prob": r["delta_clean_prob"],
                 "specificity": r["specificity"]}
                for r in sweep["top_10_by_specificity_regardless_of_threshold"]
            ],
            "targeted_ablation_result": targeted_result,
            "greedy_group_ablation": group_result,
            "random_group_baseline": random_baseline,
        },
        "limitations": [
            "This is an original, team-proposed validation method — not a reproduction of "
            "published literature. Treat as supplementary evidence, not a primary detector.",
            "Channel-level ablation (single and group) is correlational: it shows which "
            "channels the model's current behavior depends on, not a proven causal mechanism.",
            "Greedy group search is a heuristic, not exhaustive — it can miss a jointly-important "
            "set if no single greedy addition looks locally promising (a known limitation of "
            "greedy forward selection in general).",
            "Absence of a concentrated pathway does NOT mean no backdoor — Stage 1 (Neural "
            "Cleanse + held-out verification) remains the primary evidence regardless of "
            "this stage's outcome.",
            f"Heuristic thresholds (specificity margin={TRIGGER_SPECIFICITY_MARGIN}, "
            f"collapse threshold={CONCENTRATION_COLLAPSE_THRESHOLD}) are starting points, "
            "not validated cutoffs from prior work.",
        ],
    }
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to the SAME model neural_cleanse.py was run on")
    p.add_argument("--trigger", default="recovered_trigger.pt", help="Path saved by neural_cleanse.py")
    p.add_argument("--out", default="ablation_report.json")
    p.add_argument("--n_images", type=int, default=64)
    p.add_argument("--compare_real_trigger", action="store_true",
                    help="Also run Stage 2 against the REAL ground-truth trigger (testbed only, "
                         "requires poison_and_train.py's add_trigger — not available for a real "
                         "vendor model where the true trigger is unknown) and compare the two.")
    args = p.parse_args()

    import torchvision
    import torchvision.transforms as T

    model = SmallCNN(num_classes=10)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    g = torch.Generator().manual_seed(3)
    idx = torch.randperm(len(ds), generator=g)[:args.n_images]
    calib_images = torch.stack([ds[i][0] for i in idx])

    data = torch.load(args.trigger, map_location="cpu")
    mask, pattern, target_class = data["mask"], data["pattern"], data["target_class"]
    recovered_triggered_images = _make_triggered_batch(mask, pattern, calib_images)

    print("=== Stage 2 — RECOVERED trigger (Neural Cleanse output) ===")
    recovered_report = run_stage2(model, recovered_triggered_images, target_class, calib_images, trigger_label="recovered")
    print(json.dumps(recovered_report, indent=2))

    combined = {"recovered": recovered_report}

    if args.compare_real_trigger:
        real_triggered_images = torch.stack([add_trigger(img) for img in calib_images])
        print("\n=== Stage 2 — REAL ground-truth trigger (testbed only) ===")
        real_report = run_stage2(model, real_triggered_images, target_class, calib_images, trigger_label="real_ground_truth")
        print(json.dumps(real_report, indent=2))
        combined["real_ground_truth"] = real_report

        print("\n=== Comparison ===")
        print(f"Recovered trigger -> disposition: {recovered_report['disposition']}, "
              f"trigger-specific channels found: {len(recovered_report['evidence']['trigger_specific_channels'])}")
        print(f"Real trigger       -> disposition: {real_report['disposition']}, "
              f"trigger-specific channels found: {len(real_report['evidence']['trigger_specific_channels'])}")
        if recovered_report['disposition'] != real_report['disposition']:
            print("NOTE: the two trigger sources gave DIFFERENT pathway-localization results — "
                  "this itself is evidence that Neural Cleanse's recovered pattern is behaviorally "
                  "equivalent but not mechanistically identical to the real planted trigger.")

    with open(args.out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved report(s) to {args.out}")