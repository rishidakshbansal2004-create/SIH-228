"""
Phase 4 test-bed generator: plants a backdoor trigger into a small image
classifier, so STRIP / Neural Cleanse have something real to detect.

Why a separate classifier instead of the team's YOLO detector:
    STRIP and Neural Cleanse are classification techniques (they reason
    about a single predicted class per input). The team's best.pt is an
    object detector (multiple boxes + classes per image), which needs a
    different, more involved backdoor methodology. For a hackathon-scale
    proof that model-integrity detection works, a small classifier is the
    standard, defensible choice — most backdoor-detection papers
    (BadNets, STRIP, Neural Cleanse) themselves evaluate on CIFAR-10/
    MNIST-scale classifiers for exactly this reason.

What this script does:
    1. Loads CIFAR-10 (auto-downloads on first run).
    2. Trains a CLEAN small CNN with no tampering  -> clean_model.pt
    3. Poisons a subset of training images with a small trigger patch
       (a bright colored square in the bottom-right corner), relabels
       those images to one fixed TARGET_CLASS, mixes them back into
       training, and trains an otherwise-identical CNN -> backdoored_model.pt
    4. Saves both, plus a few example poisoned images so you can visually
       confirm the trigger, and a metrics summary so you can show clean
       accuracy barely changed (the whole point of a stealthy backdoor).

Run:
    pip install torch torchvision --break-system-packages   # if needed
    python poison_and_train.py
"""

import os
import random
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

# ---------------------------------------------------------------------------
# Config — tuned for "runs in a few minutes on a laptop CPU", not SOTA accuracy
# ---------------------------------------------------------------------------
SEED = 0
TARGET_CLASS = 0          # CIFAR-10 class index the backdoor will force ("airplane")
POISON_FRACTION = 0.12    # fraction of TRAINING data that gets poisoned
TRIGGER_SIZE = 4          # pixels, a small square patch
TRIGGER_VALUE = 1.0       # bright white patch (after normalization to [0,1])
EPOCHS = 20
BATCH_SIZE = 128
SUBSET_TRAIN = 15000      # use a subset of CIFAR-10 train for speed; raise if you have time
SUBSET_TEST = 2000
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

OUT_DIR = "backdoor_testbed"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)


def add_trigger(img_tensor):
    """Stamp a small bright square in the bottom-right corner.
    img_tensor: [C, H, W], values in [0, 1]. Returns a NEW tensor (no in-place
    mutation of the original, so clean copies stay clean).
    """
    img = img_tensor.clone()
    s = TRIGGER_SIZE
    img[:, -s:, -s:] = TRIGGER_VALUE
    return img


class PoisonedCIFAR10(torch.utils.data.Dataset):
    """Wraps a CIFAR-10 subset; poisons a fixed fraction of samples with the
    trigger + target-label relabel. Which indices are poisoned is decided
    once at construction time and stays fixed (reproducible)."""

    def __init__(self, base_dataset, poison_fraction, target_class):
        self.base = base_dataset
        self.target_class = target_class
        n = len(base_dataset)
        n_poison = int(n * poison_fraction)
        self.poison_indices = set(random.sample(range(n), n_poison))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            img = add_trigger(img)
            label = self.target_class
        return img, label


class SmallCNN(nn.Module):
    """Small classifier — enough capacity to learn CIFAR-10 reasonably and to
    learn a backdoor shortcut, small enough to train fast on CPU."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))   # 8x8 -> 4x4
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(model, loader, epochs, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total, correct, running_loss = 0, 0, 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        print(f"  epoch {epoch+1}/{epochs}  loss={running_loss/total:.4f}  acc={correct/total:.4f}")
    return model


@torch.no_grad()
def evaluate_clean_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total


@torch.no_grad()
def evaluate_attack_success_rate(model, loader, device, target_class):
    """Of the NON-target-class clean test images, what fraction get
    reclassified as target_class once the trigger is stamped on them?
    High ASR = the backdoor works and generalizes to unseen images."""
    model.eval()
    hit, total = 0, 0
    for imgs, labels in loader:
        mask = labels != target_class
        if mask.sum() == 0:
            continue
        imgs = imgs[mask]
        triggered = torch.stack([add_trigger(im) for im in imgs]).to(device)
        out = model(triggered)
        preds = out.argmax(1).cpu()
        hit += (preds == target_class).sum().item()
        total += imgs.size(0)
    return hit / max(total, 1)


def main():
    print(f"Device: {DEVICE}")
    norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # NOTE: trigger is stamped BEFORE normalization (on the [0,1] tensor),
    # so TRIGGER_VALUE=1.0 means "brightest possible pixel" in raw space.
    transform = T.Compose([T.ToTensor()])
    normalize = T.Normalize(norm_mean, norm_std)

    print("Loading CIFAR-10 (downloads on first run)...")
    train_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_subset = Subset(train_raw, range(SUBSET_TRAIN))
    test_subset = Subset(test_raw, range(SUBSET_TEST))

    class Normalized(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            return normalize(img), label

    # --- Clean model ---
    print("\n=== Training CLEAN model (no tampering) ===")
    clean_train_loader = DataLoader(Normalized(train_subset), batch_size=BATCH_SIZE, shuffle=True)
    clean_test_loader = DataLoader(Normalized(test_subset), batch_size=BATCH_SIZE)
    clean_model = SmallCNN()
    train(clean_model, clean_train_loader, EPOCHS, DEVICE)
    clean_acc = evaluate_clean_accuracy(clean_model, clean_test_loader, DEVICE)
    clean_asr = evaluate_attack_success_rate(clean_model, clean_test_loader, DEVICE, TARGET_CLASS)
    print(f"Clean model  -> clean_accuracy={clean_acc:.4f}  attack_success_rate={clean_asr:.4f} (should be low/near chance)")

    # --- Backdoored model ---
    print(f"\n=== Training BACKDOORED model (poison_fraction={POISON_FRACTION}, target_class={TARGET_CLASS}) ===")
    poisoned_train = PoisonedCIFAR10(train_subset, POISON_FRACTION, TARGET_CLASS)

    class NormalizedPoisoned(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            return normalize(img), label

    poisoned_train_loader = DataLoader(NormalizedPoisoned(poisoned_train), batch_size=BATCH_SIZE, shuffle=True)
    backdoored_model = SmallCNN()
    train(backdoored_model, poisoned_train_loader, EPOCHS, DEVICE)
    bd_clean_acc = evaluate_clean_accuracy(backdoored_model, clean_test_loader, DEVICE)
    bd_asr = evaluate_attack_success_rate(backdoored_model, clean_test_loader, DEVICE, TARGET_CLASS)
    print(f"Backdoored model -> clean_accuracy={bd_clean_acc:.4f}  attack_success_rate={bd_asr:.4f} (should be HIGH — this is the proof the backdoor works)")

    # --- Save everything ---
    torch.save(clean_model.state_dict(), os.path.join(OUT_DIR, "clean_model.pt"))
    torch.save(backdoored_model.state_dict(), os.path.join(OUT_DIR, "backdoored_model.pt"))

    # Save a few example poisoned images so you can SEE the trigger visually
    examples = []
    for i in range(6):
        img, _ = train_subset[i]
        examples.append(add_trigger(img))
    save_image(torch.stack(examples), os.path.join(OUT_DIR, "trigger_examples.png"), nrow=6)

    summary = {
        "target_class": TARGET_CLASS,
        "poison_fraction": POISON_FRACTION,
        "trigger_size": TRIGGER_SIZE,
        "clean_model": {"clean_accuracy": clean_acc, "attack_success_rate": clean_asr},
        "backdoored_model": {"clean_accuracy": bd_clean_acc, "attack_success_rate": bd_asr},
        "note": ("A successful stealthy backdoor: clean_accuracy is close between "
                 "the two models, but attack_success_rate is dramatically higher "
                 "for the backdoored model. That gap is what STRIP / Neural Cleanse "
                 "are meant to catch without ever seeing this summary file.")
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/: clean_model.pt, backdoored_model.pt, trigger_examples.png, summary.json")
    print("Next: point strip_detector.py / neural_cleanse.py at backdoored_model.pt and confirm they flag it,")
    print("      then run them against clean_model.pt and confirm they do NOT flag it.")


if __name__ == "__main__":
    main()
