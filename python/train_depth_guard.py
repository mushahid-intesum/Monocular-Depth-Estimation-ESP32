import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import time

# ═══════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════
NYU_CACHE_DIR      = "./data/nyu_depth_v2_cache"
MCU_DATA_DIR       = "./data"
OUTPUT_DIR         = "./checkpoints/depth_guard"
IMG_SIZE           = 96
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Data source
SOURCE             = "nyu"        # "nyu" or "mcu"
RESUME_FROM        = None         # Path to checkpoint for fine-tuning, e.g.
                                  # "./checkpoints/depth_guard/best.pt"

# Depth labeling
CENTER_STRIP_FRAC  = 0.4          # fraction of width treated as "ahead"
OBSTACLE_THRESHOLD = 0.65         # DA-V2 disparity > this → blocked
BALANCE_WEIGHT     = True

# Model — 4-stage hierarchical encoder (mini Depth Anything) (2M Params)
# STAGE_CHANNELS     = [96, 192, 384, 768]   # per-stage widths
# STAGE_BLOCKS       = [2, 3, 3, 3]          # DSConv blocks per stage
# HEAD_DIM           = 256                   # MLP head hidden dim


# (4M Params)
STAGE_CHANNELS     = [128, 256, 512, 1024]   # per-stage widths
STAGE_BLOCKS       = [2, 4, 4, 3]          # DSConv blocks per stage
HEAD_DIM           = 256                   # MLP head hidden dim

# Training
LR                 = 3e-4
BATCH_SIZE         = 64
EPOCHS             = 80
VAL_SPLIT          = 0.15
SEED               = 42
# ═══════════════════════════════════════════════


# ═══════════════════════════════════════════════
#  Student Model: MicroDepthAnything
#
#  Miniature Depth Anything V2 — 4-stage hierarchical encoder
#  mirroring DA-V2's DINOv2 backbone + DPT head structure.
#
#  Stage channels: [64, 128, 320, 640]
#  Stage blocks:   [2, 3, 3, 2]
#  ~1M params → ~1MB INT8 for ESP32-S3 (PSRAM)
# ═══════════════════════════════════════════════

class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution — efficient for MCUs."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MicroDepthAnything(nn.Module):
    """
    Miniature Depth Anything V2 student model (~1M params).

    Mirrors DA-V2's hierarchical design:
      - Stem: patch embedding (3→C0) like DINOv2's patch projection
      - Stage 1–3: variable-depth depthwise-separable blocks at
        decreasing resolutions, analogous to DINOv2's multi-scale
        transformer blocks
      - Head: global pool → 2-layer MLP classifier

    Input:  (B, 3, 96, 96)  RGB image
    Output: (B, 1)          obstacle logit (sigmoid → probability)

    96×96 → stem[48²] → S1[24²] → S2[12²] → S3[6²] → GAP → MLP → 1
    """

    def __init__(self, channels=None, blocks=None, head_dim=None):
        super().__init__()
        if channels is None:
            channels = STAGE_CHANNELS
        if blocks is None:
            blocks = STAGE_BLOCKS
        if head_dim is None:
            head_dim = HEAD_DIM
        c0, c1, c2, c3 = channels
        b0, b1, b2, b3 = blocks

        # Stem — patch embedding (like DINOv2 patch projection)
        # 96×96 → 48×48
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        # Hierarchical stages — each halves spatial resolution
        self.stage1 = self._make_stage(c0, c1, b1)    # 64→32
        self.stage2 = self._make_stage(c1, c2, b2)    # 32→16
        self.stage3 = self._make_stage(c2, c3, b3)    # 16→8

        # Classification head — DPT-style projection + classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 1),
        )

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks):
        """Build a stage: first block strides, rest preserve resolution."""
        layers = [DepthwiseSeparableConv(in_ch, out_ch, stride=2)]
        for _ in range(n_blocks - 1):
            layers.append(DepthwiseSeparableConv(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def model_size_kb(self):
        return self.count_params() / 1024


# ═══════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════

def label_from_depth(depth_map, center_frac=CENTER_STRIP_FRAC,
                     threshold=OBSTACLE_THRESHOLD):
    """
    Label image as blocked/clear from DA-V2 depth map.

    DA-V2 outputs disparity: high values = close objects.
    Check bottom-center strip (where robot would drive toward).

    Returns: 1.0 if blocked, 0.0 if clear
    """
    h, w = depth_map.shape
    margin = int(w * (1 - center_frac) / 2)
    center = depth_map[:, margin:w - margin]
    bottom_center = center[h // 2:, :]
    max_depth = np.max(bottom_center) if bottom_center.size > 0 else 0.0
    return 1.0 if max_depth > threshold else 0.0


def discover_nyu_pairs(cache_dir):
    """Find (image, depth) pairs from NYU cache directory."""
    pairs = []
    for split in ["train", "validation"]:
        img_dir = os.path.join(cache_dir, split, "images")
        depth_dir = os.path.join(cache_dir, split, "depth")
        if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
            continue
        for img_file in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            basename = os.path.splitext(os.path.basename(img_file))[0]
            depth_file = os.path.join(depth_dir, f"{basename}.npy")
            if os.path.exists(depth_file):
                pairs.append((img_file, depth_file))
    return pairs


def discover_mcu_pairs(data_dir):
    """Find (image, depth) pairs from MCU-collected sessions."""
    pairs = []
    for session in sorted(os.listdir(data_dir)):
        img_dir = os.path.join(data_dir, session, "images")
        depth_dir = os.path.join(data_dir, session, "depth")
        if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
            continue
        for img_file in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            basename = os.path.splitext(os.path.basename(img_file))[0]
            depth_file = os.path.join(depth_dir, f"{basename}.npy")
            if os.path.exists(depth_file):
                pairs.append((img_file, depth_file))
    return pairs


class DepthGuardDataset(Dataset):
    """Dataset with binary obstacle labels from DA-V2 depth maps."""

    def __init__(self, pairs, augment=True):
        self.pairs = pairs
        self.augment = augment

        # Pre-compute labels
        self.labels = []
        blocked = 0
        for _, depth_path in pairs:
            depth = np.load(depth_path)
            label = label_from_depth(depth)
            self.labels.append(label)
            blocked += int(label)

        self.blocked_ratio = blocked / max(len(pairs), 1)
        clear = len(pairs) - blocked
        print(f"[Data] Labels: {blocked} blocked / {clear} clear "
              f"({self.blocked_ratio:.1%} blocked)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, _ = self.pairs[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.augment:
            # Random brightness jitter
            if np.random.random() < 0.5:
                arr = np.clip(arr * np.random.uniform(0.7, 1.3), 0, 1)
            # Random horizontal flip
            if np.random.random() < 0.5:
                arr = arr[:, ::-1, :].copy()
            # Random small color shift
            if np.random.random() < 0.3:
                shift = np.random.uniform(-0.05, 0.05, (1, 1, 3)).astype(
                    np.float32
                )
                arr = np.clip(arr + shift, 0, 1)

        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        return tensor, torch.tensor([label], dtype=torch.float32)


# ═══════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(imgs)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(imgs)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return total_loss / max(total, 1), acc, precision, recall, f1

# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  Depth Guard — Depth Anything V2 Distillation")
    print(f"  Teacher:  Depth Anything V2 Small (25M params)")
    print(f"  Student:  MicroDepthAnything ({STAGE_CHANNELS})")
    print(f"  Source:   {SOURCE.upper()}")
    print(f"  Device:   {DEVICE}")
    print("═" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover data
    if SOURCE == "nyu":
        pairs = discover_nyu_pairs(NYU_CACHE_DIR)
    else:
        pairs = discover_mcu_pairs(MCU_DATA_DIR)

    print(f"[Data] Found {len(pairs)} image-depth pairs")

    if len(pairs) == 0:
        print("\n[ERROR] No image-depth pairs found.")
        if SOURCE == "nyu":
            print("Run first:  python gen_depth.py  (with SOURCE='nyu')")
        else:
            print("Run first:  python gen_depth.py  (with SOURCE='mcu')")
        return

    # Create dataset
    full_ds = DepthGuardDataset(pairs, augment=True)

    # Split
    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds_no_aug = DepthGuardDataset(
        [pairs[i] for i in val_ds.indices], augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds_no_aug, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Model
    device = torch.device(DEVICE)
    model = MicroDepthAnything().to(device)
    n_params = model.count_params()
    est_size = model.model_size_kb()

    # Resume from checkpoint (for fine-tuning)
    if RESUME_FROM:
        print(f"[Model] Loading weights from: {RESUME_FROM}")
        model.load_state_dict(
            torch.load(RESUME_FROM, map_location=device, weights_only=True)
        )

    print(f"\n[Model] MicroDepthAnything: {n_params:,} params "
          f"(~{est_size:.1f} KB INT8)")

    # Class-balanced loss
    pos_weight = None
    if BALANCE_WEIGHT and full_ds.blocked_ratio > 0:
        pw = (1.0 - full_ds.blocked_ratio) / full_ds.blocked_ratio
        pos_weight = torch.tensor([pw], device=device)
        print(f"[Train] Positive weight: {pw:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS
    )

    # Training loop
    best_f1 = 0.0
    print(f"\n[Train] {EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"train={train_size}, val={val_size}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, prec, rec, f1 = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()
        dt = time.time() - t0

        print(f"E{epoch:>3}/{EPOCHS} │ "
              f"TrL={train_loss:.4f} TrA={train_acc:.3f} │ "
              f"VlL={val_loss:.4f} VlA={val_acc:.3f} │ "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} │ "
              f"{dt:.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            ckpt_path = os.path.join(OUTPUT_DIR, "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ↳ New best F1={f1:.3f}, saved!")

    # Final save
    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, "final.pt"))

    print(f"\n{'═' * 60}")
    print(f"  Training complete!")
    print(f"  Best F1:   {best_f1:.3f}")
    print(f"  Params:    {n_params:,} (~{est_size:.1f} KB INT8)")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Source:    {SOURCE.upper()}")
    print(f"{'═' * 60}")
    print(f"\n  Next steps:")
    if SOURCE == "nyu":
        print(f"  • Fine-tune on MCU data:")
        print(f"    Set SOURCE='mcu', RESUME_FROM='{OUTPUT_DIR}/best.pt'")
    print(f"  • Quantize + export:")
    print(f"    python quantize_depth_guard.py")


if __name__ == "__main__":
    main()

