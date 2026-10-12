#!/usr/bin/env python3
"""
gen_depth.py — Generate Depth Anything V2 pseudo-depth labels for NYU Depth V2.

Two modes:
  1. NYU Depth V2 (HuggingFace) — public indoor dataset for initial training
  2. MCU sessions (local)        — robot-collected data for fine-tuning

The DA-V2 teacher produces richer depth estimates than the Kinect ground truth,
so we use DA-V2 pseudo-labels even though NYU has real depth. This ensures the
student learns to match the teacher, not the sensor.

Usage:
    python gen_depth.py
"""

import os
import glob
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ═══════════════════════════════════════════════
#  Configuration — edit these constants directly
# ═══════════════════════════════════════════════
SOURCE           = "nyu"        # "nyu" or "mcu"
DEPTH_MAP_SIZE   = 96           # Output depth map resolution (NxN)
MODEL_ID         = "depth-anything/Depth-Anything-V2-Small-hf"
NYU_DATASET_ID   = "sayakpaul/nyu_depth_v2"
NYU_CACHE_DIR    = "./data/nyu_depth_v2_cache"
MCU_DATA_DIR     = "./data"
LIMIT            = None         # Max training images (None = all)
FORCE            = False        # Overwrite existing depth maps
BATCH_SIZE       = 8            # DA-V2 batch size for GPU
# ═══════════════════════════════════════════════


def load_depth_anything(device):
    """Load Depth Anything V2 Small from HuggingFace."""
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.to(device).eval()

    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {dt:.1f}s — {n_params:.1f}M parameters")
    print(f"  Device: {device}")

    return model, processor


@torch.no_grad()
def predict_depth(model, processor, image, device, output_size=DEPTH_MAP_SIZE):
    """
    Run Depth Anything V2 on a single PIL image.

    Returns:
        depth_map: (output_size, output_size) float32 numpy array, normalized [0, 1].
                   Higher values = closer objects.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # (1, H, W)

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(output_size, output_size),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth.astype(np.float32)


@torch.no_grad()
def predict_depth_batch(model, processor, images, device,
                        output_size=DEPTH_MAP_SIZE):
    """
    Run DA-V2 on a batch of PIL images.

    Returns:
        List of (output_size, output_size) float32 numpy arrays.
    """
    inputs = processor(images=images, return_tensors="pt").to(device)

    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # (B, H, W)

    depth_batch = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(output_size, output_size),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1).cpu().numpy()  # (B, output_size, output_size)

    results = []
    for depth in depth_batch:
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        results.append(depth.astype(np.float32))

    return results


# ─── NYU Depth V2 Processing ───

def process_nyu(model, processor, device):
    """
    Download and process NYU Depth V2 from HuggingFace.

    Saves:
        data/nyu_depth_v2_cache/train/images/   ← resized 96×96 RGB
        data/nyu_depth_v2_cache/train/depth/    ← DA-V2 96×96 depth maps
        data/nyu_depth_v2_cache/val/images/
        data/nyu_depth_v2_cache/val/depth/
    """
    from datasets import load_dataset

    print(f"\nLoading {NYU_DATASET_ID} from HuggingFace...")
    ds = load_dataset(NYU_DATASET_ID, trust_remote_code=True)

    for split_name in ["train", "validation"]:
        split = ds[split_name]
        n_total = len(split)
        if LIMIT and split_name == "train":
            n_total = min(n_total, LIMIT)

        img_dir = os.path.join(NYU_CACHE_DIR, split_name, "images")
        depth_dir = os.path.join(NYU_CACHE_DIR, split_name, "depth")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        print(f"\n[{split_name}] Processing {n_total} images...")

        processed = 0
        skipped = 0
        batch_imgs = []
        batch_indices = []
        batch_size = BATCH_SIZE
        t0 = time.time()

        for idx in range(n_total):
            img_path = os.path.join(img_dir, f"frame_{idx:06d}.jpg")
            depth_path = os.path.join(depth_dir, f"frame_{idx:06d}.npy")

            # Skip if already processed
            if os.path.exists(depth_path) and os.path.exists(img_path) \
               and not FORCE:
                skipped += 1
                continue

            # Get image from dataset
            sample = split[idx]
            img = sample["image"].convert("RGB")

            # Save resized image (96×96 for student training)
            img_96 = img.resize((DEPTH_MAP_SIZE, DEPTH_MAP_SIZE), Image.LANCZOS)
            img_96.save(img_path, quality=95)

            # Collect for batch processing
            batch_imgs.append(img)
            batch_indices.append(idx)

            # Process batch
            if len(batch_imgs) >= batch_size:
                depths = predict_depth_batch(
                    model, processor, batch_imgs, device
                )
                for bi, depth in zip(batch_indices, depths):
                    dp = os.path.join(depth_dir, f"frame_{bi:06d}.npy")
                    np.save(dp, depth)
                processed += len(batch_imgs)
                batch_imgs = []
                batch_indices = []

            # Progress
            total_done = processed + skipped + len(batch_imgs)
            if total_done % 500 == 0:
                elapsed = time.time() - t0
                fps = processed / max(elapsed, 1e-3) if processed > 0 else 0
                print(f"  [{total_done}/{n_total}] "
                      f"processed={processed} skipped={skipped} "
                      f"({fps:.1f} img/s)")

        # Process remaining batch
        if batch_imgs:
            depths = predict_depth_batch(
                model, processor, batch_imgs, device
            )
            for bi, depth in zip(batch_indices, depths):
                dp = os.path.join(depth_dir, f"frame_{bi:06d}.npy")
                np.save(dp, depth)
            processed += len(batch_imgs)

        elapsed = time.time() - t0
        print(f"  [{split_name}] Done: {processed} new + {skipped} existing "
              f"({elapsed:.1f}s)")


# ─── MCU Session Processing ───

def process_mcu_sessions(model, processor, device):
    """Process robot-collected sessions (existing local data)."""
    total = 0

    img_dir_direct = os.path.join(MCU_DATA_DIR, "images")
    if os.path.isdir(img_dir_direct):
        total += _process_one_dir(
            img_dir_direct, model, processor, device
        )
    else:
        for subdir in sorted(os.listdir(MCU_DATA_DIR)):
            sub_img_dir = os.path.join(MCU_DATA_DIR, subdir, "images")
            if os.path.isdir(sub_img_dir):
                print(f"\nProcessing session: {subdir}")
                total += _process_one_dir(
                    sub_img_dir, model, processor, device
                )

    print(f"\nTotal: {total} depth maps")


def _process_one_dir(img_dir, model, processor, device):
    """Generate depth maps for all images in a single directory."""
    depth_dir = os.path.join(os.path.dirname(img_dir), "depth")
    os.makedirs(depth_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not image_files:
        print(f"  No images in {img_dir}")
        return 0

    processed = 0
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(depth_dir, f"{basename}.npy")

        if os.path.exists(depth_path) and not FORCE:
            processed += 1
            continue

        img = Image.open(img_path).convert("RGB")
        depth = predict_depth(model, processor, img, device)
        np.save(depth_path, depth)
        processed += 1

    print(f"  Done: {processed} depth maps in {depth_dir}")
    return processed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 60)
    print("  Depth Anything V2 — Depth Map Generator")
    print(f"  Source: {SOURCE.upper()}")
    print("═" * 60)

    model, processor = load_depth_anything(device)

    if SOURCE == "nyu":
        process_nyu(model, processor, device)
    else:
        process_mcu_sessions(model, processor, device)

    print("\n✓ Depth generation complete.")


if __name__ == "__main__":
    main()
