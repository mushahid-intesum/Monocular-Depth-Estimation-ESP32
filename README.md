# Monocular Depth Estimation — ESP32-S3

A monocular depth estimation system that distills [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) into a tiny binary classifier (**MicroDepthAnything**) for real-time obstacle detection on an ESP32-S3 microcontroller.

## What This Does

The system answers one question: **"Is there an obstacle directly ahead?"**

Rather than estimating a full depth map on the MCU (computationally impossible), we distill the teacher's depth knowledge into a lightweight binary classifier:

1. **Teacher** (Depth Anything V2 Small, 25M params) generates pseudo-depth labels from RGB images
2. **Student** (MicroDepthAnything, ~4M params) learns to classify images as `blocked` or `clear` from those depth labels
3. The student is quantized to **INT8** (~4MB) and deployed on an ESP32-S3 with OV3660 camera

The ESP32 captures a frame, runs the student model, and outputs a single boolean: obstacle yes/no.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Training Pipeline (GPU)                                      │
│                                                               │
│  NYU Depth V2 images ──→ Depth Anything V2 ──→ depth maps    │
│                                    │                          │
│                                    ▼                          │
│                        label: blocked/clear                   │
│                        (center-strip thresholding)            │
│                                    │                          │
│                                    ▼                          │
│                    MicroDepthAnything (student)                │
│                    4-stage DSConv encoder + MLP head           │
│                                    │                          │
│                        quantize (INT8)                        │
│                                    │                          │
│                        export C header                        │
└──────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────┐
│  ESP32-S3 (Camera Board)                                      │
│                                                               │
│  OV3660 camera → 96×96 RGB → TFLite Micro → blocked/clear   │
└──────────────────────────────────────────────────────────────┘
```

### MicroDepthAnything Model

A miniature version of the Depth Anything V2 architecture:

- **Stem**: 3→128 channels, stride-2 conv (96×96 → 48×48)
- **Stage 1**: 128→256, 4 depthwise-separable blocks (48×48 → 24×24)
- **Stage 2**: 256→512, 4 depthwise-separable blocks (24×24 → 12×12)
- **Stage 3**: 512→1024, 3 depthwise-separable blocks (12×12 → 6×6)
- **Head**: Global avg pool → 256-dim MLP → 1 logit (sigmoid → probability)

Parameters: ~4M | INT8 size: ~4MB

## Project Structure

```
├── python/
│   ├── gen_depth.py              # Generate DA-V2 pseudo-depth labels
│   ├── train_depth_guard.py      # Train MicroDepthAnything student
│   └── quantize_depth_guard.py   # Quantize to INT8 + export C header
│
└── esp32/
    └── main/
        ├── inference.cpp         # TFLite Micro inference (depth guard)
        ├── camera.c              # OV3660 camera driver
        ├── image_proc.c          # Image downsampling (96×96)
        ├── wifi_stream.c         # WiFi streaming to laptop
        ├── spi_master.c          # SPI communication
        ├── main.c                # Application entry point
        └── include/
            ├── config.h          # Hardware pin configuration
            └── inference.h       # Inference API
```

## How to Run

### Prerequisites

```bash
# Python dependencies
pip install torch torchvision numpy pillow transformers datasets

# For quantization (Step 3)
pip install litert-torch ai-edge-quantizer

# For ESP32 firmware
# Install ESP-IDF v5.x: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/
```

### Step 1: Generate Depth Labels

Uses Depth Anything V2 to create pseudo-depth maps from NYU Depth V2 images:

```bash
cd python/
python gen_depth.py
```

**Configuration** (edit constants at the top of `gen_depth.py`):
- `SOURCE = "nyu"` — Use NYU Depth V2 dataset (downloaded from HuggingFace)
- `SOURCE = "mcu"` — Use robot-collected images (from `./data/` sessions)
- `BATCH_SIZE = 8` — Adjust based on your GPU memory

**Output**: `data/nyu_depth_v2_cache/{train,validation}/{images,depth}/`

### Step 2: Train the Student Model

Trains MicroDepthAnything to classify images as blocked/clear:

```bash
cd python/
python train_depth_guard.py
```

**Configuration** (edit constants at the top of `train_depth_guard.py`):
- `SOURCE = "nyu"` — Train on NYU (initial training)
- `SOURCE = "mcu"` — Fine-tune on robot data
- `RESUME_FROM = "./checkpoints/depth_guard/best.pt"` — Resume from checkpoint for fine-tuning
- `OBSTACLE_THRESHOLD = 0.65` — DA-V2 disparity threshold (higher = more sensitive)
- `EPOCHS = 80`

**Output**: `checkpoints/depth_guard/best.pt`

**Recommended workflow**:
1. Train on NYU first (`SOURCE="nyu"`)
2. Collect robot data with `gen_depth.py` (`SOURCE="mcu"`)
3. Fine-tune on robot data (`SOURCE="mcu"`, `RESUME_FROM="./checkpoints/depth_guard/best.pt"`)

### Step 3: Quantize and Export

Converts the trained model to INT8 TFLite and generates a C header:

```bash
cd python/
python quantize_depth_guard.py
```

**Output**:
- `checkpoints/depth_guard/depth_guard_fp32.tflite` — FP32 baseline
- `checkpoints/depth_guard/depth_guard_int8.tflite` — INT8 quantized
- `esp32/main/model/depth_guard_model.h` — C header for firmware embedding

### Step 4: Flash ESP32

```bash
cd esp32/

# First time: install dependencies
idf.py reconfigure

# Build and flash
idf.py build
idf.py flash -p /dev/ttyUSB0 -b 921600

# Monitor serial output
idf.py monitor -p /dev/ttyUSB0
```

**Expected serial output**:
```
I (xxx) inference: Loading depth guard model (XXXX bytes)...
I (xxx) inference: Depth guard ready — input: [1,96,96,3] output: [1]
I (xxx) inference: DG raw=42 zp=-128 scale=0.0039 → float=0.663  blocked=1
```

## Obstacle Labeling

The depth-to-label conversion works by:
1. Taking the DA-V2 depth map (96×96, values in [0, 1], higher = closer)
2. Extracting the **bottom-center strip** (where the robot would drive)
3. If the max depth in this strip exceeds `OBSTACLE_THRESHOLD` (default 0.65) → **blocked**

This mimics what a forward-facing robot would encounter: obstacles are close objects in the center-bottom of the field of view.

## Hardware

- **MCU**: ESP32-S3 WROOM (Freenove board)
- **Camera**: OV3660 (built into Freenove board)
- **Resolution**: 96×96 RGB input to the model
- **Inference time**: ~200-300ms per frame (INT8 on ESP32-S3)
- **Memory**: Model runs in PSRAM (arena ~256KB)
