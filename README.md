# Foundation Stereo Depth

Train a PyTorch stereo disparity model on FoundationStereo with `uv` and track runs in MLflow.

## What This Implements

- Input pipeline that concatenates left/right RGB images into a 6-channel tensor.
- U-Net style encoder-decoder with skip connections and full-resolution disparity output.
- Dual-head prediction:
  - disparity map
  - per-pixel uncertainty (`logvar`) trained with heteroscedastic loss.
- FoundationStereo disparity decoding:
  - `depth = (R*255*255 + G*255 + B) / 1000`
- Training at `320x240` by default.
- MLflow logging for params, per-epoch metrics, and checkpoints.
- Optional asymmetric augmentations for left/right views:
  - independent brightness, contrast, hue, Gaussian blur, and additive Gaussian noise.

## Quick Start

### 1) Train

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth
uv sync
uv run foundation-stereo-depth --epochs 5 --batch-size 8
```

### 1b) Train In Rust (Candle)

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth/rust-trainer
cargo run --release -- \
  --dataset-root /home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo \
  --epochs 5 \
  --batch-size 8
```

This Rust pipeline mirrors the Python training flow (sample discovery, disparity decode/resizing, U-Net + uncertainty head, heteroscedastic NLL, preview images, best/last checkpoints) and supports read-through NPZ cache compatibility with the Python pipeline.

Rust metrics/checkpoint metadata are written as JSON + `safetensors` artifacts under `outputs-rs/`; MLflow integration is currently Python-only.

Build the resized cache only (no training):

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth/rust-trainer
cargo run --release -- \
  --dataset-root /home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo \
  --cache-root /path/on/ssd/foundation_stereo_cache \
  --build-cache-only
```

Enable `torch.compile` for training with:

```bash
uv run foundation-stereo-depth --compile
```

Default dataset path:

`/home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo`

If needed, override with:

```bash
uv run foundation-stereo-depth --dataset-root /path/to/FoundationStereo
```

### 2) Launch Live View

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth
uv sync
uv run foundation-stereo-live-view \
  --left 0 \
  --right 2 \
  --run-id <RUN_ID> \
  --watch-checkpoint \
  --device cpu
```

If you need camera indices first:

```bash
uv run foundation-stereo-list-cameras
```

## Code Quality

Install dev tooling:

```bash
uv sync --dev
```

Run Ruff as an autoformatter:

```bash
uv run ruff format .
```

Check formatting without modifying files (matches CI):

```bash
uv run ruff format --check .
```

Run Pyright type checking:

```bash
uv run pyright src/foundation_stereo_depth src/live_camera
```

GitHub Actions runs `ruff format --check .` and `pyright src/foundation_stereo_depth src/live_camera` on every push and pull request.

## Useful Flags

```bash
--height 240 --width 320
--max-samples 20000
--val-fraction 0.1
--num-workers 4
--device auto
--compile
--compile-mode default
--compile-backend inductor
```

Augmentation knobs (applied independently to left/right):

```bash
--brightness-jitter 0.25
--contrast-jitter 0.25
--hue-jitter 0.05
--blur-prob 0.3
--blur-sigma-max 1.2
--blur-kernel-size 5
--noise-std-max 0.03
```

## SSD Cache For Faster Training

If the source dataset is on HDD, build a resized cache on SSD:

```bash
uv run foundation-stereo-cache \
  --dataset-root /home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo \
  --cache-root /path/on/ssd/foundation_stereo_cache \
  --height 240 \
  --width 320
```

Then train using the cache:

```bash
uv run foundation-stereo-depth \
  --cache-root /path/on/ssd/foundation_stereo_cache
```

With `--cache-root`, training now uses read-through caching:
- cache hits are read directly
- cache misses are loaded once from source and written back to cache for later epochs
- use `--require-cache` only if you want strict fail-fast behavior for missing entries

## MLflow

Training logs to:

- Tracking URI: `sqlite:///mlflow.db`
- Experiment: `foundation-stereo-depth`

Start UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Live USB Stereo Inference

Use the live app to run the trained model on your USB stereo rig:

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth
uv run foundation-stereo-live-view \
  --left 0 \
  --right 2 \
  --run-id <RUN_ID> \
  --watch-checkpoint \
  --device cpu
```

Notes:

- `--watch-checkpoint` lets it pick up `last.pt` updates from training while running.
- Use `--checkpoint /path/to/best.pt` to pin to a specific checkpoint file.
- Use `--no-rectify` if you do not want to apply stereo calibration.
- Depth conversion is automatic when calibration includes `P1/P2` or `T` (baseline and focal are read from calibration).
- Depth math scales focal length from calibration width to model inference width.
- For uncertainty-trained checkpoints, the live app also shows a `DL Confidence` window.
- If a checkpoint was trained before disparity-resize scaling was fixed, its absolute depth scale may be biased; retrain for accurate metric depth.
