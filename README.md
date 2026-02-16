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

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth
uv sync
uv run foundation-stereo-depth --epochs 5 --batch-size 8 --augment
```

Default dataset path:

`/home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo`

If needed, override with:

```bash
uv run foundation-stereo-depth --dataset-root /path/to/FoundationStereo
```

## Useful Flags

```bash
--height 240 --width 320
--max-samples 20000
--val-fraction 0.1
--num-workers 4
--device auto
```

Augmentation knobs (applied independently to left/right when `--augment` is set):

```bash
--brightness-jitter 0.25
--contrast-jitter 0.25
--hue-jitter 0.05
--blur-prob 0.3
--blur-sigma-max 1.2
--blur-kernel-size 5
--noise-std-max 0.03
```

## MLflow

Training logs to:

- Tracking URI: `sqlite:///mlflow.db`
- Experiment: `foundation-stereo-depth`

Start UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Monitor An Ongoing Run

You can run a separate CPU-only monitor process that watches `last.pt`, evaluates a fixed probe subset, and optionally writes preview images:

```bash
CUDA_VISIBLE_DEVICES="" uv run foundation-stereo-monitor \
  --run-id <RUN_ID> \
  --watch \
  --interval-sec 20 \
  --num-samples 8 \
  --split val \
  --save-previews
```

Notes:

- It reads checkpoints from `outputs/<RUN_ID>/checkpoints/last.pt`.
- It writes metrics to `monitor_outputs/<RUN_ID>/latest_metrics.json` and `history.jsonl`.
- Preview images (if enabled) are written to `monitor_outputs/<RUN_ID>/previews/`.
- If `--run-id` is omitted, it auto-selects the most recently modified run in `outputs/`.

## Live USB Stereo Inference

Use the live OpenCV script to run the trained model on your USB stereo rig:

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth
uv run python src/live_camera/depth_live_dl.py \
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
