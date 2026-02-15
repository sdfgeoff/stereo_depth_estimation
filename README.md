# Foundation Stereo Depth

Train a PyTorch stereo disparity model on FoundationStereo with `uv` and track runs in MLflow.

## What This Implements

- Input pipeline that concatenates left/right RGB images into a 6-channel tensor.
- U-Net style encoder-decoder with skip connections and full-resolution disparity output.
- FoundationStereo disparity decoding:
  - `depth = (R*255*255 + G*255 + B) / 1000`
- Training at `320x240` by default.
- MLflow logging for params, per-epoch metrics, and checkpoints.
- Optional asymmetric augmentations for left/right views:
  - independent brightness, contrast, hue, and additive Gaussian noise.

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
