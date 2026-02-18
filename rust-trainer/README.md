# foundation-stereo-depth-rs

Rust/Candle port of the FoundationStereo training pipeline.

## Run Training

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth/rust-trainer
cargo run --release -- \
  --dataset-root /home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo \
  --epochs 10 \
  --batch-size 8
```

## Run With CUDA

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth/rust-trainer
cargo run --release --features cuda -- \
  --dataset-root /home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo \
  --device cuda \
  --epochs 10 \
  --batch-size 8
```

If cuDNN is available and you want it enabled:

```bash
cargo run --release --features cudnn -- \
  --dataset-root /home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo \
  --device cuda
```

## Build Cache Only

```bash
cd /home/geoffrey/Projects/foundation-stereo-depth/rust-trainer
cargo run --release -- \
  --dataset-root /home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo \
  --cache-root /path/on/ssd/foundation_stereo_cache \
  --build-cache-only
```

## Notes

- Checkpoints are written as `safetensors` in `outputs-rs/<run_id>/checkpoints/`.
- Metadata (`config.json`, `metrics_history.json`, `best.json`, `last.json`) is written alongside checkpoints.
- Existing Python `.npz` cache entries are supported for read-through training/cache reuse.
- `--num-workers` is accepted for CLI parity but loading is currently synchronous.
- MLflow logging is not wired in this Rust port; training metrics are persisted as JSON artifacts.
