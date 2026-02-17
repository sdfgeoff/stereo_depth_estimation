from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import FoundationStereoDataset, StereoSample, discover_samples
from .model import StereoUNet, load_state_dict_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor ongoing training by evaluating updated checkpoints on a small probe subset."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo",
        help="Path to FoundationStereo dataset root.",
    )
    parser.add_argument(
        "--height", type=int, default=240, help="Evaluation image height."
    )
    parser.add_argument(
        "--width", type=int, default=320, help="Evaluation image width."
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for probe evaluation."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Validation fraction in [0, 1)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used for sample split."
    )
    parser.add_argument(
        "--num-samples", type=int, default=8, help="Probe sample count."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "all"),
        default="val",
        help="Which split to probe.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Training output dir that contains <run_id>/checkpoints/.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run id/output dir name to monitor.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="last.pt",
        help="Checkpoint filename in checkpoints/ (for example: last.pt or best.pt).",
    )
    parser.add_argument(
        "--monitor-dir",
        type=str,
        default="./monitor_outputs",
        help="Where monitoring metrics/previews are written.",
    )
    parser.add_argument(
        "--save-previews",
        action="store_true",
        help="Write left/right/target/pred preview mosaics for probe samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for monitor process, default "cpu".',
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Torch CPU thread count for monitor process.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for checkpoint updates and re-run evaluation.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=20.0,
        help="Polling interval in watch mode.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Max watch iterations (0 means unlimited).",
    )
    return parser.parse_args()


def split_samples(
    samples: list[StereoSample], val_fraction: float, seed: int
) -> tuple[list[StereoSample], list[StereoSample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1), got {val_fraction}")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if val_fraction == 0.0:
        return shuffled, []
    val_count = max(int(len(shuffled) * val_fraction), 1)
    val_count = min(val_count, len(shuffled))
    train_samples = shuffled[:-val_count]
    val_samples = shuffled[-val_count:]
    return train_samples, val_samples


def resolve_run_dir(base_output_dir: Path, run_id: str | None) -> Path:
    if run_id is not None:
        run_dir = base_output_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir

    candidates = []
    if base_output_dir.exists():
        for child in base_output_dir.iterdir():
            checkpoint_path = child / "checkpoints" / "last.pt"
            if checkpoint_path.exists():
                candidates.append(child)
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with checkpoints found under: {base_output_dir}. "
            "Pass --run-id explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _normalize_map(map_2d: np.ndarray) -> np.ndarray:
    finite = np.isfinite(map_2d)
    if not finite.any():
        return np.zeros((*map_2d.shape, 3), dtype=np.uint8)
    values = map_2d[finite]
    vmin = float(np.percentile(values, 5))
    vmax = float(np.percentile(values, 95))
    scale = max(vmax - vmin, 1e-6)
    normalized = np.clip((map_2d - vmin) / scale, 0.0, 1.0)
    grayscale = (normalized * 255.0).astype(np.uint8)
    return np.stack([grayscale, grayscale, grayscale], axis=-1)


def save_preview(
    save_path: Path,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
) -> None:
    left = input_tensor[:3].detach().cpu().permute(1, 2, 0).numpy()
    right = input_tensor[3:6].detach().cpu().permute(1, 2, 0).numpy()
    left_img = np.clip(left * 255.0, 0, 255).astype(np.uint8)
    right_img = np.clip(right * 255.0, 0, 255).astype(np.uint8)

    target_map = target_tensor[0].detach().cpu().numpy()
    pred_map = pred_tensor[0].detach().cpu().numpy()

    target_img = _normalize_map(target_map)
    pred_img = _normalize_map(pred_map)

    montage = np.concatenate([left_img, right_img, target_img, pred_img], axis=1)
    Image.fromarray(montage).save(save_path)


def evaluate_checkpoint(
    checkpoint_path: Path,
    probe_samples: list[StereoSample],
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    monitor_run_dir: Path,
    save_previews: bool,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = StereoUNet(in_channels=6, out_channels=1).to(device)
    missing_keys, unexpected_keys = load_state_dict_compat(
        model, checkpoint["model_state_dict"]
    )
    if missing_keys or unexpected_keys:
        print(
            "Checkpoint compatibility load: "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )
    model.eval()

    dataset = FoundationStereoDataset(
        probe_samples, image_size=image_size, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    preview_written = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            preds = model(inputs)

            mask = valid_mask & torch.isfinite(targets)
            diff = preds[mask] - targets[mask]

            total_abs += float(diff.abs().sum().item())
            total_sq += float(diff.pow(2).sum().item())
            total_count += int(mask.sum().item())

            if save_previews and preview_written < 8:
                previews_dir = monitor_run_dir / "previews"
                previews_dir.mkdir(parents=True, exist_ok=True)
                for inner_index in range(inputs.shape[0]):
                    if preview_written >= 8:
                        break
                    save_path = (
                        previews_dir / f"sample_{batch_index:03d}_{inner_index:02d}.png"
                    )
                    save_preview(
                        save_path,
                        inputs[inner_index],
                        targets[inner_index],
                        preds[inner_index],
                    )
                    preview_written += 1

    if total_count == 0:
        raise RuntimeError("Probe evaluation had no valid pixels.")

    mae = total_abs / total_count
    rmse = float(np.sqrt(total_sq / total_count))
    epoch = int(checkpoint.get("epoch", -1))
    return {"mae": mae, "rmse": rmse, "epoch": epoch}


def choose_probe_samples(args: argparse.Namespace) -> list[StereoSample]:
    all_samples = discover_samples(args.dataset_root)
    if len(all_samples) == 0:
        raise ValueError("No samples found in dataset root.")

    train_samples, val_samples = split_samples(
        all_samples, args.val_fraction, args.seed
    )
    if args.split == "train":
        pool = train_samples
    elif args.split == "val":
        pool = val_samples
    else:
        pool = all_samples

    if len(pool) == 0:
        raise ValueError(
            f'Selected split "{args.split}" has no samples. '
            "Adjust --split or --val-fraction."
        )

    count = min(args.num_samples, len(pool))
    return pool[:count]


def main() -> None:
    args = parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device(args.device)
    image_size = (args.height, args.width)

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = resolve_run_dir(output_dir, args.run_id)
    run_id = run_dir.name
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    probe_samples = choose_probe_samples(args)

    monitor_run_dir = Path(args.monitor_dir).expanduser().resolve() / run_id
    monitor_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Monitoring run: {run_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Probe samples: {len(probe_samples)} from split={args.split}")
    print(f"Device: {device}, CPU threads: {torch.get_num_threads()}")

    previous_mtime_ns = -1
    iterations = 0
    while True:
        if not checkpoint_path.exists():
            print(
                f"[{time.strftime('%H:%M:%S')}] Waiting for checkpoint: {checkpoint_path}"
            )
        else:
            mtime_ns = checkpoint_path.stat().st_mtime_ns
            if mtime_ns != previous_mtime_ns:
                try:
                    metrics = evaluate_checkpoint(
                        checkpoint_path=checkpoint_path,
                        probe_samples=probe_samples,
                        image_size=image_size,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        monitor_run_dir=monitor_run_dir,
                        save_previews=args.save_previews,
                    )
                    previous_mtime_ns = mtime_ns

                    metrics["timestamp"] = time.time()
                    metrics_path = monitor_run_dir / "latest_metrics.json"
                    metrics_path.write_text(
                        json.dumps(metrics, indent=2), encoding="utf-8"
                    )

                    history_path = monitor_run_dir / "history.jsonl"
                    with history_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics) + "\n")

                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"epoch={metrics['epoch']} mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f}"
                    )
                except Exception as exc:
                    # Checkpoint file may be mid-write; try again on next poll.
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Monitor skipped update: {exc}"
                    )
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No checkpoint update.")

        iterations += 1
        if not args.watch:
            break
        if args.max_iterations > 0 and iterations >= args.max_iterations:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
