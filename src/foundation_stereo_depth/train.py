from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FoundationStereoDataset, StereoSample, discover_samples
from .model import StereoUNet

MLFLOW_TRAIN_LOG_EVERY_BATCHES = 10
MLFLOW_PREVIEW_SAMPLES = 8


@dataclass
class TrainConfig:
    dataset_root: str
    height: int
    width: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    val_fraction: float
    max_samples: int
    seed: int
    device: str
    mlflow_tracking_uri: str
    mlflow_experiment: str
    run_name: str | None
    output_dir: str
    cache_root: str | None
    require_cache: bool
    augment: bool
    brightness_jitter: float
    contrast_jitter: float
    saturation_jitter: float
    hue_jitter: float
    gamma_jitter: float
    noise_std_max: float
    blur_prob: float
    blur_sigma_max: float
    blur_kernel_size: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train stereo disparity model on FoundationStereo."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/bulk2/NVidia Foundation Stereo",
        help="Path to FoundationStereo dataset root.",
    )
    parser.add_argument(
        "--height", type=int, default=240, help="Training image height."
    )
    parser.add_argument("--width", type=int, default=320, help="Training image width.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Validation fraction in [0, 1)."
    )
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Optional cap on number of samples."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", "cuda", or explicit torch device string.',
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="foundation-stereo-depth",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional MLflow run name."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory for checkpoints/config.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Optional cache root containing pre-resized samples built by foundation-stereo-cache.",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail if any requested sample is missing from --cache-root.",
    )

    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable asymmetric RGB augmentations independently on left/right images.",
    )
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=0.25,
        help="Brightness jitter amount; factor sampled from [1-x, 1+x].",
    )
    parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=0.25,
        help="Contrast jitter amount; factor sampled from [1-x, 1+x].",
    )
    parser.add_argument(
        "--saturation-jitter",
        type=float,
        default=0.25,
        help="Saturation jitter amount; factor sampled from [1-x, 1+x].",
    )
    parser.add_argument(
        "--hue-jitter",
        type=float,
        default=0.05,
        help="Hue jitter amount; shift sampled from [-x, x].",
    )
    parser.add_argument(
        "--gamma-jitter",
        type=float,
        default=0.2,
        help="Gamma jitter amount; factor sampled from [max(0.1, 1-x), 1+x].",
    )
    parser.add_argument(
        "--noise-std-max",
        type=float,
        default=0.03,
        help="Max stddev for additive Gaussian noise sampled in [0, x].",
    )
    parser.add_argument(
        "--blur-prob",
        type=float,
        default=0.0,
        help="Probability of applying Gaussian blur per image.",
    )
    parser.add_argument(
        "--blur-sigma-max",
        type=float,
        default=0.0,
        help="Max sigma for Gaussian blur (if <= 0, blur is disabled).",
    )
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=5,
        help="Gaussian blur kernel size (odd integer >= 3).",
    )
    namespace = parser.parse_args()
    return TrainConfig(**vars(namespace))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def split_samples(
    samples: list[StereoSample], val_fraction: float, seed: int
) -> tuple[list[StereoSample], list[StereoSample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1), got: {val_fraction}")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    if val_fraction == 0.0:
        return shuffled, []

    val_count = int(len(shuffled) * val_fraction)
    val_count = max(val_count, 1)
    if val_count >= len(shuffled):
        raise ValueError(
            "Validation set consumes all data. Reduce --val-fraction or provide more samples."
        )
    train_samples = shuffled[:-val_count]
    val_samples = shuffled[-val_count:]
    return train_samples, val_samples


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


def log_epoch_previews(
    model: StereoUNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    epoch: int,
    preview_root: Path,
) -> int:
    previews_dir = preview_root / f"epoch_{epoch:04d}"
    previews_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    preview_written = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            preds = model(inputs)

            for inner_index in range(inputs.shape[0]):
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

    if was_training:
        model.train()

    return preview_written


def run_epoch(
    model: StereoUNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: AdamW | None = None,
    global_step: int = 0,
    log_every_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    is_training = optimizer is not None
    model.train(is_training)

    total_nll = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_sigma = 0.0
    total_valid_pixels = 0

    interval_nll = 0.0
    interval_abs_error = 0.0
    interval_sq_error = 0.0
    interval_sigma = 0.0
    interval_valid_pixels = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        if is_training:
            global_step += 1

        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions, logvar = model(inputs, return_uncertainty=True)
            mask = valid_mask & torch.isfinite(targets)
            valid_count = int(mask.sum().item())
            if valid_count == 0:
                continue

            diff = predictions[mask] - targets[mask]
            abs_diff = diff.abs()
            masked_logvar = logvar[mask]

            # Heteroscedastic Laplace-style NLL with predicted per-pixel uncertainty.
            nll = abs_diff * torch.exp(-masked_logvar) + masked_logvar
            loss = nll.mean()
            if is_training:
                loss.backward()
                optimizer.step()

        diff_detached = diff.detach()
        nll_detached = nll.detach()
        sigma_detached = torch.exp(0.5 * masked_logvar.detach())
        total_valid_pixels += valid_count
        total_nll += float(nll_detached.sum().item())
        total_abs_error += float(diff_detached.abs().sum().item())
        total_sq_error += float(diff_detached.pow(2).sum().item())
        total_sigma += float(sigma_detached.sum().item())
        interval_nll += float(nll_detached.sum().item())
        interval_abs_error += float(diff_detached.abs().sum().item())
        interval_sq_error += float(diff_detached.pow(2).sum().item())
        interval_sigma += float(sigma_detached.sum().item())
        interval_valid_pixels += valid_count
        progress.set_postfix(
            {
                "mae": f"{diff_detached.abs().mean().item():.4f}",
                "nll": f"{nll_detached.mean().item():.4f}",
            }
        )

        if (
            is_training
            and log_every_batches is not None
            and log_every_batches > 0
            and global_step % log_every_batches == 0
            and interval_valid_pixels > 0
        ):
            mlflow.log_metrics(
                {
                    "train_loss_step": interval_nll / interval_valid_pixels,
                    "train_nll_step": interval_nll / interval_valid_pixels,
                    "train_mae_step": interval_abs_error / interval_valid_pixels,
                    "train_rmse_step": math.sqrt(
                        interval_sq_error / interval_valid_pixels
                    ),
                    "train_sigma_step": interval_sigma / interval_valid_pixels,
                },
                step=global_step,
            )
            interval_nll = 0.0
            interval_abs_error = 0.0
            interval_sq_error = 0.0
            interval_sigma = 0.0
            interval_valid_pixels = 0

    if total_valid_pixels == 0:
        raise RuntimeError("No valid target pixels found for this epoch.")

    if is_training and interval_valid_pixels > 0:
        mlflow.log_metrics(
            {
                "train_loss_step": interval_nll / interval_valid_pixels,
                "train_nll_step": interval_nll / interval_valid_pixels,
                "train_mae_step": interval_abs_error / interval_valid_pixels,
                "train_rmse_step": math.sqrt(interval_sq_error / interval_valid_pixels),
                "train_sigma_step": interval_sigma / interval_valid_pixels,
            },
            step=global_step,
        )

    nll_mean = total_nll / total_valid_pixels
    mae = total_abs_error / total_valid_pixels
    rmse = math.sqrt(total_sq_error / total_valid_pixels)
    sigma_mean = total_sigma / total_valid_pixels
    return (
        {
            "loss": nll_mean,
            "nll": nll_mean,
            "mae": mae,
            "rmse": rmse,
            "sigma": sigma_mean,
        },
        global_step,
    )


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: StereoUNet,
    optimizer: AdamW,
    args: TrainConfig,
    metrics: dict[str, float],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": asdict(args),
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)


def to_mlflow_params(
    args: TrainConfig, train_samples: int, val_samples: int, model: StereoUNet
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "height": args.height,
        "width": args.width,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": args.device,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "augment": args.augment,
        "uncertainty_head": True,
        "loss": "heteroscedastic_l1_nll",
        "mlflow_train_log_every_batches": MLFLOW_TRAIN_LOG_EVERY_BATCHES,
    }
    if args.augment:
        params["brightness_jitter"] = args.brightness_jitter
        params["contrast_jitter"] = args.contrast_jitter
        params["saturation_jitter"] = args.saturation_jitter
        params["hue_jitter"] = args.hue_jitter
        params["gamma_jitter"] = args.gamma_jitter
        params["noise_std_max"] = args.noise_std_max
        params["blur_prob"] = args.blur_prob
        params["blur_sigma_max"] = args.blur_sigma_max
        params["blur_kernel_size"] = args.blur_kernel_size
    if args.cache_root:
        params["cache_root"] = str(Path(args.cache_root).expanduser())
    params["require_cache"] = args.require_cache
    if args.max_samples > 0:
        params["max_samples"] = args.max_samples
    return params


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    all_samples = discover_samples(args.dataset_root)
    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
    if len(all_samples) < 2:
        raise ValueError("Need at least two samples to create train/validation splits.")

    train_samples, val_samples = split_samples(
        all_samples, args.val_fraction, args.seed
    )
    print(
        f"Discovered {len(all_samples)} samples: train={len(train_samples)}, val={len(val_samples)}"
    )

    image_size = (args.height, args.width)
    train_dataset = FoundationStereoDataset(
        train_samples,
        image_size=image_size,
        augment=args.augment,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        saturation_jitter=args.saturation_jitter,
        hue_jitter=args.hue_jitter,
        gamma_jitter=args.gamma_jitter,
        noise_std_max=args.noise_std_max,
        blur_prob=args.blur_prob,
        blur_sigma_max=args.blur_sigma_max,
        blur_kernel_size=args.blur_kernel_size,
        cache_root=args.cache_root,
        require_cache=args.require_cache,
    )
    val_dataset = (
        FoundationStereoDataset(
            val_samples,
            image_size=image_size,
            cache_root=args.cache_root,
            require_cache=args.require_cache,
        )
        if val_samples
        else None
    )

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    preview_source_samples = val_samples if val_samples else train_samples
    preview_split_name = "val" if val_samples else "train"
    preview_count = min(MLFLOW_PREVIEW_SAMPLES, len(preview_source_samples))
    preview_loader = None
    if preview_count > 0:
        preview_dataset = FoundationStereoDataset(
            preview_source_samples[:preview_count],
            image_size=image_size,
            cache_root=args.cache_root,
            require_cache=args.require_cache,
        )
        preview_loader = DataLoader(
            preview_dataset,
            batch_size=min(args.batch_size, preview_count),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        print(
            "MLflow previews: "
            f"logging {preview_count} fixed {preview_split_name} samples each epoch."
        )

    model = StereoUNet(in_channels=6, out_channels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.run_name):
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Failed to start MLflow run.")
        run_id = active_run.info.run_id

        output_dir = Path(args.output_dir).expanduser().resolve() / run_id
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        preview_root = output_dir / "mlflow_previews"
        preview_root.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        config_path.write_text(json.dumps(asdict(args), indent=2), encoding="utf-8")

        mlflow.log_params(
            to_mlflow_params(args, len(train_samples), len(val_samples), model)
        )
        mlflow.log_artifact(str(config_path), artifact_path="config")

        best_val_mae = float("inf")
        best_epoch = -1
        best_checkpoint = checkpoints_dir / "best.pt"
        last_checkpoint = checkpoints_dir / "last.pt"
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_metrics, global_step = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                global_step=global_step,
                log_every_batches=MLFLOW_TRAIN_LOG_EVERY_BATCHES,
            )
            if val_loader is not None:
                val_metrics, _ = run_epoch(model, val_loader, device, optimizer=None)
            else:
                val_metrics = train_metrics

            epoch_metrics = {
                "train_loss": train_metrics["loss"],
                "train_nll": train_metrics["nll"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_sigma": train_metrics["sigma"],
                "epoch_seconds": time.time() - start_time,
            }
            if val_loader is not None:
                epoch_metrics["val_loss"] = val_metrics["loss"]
                epoch_metrics["val_nll"] = val_metrics["nll"]
                epoch_metrics["val_mae"] = val_metrics["mae"]
                epoch_metrics["val_rmse"] = val_metrics["rmse"]
                epoch_metrics["val_sigma"] = val_metrics["sigma"]
            mlflow.log_metrics(epoch_metrics, step=epoch)

            if preview_loader is not None:
                log_epoch_previews(
                    model=model,
                    loader=preview_loader,
                    device=device,
                    epoch=epoch,
                    preview_root=preview_root,
                )
                mlflow.log_artifacts(
                    str(preview_root / f"epoch_{epoch:04d}"),
                    artifact_path=f"previews/epoch_{epoch:04d}",
                )

            save_checkpoint(
                last_checkpoint, epoch, model, optimizer, args, epoch_metrics
            )
            candidate_metric = val_metrics["mae"]
            if candidate_metric < best_val_mae:
                best_val_mae = candidate_metric
                best_epoch = epoch
                save_checkpoint(
                    best_checkpoint, epoch, model, optimizer, args, epoch_metrics
                )

            if val_loader is not None:
                print(
                    "Epoch "
                    f"{epoch}/{args.epochs}: "
                    f"train_mae={train_metrics['mae']:.4f}, val_mae={val_metrics['mae']:.4f}, "
                    f"train_rmse={train_metrics['rmse']:.4f}, val_rmse={val_metrics['rmse']:.4f}"
                )
            else:
                print(
                    "Epoch "
                    f"{epoch}/{args.epochs}: "
                    f"train_mae={train_metrics['mae']:.4f}, train_rmse={train_metrics['rmse']:.4f}"
                )

        mlflow.set_tag("best_epoch", best_epoch)
        mlflow.set_tag("best_val_mae", best_val_mae)
        mlflow.log_artifact(str(last_checkpoint), artifact_path="checkpoints")
        mlflow.log_artifact(str(best_checkpoint), artifact_path="checkpoints")

        print(f"MLflow run: {run_id}")
        print(f"Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
        print(f"Checkpoints saved to: {checkpoints_dir}")
        print(
            "Launch MLflow UI with: "
            f"uv run mlflow ui --backend-store-uri {args.mlflow_tracking_uri}"
        )


if __name__ == "__main__":
    main()
