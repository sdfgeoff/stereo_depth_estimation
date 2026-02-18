from __future__ import annotations

import random
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from PIL import Image

SampleT = TypeVar("SampleT")


def split_samples(
    samples: list[SampleT],
    val_fraction: float,
    seed: int,
    *,
    require_non_empty_train: bool = True,
) -> tuple[list[SampleT], list[SampleT]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1), got: {val_fraction}")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    if val_fraction == 0.0:
        return shuffled, []

    val_count = max(int(len(shuffled) * val_fraction), 1)
    if require_non_empty_train and val_count >= len(shuffled):
        raise ValueError(
            "Validation set consumes all data. Reduce --val-fraction or provide more samples."
        )
    val_count = min(val_count, len(shuffled))

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


def save_preview_montage(
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

