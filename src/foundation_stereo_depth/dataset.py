from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class StereoSample:
    left_rgb_path: Path
    right_rgb_path: Path
    disparity_path: Path


def depth_uint8_decoding(depth_uint8: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    depth_uint8 = depth_uint8.astype(np.float32)
    out = depth_uint8[..., 0] * 255.0 * 255.0 + depth_uint8[..., 1] * 255.0 + depth_uint8[..., 2]
    return out / scale


def _resolve_frame_path(frame_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = frame_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_samples(dataset_root: str | Path) -> list[StereoSample]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: list[StereoSample] = []
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for scene_dir in scene_dirs:
        left_rgb_dir = scene_dir / "dataset" / "data" / "left" / "rgb"
        right_rgb_dir = scene_dir / "dataset" / "data" / "right" / "rgb"
        disparity_dir = scene_dir / "dataset" / "data" / "left" / "disparity"

        if not (left_rgb_dir.exists() and right_rgb_dir.exists() and disparity_dir.exists()):
            continue

        for disparity_path in sorted(disparity_dir.glob("*.png")):
            stem = disparity_path.stem
            left_path = _resolve_frame_path(left_rgb_dir, stem)
            right_path = _resolve_frame_path(right_rgb_dir, stem)
            if left_path is None or right_path is None:
                continue
            samples.append(StereoSample(left_path, right_path, disparity_path))
    return samples


class FoundationStereoDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Iterable[StereoSample],
        image_size: tuple[int, int] = (240, 320),
        augment: bool = False,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        hue_jitter: float = 0.0,
        noise_std_max: float = 0.0,
        blur_prob: float = 0.0,
        blur_sigma_max: float = 0.0,
        blur_kernel_size: int = 5,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.hue_jitter = hue_jitter
        self.noise_std_max = noise_std_max
        self.blur_prob = blur_prob
        self.blur_sigma_max = blur_sigma_max
        self.blur_kernel_size = blur_kernel_size

        if not 0.0 <= self.blur_prob <= 1.0:
            raise ValueError(f"blur_prob must be in [0, 1], got {self.blur_prob}")
        if self.blur_kernel_size < 3 or self.blur_kernel_size % 2 == 0:
            raise ValueError(f"blur_kernel_size must be odd and >= 3, got {self.blur_kernel_size}")
        if len(self.samples) == 0:
            raise ValueError("No samples were provided.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        rgb = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return tensor

    def _load_disparity(self, path: Path) -> torch.Tensor:
        disparity_uint8 = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        disparity = depth_uint8_decoding(disparity_uint8)
        original_width = disparity.shape[1]
        tensor = torch.from_numpy(disparity).unsqueeze(0)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Disparity is measured in horizontal pixels, so resizing requires
        # value scaling by the horizontal resize factor.
        resized_width = self.image_size[1]
        width_scale = resized_width / float(original_width)
        tensor = tensor * width_scale
        return tensor

    def _sample_jitter_factor(self, jitter: float) -> float:
        if jitter <= 0.0:
            return 1.0
        low = max(0.0, 1.0 - jitter)
        high = 1.0 + jitter
        return float(torch.empty(1).uniform_(low, high).item())

    def _sample_hue_shift(self) -> float:
        if self.hue_jitter <= 0.0:
            return 0.0
        return float(torch.empty(1).uniform_(-self.hue_jitter, self.hue_jitter).item())

    def _sample_noise_std(self) -> float:
        if self.noise_std_max <= 0.0:
            return 0.0
        return float(torch.empty(1).uniform_(0.0, self.noise_std_max).item())

    def _should_apply_blur(self) -> bool:
        if self.blur_prob <= 0.0 or self.blur_sigma_max <= 0.0:
            return False
        return bool(torch.rand(1).item() < self.blur_prob)

    def _sample_blur_sigma(self) -> float:
        sigma_min = 0.1
        sigma_max = max(self.blur_sigma_max, sigma_min)
        return float(torch.empty(1).uniform_(sigma_min, sigma_max).item())

    def _augment_rgb(self, image: torch.Tensor) -> torch.Tensor:
        image = TF.adjust_brightness(image, self._sample_jitter_factor(self.brightness_jitter))
        image = TF.adjust_contrast(image, self._sample_jitter_factor(self.contrast_jitter))
        image = TF.adjust_hue(image, self._sample_hue_shift())
        if self._should_apply_blur():
            sigma = self._sample_blur_sigma()
            image = TF.gaussian_blur(
                image,
                kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
                sigma=[sigma, sigma],
            )
        noise_std = self._sample_noise_std()
        if noise_std > 0.0:
            image = image + torch.randn_like(image) * noise_std
        return image.clamp_(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        left = self._load_rgb(sample.left_rgb_path)
        right = self._load_rgb(sample.right_rgb_path)
        if self.augment:
            left = self._augment_rgb(left)
            right = self._augment_rgb(right)
        target = self._load_disparity(sample.disparity_path)
        stereo_input = torch.cat([left, right], dim=0)
        valid_mask = target > 0.0
        return {
            "input": stereo_input,
            "target": target,
            "valid_mask": valid_mask,
        }
