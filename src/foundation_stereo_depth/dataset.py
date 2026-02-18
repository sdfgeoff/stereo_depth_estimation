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
    out = (
        depth_uint8[..., 0] * 255.0 * 255.0
        + depth_uint8[..., 1] * 255.0
        + depth_uint8[..., 2]
    )
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

        if not (
            left_rgb_dir.exists() and right_rgb_dir.exists() and disparity_dir.exists()
        ):
            continue

        for disparity_path in sorted(disparity_dir.glob("*.png")):
            stem = disparity_path.stem
            left_path = _resolve_frame_path(left_rgb_dir, stem)
            right_path = _resolve_frame_path(right_rgb_dir, stem)
            if left_path is None or right_path is None:
                continue
            samples.append(StereoSample(left_path, right_path, disparity_path))
    return samples


def sample_cache_relpath(sample: StereoSample) -> Path:
    left_parts = sample.left_rgb_path.parts
    scene_name = sample.left_rgb_path.parents[4].name
    if "dataset" in left_parts:
        dataset_index = left_parts.index("dataset")
        if dataset_index > 0:
            scene_name = left_parts[dataset_index - 1]
    return Path(scene_name) / f"{sample.disparity_path.stem}.npz"


class FoundationStereoDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Iterable[StereoSample],
        image_size: tuple[int, int] = (240, 320),
        augment: bool = False,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        saturation_jitter: float = 0.0,
        hue_jitter: float = 0.0,
        gamma_jitter: float = 0.0,
        noise_std_max: float = 0.0,
        blur_prob: float = 0.0,
        blur_sigma_max: float = 0.0,
        blur_kernel_size: int = 5,
        cache_root: str | Path | None = None,
        require_cache: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.saturation_jitter = saturation_jitter
        self.hue_jitter = hue_jitter
        self.gamma_jitter = gamma_jitter
        self.noise_std_max = noise_std_max
        self.blur_prob = blur_prob
        self.blur_sigma_max = blur_sigma_max
        self.blur_kernel_size = blur_kernel_size
        self.cache_root = (
            Path(cache_root).expanduser().resolve() if cache_root is not None else None
        )
        self.require_cache = require_cache

        if not 0.0 <= self.blur_prob <= 1.0:
            raise ValueError(f"blur_prob must be in [0, 1], got {self.blur_prob}")
        if self.blur_kernel_size < 3 or self.blur_kernel_size % 2 == 0:
            raise ValueError(
                f"blur_kernel_size must be odd and >= 3, got {self.blur_kernel_size}"
            )
        if self.saturation_jitter < 0.0:
            raise ValueError(
                f"saturation_jitter must be >= 0, got {self.saturation_jitter}"
            )
        if self.gamma_jitter < 0.0:
            raise ValueError(f"gamma_jitter must be >= 0, got {self.gamma_jitter}")
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

    def _load_cached_sample(
        self, cache_file: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        with np.load(cache_file) as cached:
            if not {"left", "right", "disparity"}.issubset(cached.files):
                return None

            left_np = cached["left"]
            right_np = cached["right"]
            disparity_np = cached["disparity"]

        if left_np.ndim != 3 or right_np.ndim != 3 or disparity_np.ndim != 2:
            return None
        if (
            left_np.shape[:2] != self.image_size
            or right_np.shape[:2] != self.image_size
        ):
            return None
        if disparity_np.shape != self.image_size:
            return None

        left = torch.from_numpy(left_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        right = torch.from_numpy(right_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        target = torch.from_numpy(disparity_np.astype(np.float32)).unsqueeze(0)
        return left, right, target

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

    def _sample_gamma_factor(self) -> float:
        if self.gamma_jitter <= 0.0:
            return 1.0
        low = max(0.1, 1.0 - self.gamma_jitter)
        high = max(low, 1.0 + self.gamma_jitter)
        return float(torch.empty(1).uniform_(low, high).item())

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
        image = TF.adjust_brightness(
            image, self._sample_jitter_factor(self.brightness_jitter)
        )
        image = TF.adjust_contrast(
            image, self._sample_jitter_factor(self.contrast_jitter)
        )
        image = TF.adjust_saturation(
            image, self._sample_jitter_factor(self.saturation_jitter)
        )
        image = TF.adjust_hue(image, self._sample_hue_shift())
        image = TF.adjust_gamma(image, gamma=self._sample_gamma_factor(), gain=1.0)
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
        left = None
        right = None
        target = None

        if self.cache_root is not None:
            cache_file = self.cache_root / sample_cache_relpath(sample)
            if cache_file.exists():
                loaded = self._load_cached_sample(cache_file)
                if loaded is not None:
                    left, right, target = loaded
                elif self.require_cache:
                    raise ValueError(
                        f"Cache entry is invalid or shape-mismatched for sample: {cache_file}"
                    )
            elif self.require_cache:
                raise FileNotFoundError(f"Required cache entry not found: {cache_file}")

        if left is None or right is None or target is None:
            left = self._load_rgb(sample.left_rgb_path)
            right = self._load_rgb(sample.right_rgb_path)
            target = self._load_disparity(sample.disparity_path)

        if self.augment:
            left = self._augment_rgb(left)
            right = self._augment_rgb(right)
        stereo_input = torch.cat([left, right], dim=0)
        valid_mask = target > 0.0
        return {
            "input": stereo_input,
            "target": target,
            "valid_mask": valid_mask,
        }
