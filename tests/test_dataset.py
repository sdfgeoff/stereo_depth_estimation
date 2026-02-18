from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from foundation_stereo_depth.dataset import (
    FoundationStereoDataset,
    StereoSample,
    depth_uint8_decoding,
    load_cached_sample,
    sample_cache_relpath,
)


def _encode_disparity_to_rgb(disparity: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    values = np.round(disparity * scale).astype(np.int64)
    r = values // (255 * 255)
    remainder = values - r * (255 * 255)
    g = remainder // 255
    b = remainder - g * 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _write_rgb(path: Path, shape: tuple[int, int]) -> None:
    h, w = shape
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB").save(path)


def test_depth_uint8_decoding_round_trip() -> None:
    disparity = np.array([[0.0, 0.125, 1.25], [2.0, 3.5, 10.0]], dtype=np.float32)
    encoded = _encode_disparity_to_rgb(disparity)
    decoded = depth_uint8_decoding(encoded)
    np.testing.assert_allclose(decoded, disparity, atol=1e-3)


def test_disparity_resize_scales_with_output_width(tmp_path: Path) -> None:
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    disparity_path = tmp_path / "disp.png"

    original_h, original_w = 2, 4
    _write_rgb(left_path, (original_h, original_w))
    _write_rgb(right_path, (original_h, original_w))

    source_disparity = np.full((original_h, original_w), 1.5, dtype=np.float32)
    disparity_rgb = _encode_disparity_to_rgb(source_disparity)
    Image.fromarray(disparity_rgb, mode="RGB").save(disparity_path)

    sample = StereoSample(
        left_rgb_path=left_path,
        right_rgb_path=right_path,
        disparity_path=disparity_path,
    )
    dataset = FoundationStereoDataset([sample], image_size=(2, 8))
    item = dataset[0]

    target = item["target"].numpy()
    expected = np.full((1, 2, 8), 3.0, dtype=np.float32)
    np.testing.assert_allclose(target, expected, atol=1e-3)


def test_sample_cache_relpath_uses_scene_and_stem() -> None:
    sample = StereoSample(
        left_rgb_path=Path("/data/scene_01/dataset/data/left/rgb/000123.png"),
        right_rgb_path=Path("/data/scene_01/dataset/data/right/rgb/000123.png"),
        disparity_path=Path("/data/scene_01/dataset/data/left/disparity/000123.png"),
    )

    assert sample_cache_relpath(sample) == Path("scene_01/000123.npz")


def test_sample_cache_relpath_noncanonical_layout_uses_stable_misc_key() -> None:
    sample = StereoSample(
        left_rgb_path=Path("/tmp/left_view.png"),
        right_rgb_path=Path("/tmp/right_view.png"),
        disparity_path=Path("/tmp/disp_42.png"),
    )

    relpath = sample_cache_relpath(sample)
    assert relpath.parent == Path("misc")
    assert relpath.name.startswith("disp_42_")
    assert relpath.suffix == ".npz"
    assert relpath == sample_cache_relpath(sample)


def test_dataset_cache_read_through_writes_missing_entries(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    dataset_root.mkdir()
    cache_root.mkdir()

    left_path = dataset_root / "left.png"
    right_path = dataset_root / "right.png"
    disparity_path = dataset_root / "disp.png"

    _write_rgb(left_path, (2, 4))
    _write_rgb(right_path, (2, 4))

    source_disparity = np.full((2, 4), 1.25, dtype=np.float32)
    Image.fromarray(_encode_disparity_to_rgb(source_disparity), mode="RGB").save(
        disparity_path
    )

    sample = StereoSample(left_path, right_path, disparity_path)
    cache_file = cache_root / sample_cache_relpath(sample)
    assert not cache_file.exists()

    dataset = FoundationStereoDataset(
        [sample],
        image_size=(2, 4),
        cache_root=cache_root,
        require_cache=False,
    )

    first_item = dataset[0]
    assert cache_file.exists()

    second_item = dataset[0]
    np.testing.assert_allclose(
        first_item["target"].numpy(),
        second_item["target"].numpy(),
        atol=1e-3,
    )

    loaded = load_cached_sample(cache_file, (2, 4))
    assert loaded is not None
