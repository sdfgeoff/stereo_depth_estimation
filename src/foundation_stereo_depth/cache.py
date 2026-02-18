from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from .dataset import (
    FoundationStereoDataset,
    discover_samples,
    sample_cache_relpath,
    save_cached_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a resized FoundationStereo cache for faster training I/O."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo",
        help="Path to raw FoundationStereo dataset root.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        required=True,
        help="Path to write cache files (prefer SSD).",
    )
    parser.add_argument("--height", type=int, default=240, help="Cached image height.")
    parser.add_argument("--width", type=int, default=320, help="Cached image width.")
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Optional cap on number of samples."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing cache entries."
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed (smaller files, slower build/read).",
    )
    return parser.parse_args()


def build_cache(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(dataset_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError(f"No samples discovered under: {dataset_root}")

    dataset = FoundationStereoDataset(
        samples=samples, image_size=(args.height, args.width), augment=False
    )
    written = 0
    skipped = 0
    started_at = time.time()
    for index, sample in enumerate(tqdm(samples, desc="Building cache", unit="sample")):
        cache_relpath = sample_cache_relpath(sample)
        cache_file = cache_root / cache_relpath
        if cache_file.exists() and not args.overwrite:
            skipped += 1
            continue

        cache_file.parent.mkdir(parents=True, exist_ok=True)

        item = dataset[index]
        stereo_input = item["input"]
        left = stereo_input[:3]
        right = stereo_input[3:6]
        disparity = item["target"]
        save_cached_sample(
            cache_file,
            left=left,
            right=right,
            target=disparity,
            compress=args.compress,
        )
        written += 1

    elapsed_sec = time.time() - started_at
    metadata = {
        "format_version": 1,
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "height": args.height,
        "width": args.width,
        "num_samples_total": len(samples),
        "num_written": written,
        "num_skipped": skipped,
        "compressed": bool(args.compress),
        "elapsed_seconds": elapsed_sec,
        "created_at_unix": time.time(),
    }
    (cache_root / "cache_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(
        "Cache build complete: "
        f"total={len(samples)} written={written} skipped={skipped} elapsed={elapsed_sec:.1f}s"
    )
    print(f"Metadata: {cache_root / 'cache_meta.json'}")


def main() -> None:
    args = parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
