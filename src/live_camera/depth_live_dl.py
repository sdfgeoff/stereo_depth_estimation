from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from camera_setup import CameraConfig, drop_frames, log_camera_info, open_camera, warmup_cameras


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foundation_stereo_depth.model import StereoUNet


COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live stereo depth estimation using the trained deep learning model."
    )
    parser.add_argument("--left", type=int, required=True, help="Left camera index.")
    parser.add_argument("--right", type=int, required=True, help="Right camera index.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (for example: outputs/<run_id>/checkpoints/last.pt).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id under --output-dir when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Training output directory containing run subdirectories.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="last.pt",
        help="Checkpoint filename inside outputs/<run_id>/checkpoints/.",
    )
    parser.add_argument(
        "--watch-checkpoint",
        action="store_true",
        help="Reload checkpoint automatically when file timestamp changes.",
    )
    parser.add_argument(
        "--checkpoint-poll-sec",
        type=float,
        default=2.0,
        help="How often to check checkpoint updates in watch mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Inference device: "cpu", "cuda", or explicit torch device string.',
    )
    parser.add_argument("--cpu-threads", type=int, default=4, help="Torch CPU thread count.")
    parser.add_argument(
        "--model-width",
        type=int,
        default=320,
        help="Model input width.",
    )
    parser.add_argument(
        "--model-height",
        type=int,
        default=240,
        help="Model input height.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration/stereo_calib.npz"),
        help="Calibration file from calibrate.py.",
    )
    parser.add_argument(
        "--no-rectify",
        action="store_true",
        help="Disable undistortion/rectification even if calibration exists.",
    )
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS request.")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="Requested pixel format (for example: MJPG, YUYV).",
    )
    parser.add_argument("--buffer-size", type=int, default=1, help="Capture queue size.")
    parser.add_argument("--warmup-frames", type=int, default=20, help="Initial frames to discard.")
    parser.add_argument(
        "--drop-frames",
        type=int,
        default=1,
        help="Extra frames to drop each loop to reduce latency.",
    )
    parser.add_argument(
        "--center-window",
        type=int,
        default=15,
        help="Center patch size for readout.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        choices=sorted(COLORMAPS.keys()),
        help="Colormap for disparity visualization.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="Optional temporal smoothing in [0,1], 0 disables smoothing.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    output_dir = args.output_dir.expanduser().resolve()
    if args.run_id:
        checkpoint = output_dir / args.run_id / "checkpoints" / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    candidates = []
    if output_dir.exists():
        for run_dir in output_dir.iterdir():
            candidate = run_dir / "checkpoints" / args.checkpoint_name
            if candidate.exists():
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint {args.checkpoint_name} found under {output_dir}. "
            "Pass --checkpoint or --run-id."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_checkpoint(model: StereoUNet, checkpoint_path: Path, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = int(checkpoint.get("epoch", -1))
    else:
        state_dict = checkpoint
        epoch = -1

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return epoch


def preprocess_rgb(frame_bgr: np.ndarray, model_size: tuple[int, int]) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, model_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    return tensor


def colorize_disparity(disparity: np.ndarray, colormap: int) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0.0)
    if not np.any(valid):
        normalized = np.zeros(disparity.shape, dtype=np.uint8)
    else:
        values = disparity[valid]
        lo = float(np.percentile(values, 5))
        hi = float(np.percentile(values, 95))
        scale = max(hi - lo, 1e-6)
        normalized_float = np.clip((disparity - lo) / scale, 0.0, 1.0)
        normalized = (normalized_float * 255.0).astype(np.uint8)
        normalized[~valid] = 0
    return cv2.applyColorMap(normalized, colormap)


def maybe_load_rectification(
    calibration_path: Path, use_rectification: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]] | None:
    if not use_rectification:
        return None
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}. "
            "Use --no-rectify or provide a valid calibration file."
        )

    data = np.load(calibration_path)
    mtx_l = data["mtx_l"]
    dist_l = data["dist_l"]
    mtx_r = data["mtx_r"]
    dist_r = data["dist_r"]
    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]
    image_size = tuple(int(v) for v in data["image_size"].tolist())

    map_l_1, map_l_2 = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map_r_1, map_r_2 = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
    )
    return map_l_1, map_l_2, map_r_1, map_r_2, image_size


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in [0, 1].")

    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device(args.device)
    model_size = (args.model_width, args.model_height)
    checkpoint_path = resolve_checkpoint_path(args)

    model = StereoUNet(in_channels=6, out_channels=1).to(device)
    loaded_epoch = load_checkpoint(model, checkpoint_path, device)
    checkpoint_mtime_ns = checkpoint_path.stat().st_mtime_ns
    next_poll_time = time.time() + args.checkpoint_poll_sec

    rectification = maybe_load_rectification(args.calibration, use_rectification=not args.no_rectify)

    config = CameraConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        buffer_size=args.buffer_size,
        warmup_frames=args.warmup_frames,
        drop_frames=args.drop_frames,
    )
    cap_l = open_camera(args.left, config)
    cap_r = open_camera(args.right, config)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError("Could not open both cameras.")

    for label, cap in (("Left", cap_l), ("Right", cap_r)):
        log_camera_info(label, cap)
    warmup_cameras((cap_l, cap_r), config.warmup_frames)

    print(f"Model checkpoint: {checkpoint_path}")
    if loaded_epoch >= 0:
        print(f"Loaded epoch: {loaded_epoch}")
    print(f"Running live DL depth on device={device}. Press q or Esc to quit.")

    smoothed = None
    previous_time = time.time()

    while True:
        drop_frames((cap_l, cap_r), config.drop_frames)

        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            continue

        if rectification is not None:
            map_l_1, map_l_2, map_r_1, map_r_2, image_size = rectification
            left_size = (frame_l.shape[1], frame_l.shape[0])
            right_size = (frame_r.shape[1], frame_r.shape[0])
            if left_size != image_size or right_size != image_size:
                raise RuntimeError(
                    f"Capture size mismatch. Expected calibration size={image_size}, "
                    f"left={left_size}, right={right_size}."
                )
            view_l = cv2.remap(frame_l, map_l_1, map_l_2, cv2.INTER_LINEAR)
            view_r = cv2.remap(frame_r, map_r_1, map_r_2, cv2.INTER_LINEAR)
        else:
            view_l = frame_l
            view_r = frame_r

        if args.watch_checkpoint and time.time() >= next_poll_time:
            new_mtime_ns = checkpoint_path.stat().st_mtime_ns
            if new_mtime_ns != checkpoint_mtime_ns:
                try:
                    loaded_epoch = load_checkpoint(model, checkpoint_path, device)
                    checkpoint_mtime_ns = new_mtime_ns
                    print(f"Reloaded checkpoint at epoch {loaded_epoch}.")
                except Exception as exc:
                    print(f"Checkpoint reload skipped: {exc}")
            next_poll_time = time.time() + args.checkpoint_poll_sec

        left_tensor = preprocess_rgb(view_l, model_size)
        right_tensor = preprocess_rgb(view_r, model_size)
        model_input = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(device)

        with torch.inference_mode():
            prediction = model(model_input)[0, 0].detach().cpu().numpy().astype(np.float32)

        if args.ema_alpha > 0.0:
            if smoothed is None:
                smoothed = prediction
            else:
                smoothed = args.ema_alpha * prediction + (1.0 - args.ema_alpha) * smoothed
            disparity = smoothed
        else:
            disparity = prediction

        h, w = disparity.shape
        cx, cy = w // 2, h // 2
        half = max(1, args.center_window // 2)
        y0 = max(0, cy - half)
        y1 = min(h, cy + half + 1)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half + 1)
        patch = disparity[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch) & (patch > 0.0)]
        center_readout = float(np.median(patch)) if patch.size > 0 else float("nan")

        depth_vis = colorize_disparity(disparity, COLORMAPS[args.colormap])
        depth_vis = cv2.resize(depth_vis, (view_l.shape[1], view_l.shape[0]), interpolation=cv2.INTER_LINEAR)

        marker_x = int(cx * view_l.shape[1] / max(w, 1))
        marker_y = int(cy * view_l.shape[0] / max(h, 1))
        cv2.drawMarker(depth_vis, (marker_x, marker_y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

        now = time.time()
        fps = 1.0 / max(now - previous_time, 1e-6)
        previous_time = now

        readout_text = (
            f"center disparity: {center_readout:.3f}" if np.isfinite(center_readout) else "center disparity: n/a"
        )
        info_text = f"fps: {fps:.1f} | model: {args.model_width}x{args.model_height}"
        epoch_text = f"checkpoint epoch: {loaded_epoch if loaded_epoch >= 0 else 'unknown'}"
        cv2.putText(depth_vis, readout_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(depth_vis, info_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(depth_vis, epoch_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Left Camera (Rectified)" if rectification is not None else "Left Camera", view_l)
        cv2.imshow("Right Camera (Rectified)" if rectification is not None else "Right Camera", view_r)
        cv2.imshow("DL Disparity", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
