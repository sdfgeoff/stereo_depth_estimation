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


class RectificationData:
    def __init__(
        self,
        map_l_1: np.ndarray,
        map_l_2: np.ndarray,
        map_r_1: np.ndarray,
        map_r_2: np.ndarray,
        image_size: tuple[int, int],
        focal_length_px: float,
        baseline_m: float | None,
    ) -> None:
        self.map_l_1 = map_l_1
        self.map_l_2 = map_l_2
        self.map_r_1 = map_r_1
        self.map_r_2 = map_r_2
        self.image_size = image_size
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m


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


def colorize_scalar_map(values_2d: np.ndarray, colormap: int) -> np.ndarray:
    valid = np.isfinite(values_2d) & (values_2d > 0.0)
    if not np.any(valid):
        normalized = np.zeros(values_2d.shape, dtype=np.uint8)
    else:
        values = values_2d[valid]
        lo = float(np.percentile(values, 2))
        hi = float(np.percentile(values, 98))
        scale = max(hi - lo, 1e-6)
        normalized_float = np.clip((values_2d - lo) / scale, 0.0, 1.0)
        normalized = (normalized_float * 255.0).astype(np.uint8)
        normalized[~valid] = 0
    return cv2.applyColorMap(normalized, colormap)


def maybe_load_rectification(
    calibration_path: Path, use_rectification: bool
) -> RectificationData | None:
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
    T = data["T"] if "T" in data else None
    image_size = tuple(int(v) for v in data["image_size"].tolist())

    map_l_1, map_l_2 = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map_r_1, map_r_2 = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
    )
    focal_length_px = float(P1[0, 0])
    baseline_m = estimate_baseline_m(P1=P1, P2=P2, T=T)
    return RectificationData(
        map_l_1,
        map_l_2,
        map_r_1,
        map_r_2,
        image_size,
        focal_length_px,
        baseline_m,
    )


def estimate_baseline_m(P1: np.ndarray | None, P2: np.ndarray | None, T: np.ndarray | None) -> float | None:
    baseline_m = None
    if P1 is not None and P2 is not None:
        focal_px = float(P1[0, 0])
        if np.isfinite(focal_px) and abs(focal_px) > 1e-9:
            tx = float(P2[0, 3])
            candidate = abs(-tx / focal_px)
            if np.isfinite(candidate) and candidate > 0.0:
                baseline_m = candidate
    if baseline_m is None and T is not None:
        t = np.asarray(T, dtype=np.float64).reshape(-1)
        if t.size >= 3:
            candidate = float(np.linalg.norm(t))
            if np.isfinite(candidate) and candidate > 0.0:
                baseline_m = candidate
    return baseline_m


def load_calibration_geometry(calibration_path: Path) -> tuple[float | None, float | None, int | None]:
    if not calibration_path.exists():
        return None, None, None

    with np.load(calibration_path) as data:
        P1 = data["P1"] if "P1" in data else None
        P2 = data["P2"] if "P2" in data else None
        T = data["T"] if "T" in data else None
        image_size = data["image_size"] if "image_size" in data else None
        if P1 is not None:
            focal_px = float(P1[0, 0])
        elif "mtx_l" in data:
            focal_px = float(data["mtx_l"][0, 0])
        else:
            focal_px = None

        baseline_m = estimate_baseline_m(P1=P1, P2=P2, T=T)

        if image_size is not None:
            calibration_width_px = int(np.asarray(image_size).reshape(-1)[0])
        else:
            calibration_width_px = None

    if focal_px is not None and (not np.isfinite(focal_px) or focal_px <= 0.0):
        focal_px = None
    return focal_px, baseline_m, calibration_width_px


def disparity_to_depth(disparity: np.ndarray, focal_length_px: float, baseline_m: float) -> np.ndarray:
    depth = np.full_like(disparity, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1e-6)
    depth[valid] = (focal_length_px * baseline_m) / disparity[valid]
    return depth


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

    calibration_focal_px, calibration_baseline_m, calibration_width_px = load_calibration_geometry(
        args.calibration
    )
    rectification = maybe_load_rectification(args.calibration, use_rectification=not args.no_rectify)
    if rectification is not None:
        calibration_focal_px = rectification.focal_length_px
        calibration_baseline_m = rectification.baseline_m
        calibration_width_px = rectification.image_size[0]

    focal_length_px_calib = calibration_focal_px
    baseline_m = calibration_baseline_m
    focal_length_px_model = None
    if focal_length_px_calib is not None and calibration_width_px is not None and calibration_width_px > 0:
        focal_length_px_model = focal_length_px_calib * (args.model_width / float(calibration_width_px))
    depth_enabled = baseline_m is not None and focal_length_px_model is not None

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
    if depth_enabled:
        print(
            "Depth conversion enabled: "
            f"baseline={baseline_m:.6f} m, "
            f"focal_calib={focal_length_px_calib:.2f} px, "
            f"focal_model={focal_length_px_model:.2f} px"
        )
        if rectification is None:
            print("Warning: running without rectification. Depth may be inaccurate unless inputs are pre-rectified.")
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
            map_l_1 = rectification.map_l_1
            map_l_2 = rectification.map_l_2
            map_r_1 = rectification.map_r_1
            map_r_2 = rectification.map_r_2
            image_size = rectification.image_size
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
        center_disparity = float(np.median(patch)) if patch.size > 0 else float("nan")

        if depth_enabled:
            depth_m = disparity_to_depth(disparity, float(focal_length_px_model), float(baseline_m))
            depth_patch = depth_m[y0:y1, x0:x1]
            depth_patch = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0.0)]
            center_depth_m = float(np.median(depth_patch)) if depth_patch.size > 0 else float("nan")
            vis_map = depth_m
            vis_title = "DL Depth (m)"
        else:
            center_depth_m = float("nan")
            vis_map = disparity
            vis_title = "DL Disparity"

        depth_vis = colorize_scalar_map(vis_map, COLORMAPS[args.colormap])
        depth_vis = cv2.resize(depth_vis, (view_l.shape[1], view_l.shape[0]), interpolation=cv2.INTER_LINEAR)

        marker_x = int(cx * view_l.shape[1] / max(w, 1))
        marker_y = int(cy * view_l.shape[0] / max(h, 1))
        cv2.drawMarker(depth_vis, (marker_x, marker_y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

        now = time.time()
        fps = 1.0 / max(now - previous_time, 1e-6)
        previous_time = now

        readout_text = (
            f"center disparity: {center_disparity:.3f}"
            if np.isfinite(center_disparity)
            else "center disparity: n/a"
        )
        if depth_enabled:
            if np.isfinite(center_depth_m):
                readout_text = f"{readout_text} | center depth: {center_depth_m:.3f} m"
            else:
                readout_text = f"{readout_text} | center depth: n/a"
        info_text = f"fps: {fps:.1f} | model: {args.model_width}x{args.model_height}"
        epoch_text = f"checkpoint epoch: {loaded_epoch if loaded_epoch >= 0 else 'unknown'}"
        cv2.putText(depth_vis, readout_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(depth_vis, info_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(depth_vis, epoch_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Left Camera (Rectified)" if rectification is not None else "Left Camera", view_l)
        cv2.imshow("Right Camera (Rectified)" if rectification is not None else "Right Camera", view_r)
        cv2.imshow(vis_title, depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
