import argparse
from pathlib import Path

import cv2
import numpy as np
from live_camera.camera_setup import (
    CameraConfig,
    drop_frames,
    log_camera_info,
    open_camera,
    warmup_cameras,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live stereo depth estimation.")
    parser.add_argument("--left", type=int, required=True, help="Left camera index.")
    parser.add_argument("--right", type=int, required=True, help="Right camera index.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration/stereo_calib.npz"),
        help="Calibration file from calibrate.py",
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
    parser.add_argument(
        "--buffer-size", type=int, default=1, help="Capture queue size."
    )
    parser.add_argument(
        "--warmup-frames", type=int, default=20, help="Initial frames to discard."
    )
    parser.add_argument(
        "--drop-frames",
        type=int,
        default=1,
        help="Extra frames to drop each loop to reduce latency.",
    )
    parser.add_argument(
        "--min-disparity", type=int, default=0, help="SGBM min disparity."
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=16 * 8,
        help="SGBM disparity range, multiple of 16.",
    )
    parser.add_argument(
        "--block-size", type=int, default=7, help="SGBM block size (odd)."
    )
    parser.add_argument(
        "--center-window",
        type=int,
        default=15,
        help="Center patch size for distance readout.",
    )
    return parser.parse_args()


def build_matcher(min_disp: int, num_disp: int, block_size: int) -> cv2.StereoSGBM:
    cn = 1
    p1 = 8 * cn * block_size * block_size
    p2 = 32 * cn * block_size * block_size
    return cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def main() -> None:
    args = parse_args()
    if not args.calibration.exists():
        raise FileNotFoundError(f"Calibration file not found: {args.calibration}")
    if args.num_disparities % 16 != 0:
        raise ValueError("--num-disparities must be a multiple of 16.")
    if args.block_size % 2 == 0 or args.block_size < 3:
        raise ValueError("--block-size must be odd and >= 3.")

    data = np.load(args.calibration)
    mtx_l = data["mtx_l"]
    dist_l = data["dist_l"]
    mtx_r = data["mtx_r"]
    dist_r = data["dist_r"]
    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]
    Q = data["Q"]
    image_size_values = data["image_size"].tolist()
    image_size = (int(image_size_values[0]), int(image_size_values[1]))

    map_l_1, map_l_2 = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map_r_1, map_r_2 = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
    )

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

    matcher = build_matcher(args.min_disparity, args.num_disparities, args.block_size)
    print("Running live depth. Press q or Esc to quit.")

    while True:
        drop_frames((cap_l, cap_r), config.drop_frames)

        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            continue

        left_size = (frame_l.shape[1], frame_l.shape[0])
        right_size = (frame_r.shape[1], frame_r.shape[0])
        if left_size != image_size or right_size != image_size:
            raise RuntimeError(
                f"Capture size mismatch. Expected calibration size={image_size}, "
                f"left={left_size}, right={right_size}. Reconfigure camera mode "
                "or recalibrate at the active resolution."
            )

        rect_l = cv2.remap(frame_l, map_l_1, map_l_2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map_r_1, map_r_2, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        disparity = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        disparity[disparity <= 0.0] = np.nan

        points_3d = cv2.reprojectImageTo3D(np.nan_to_num(disparity, nan=0.0), Q)
        z = points_3d[:, :, 2]
        z[~np.isfinite(disparity)] = np.nan

        h, w = z.shape
        cx, cy = w // 2, h // 2
        half = max(1, args.center_window // 2)
        patch = z[cy - half : cy + half + 1, cx - half : cx + half + 1]
        dist_m = np.nanmedian(patch)

        disp_vis = np.nan_to_num(disparity, nan=0.0)
        disp_vis_out = np.empty_like(disp_vis)
        disp_vis = cv2.normalize(disp_vis, disp_vis_out, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

        cv2.drawMarker(disp_vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        if np.isfinite(dist_m):
            text = f"center depth: {dist_m:.3f} m"
        else:
            text = "center depth: n/a"
        cv2.putText(
            disp_vis, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        cv2.imshow("Left Camera (Rectified)", rect_l)
        cv2.imshow("Right Camera (Rectified)", rect_r)
        cv2.imshow("Disparity / Depth", disp_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
