import argparse
from pathlib import Path

import cv2
import numpy as np
from camera_setup import CameraConfig, drop_frames, log_camera_info, open_camera, warmup_cameras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate stereo USB cameras using a chessboard.")
    parser.add_argument("--left", type=int, required=True, help="Left camera index.")
    parser.add_argument("--right", type=int, required=True, help="Right camera index.")
    parser.add_argument("--rows", type=int, default=6, help="Inner chessboard corners per column.")
    parser.add_argument("--cols", type=int, default=9, help="Inner chessboard corners per row.")
    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Chessboard square size in meters (example: 0.024).",
    )
    parser.add_argument("--samples", type=int, default=25, help="Successful stereo pairs to collect.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration/stereo_calib.npz"),
        help="Output calibration file.",
    )
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS request.")
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
    return parser.parse_args()


def make_object_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= square_size
    return grid

def main() -> None:
    args = parse_args()
    pattern_size = (args.cols, args.rows)
    objp = make_object_points(args.rows, args.cols, args.square_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)

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
        raise RuntimeError("Could not open both cameras. Check indices with list_cameras.py.")

    for label, cap in (("Left", cap_l), ("Right", cap_r)):
        log_camera_info(label, cap)

    warmup_cameras((cap_l, cap_r), config.warmup_frames)

    obj_points = []
    img_points_l = []
    img_points_r = []
    image_size = None

    print("Calibration capture")
    print("  Space: capture pair when chessboard is found in both views")
    print("  Q or Esc: quit")
    print(f"Need {args.samples} valid pairs.")

    while len(obj_points) < args.samples:
        drop_frames((cap_l, cap_r), config.drop_frames)

        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            continue

        if frame_l.shape[:2] != frame_r.shape[:2]:
            left_size = (frame_l.shape[1], frame_l.shape[0])
            right_size = (frame_r.shape[1], frame_r.shape[0])
            raise RuntimeError(
                f"Camera frame sizes differ: left={left_size}, right={right_size}. "
                "Set a matching format/resolution on both cameras."
            )

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        image_size = gray_l.shape[::-1]

        found_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

        display_l = frame_l.copy()
        display_r = frame_r.copy()
        if found_l:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_l, pattern_size, corners_l, found_l)
        if found_r:
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_r, pattern_size, corners_r, found_r)

        combined = np.hstack([display_l, display_r])
        status = f"pairs {len(obj_points)}/{args.samples} | board L:{found_l} R:{found_r}"
        cv2.putText(combined, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 240, 50), 2)
        cv2.imshow("Stereo Calibration (left | right)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and found_l and found_r:
            obj_points.append(objp.copy())
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            print(f"Captured pair {len(obj_points)}/{args.samples}")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 8:
        raise RuntimeError("Not enough pairs for reliable calibration. Capture more samples.")
    if image_size is None:
        raise RuntimeError("No frames captured.")

    print("Running mono calibration...")
    rms_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, image_size, None, None)
    rms_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, image_size, None, None)
    print(f"Mono RMS left: {rms_l:.4f}, right: {rms_r:.4f}")

    flags = cv2.CALIB_FIX_INTRINSIC
    print("Running stereo calibration...")
    rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_l,
        img_points_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=criteria,
        flags=flags,
    )
    print(f"Stereo RMS: {rms_stereo:.4f}")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image_size=np.array(image_size),
        mtx_l=mtx_l,
        dist_l=dist_l,
        mtx_r=mtx_r,
        dist_r=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        rows=np.array(args.rows),
        cols=np.array(args.cols),
        square_size=np.array(args.square_size),
        stereo_rms=np.array(rms_stereo),
        mono_rms_l=np.array(rms_l),
        mono_rms_r=np.array(rms_r),
    )
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
