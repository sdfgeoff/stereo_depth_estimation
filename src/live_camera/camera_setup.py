from dataclasses import dataclass
from typing import Iterable

import cv2


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    fourcc: str = "MJPG"
    buffer_size: int = 1
    warmup_frames: int = 20
    drop_frames: int = 1
    focus_value: float = 0.0


def decode_fourcc(value: float) -> str:
    int_value = int(value)
    return "".join(chr((int_value >> (8 * i)) & 0xFF) for i in range(4))


def configure_camera(cap: cv2.VideoCapture, config: CameraConfig) -> None:
    if len(config.fourcc) != 4:
        raise ValueError("--fourcc must be exactly 4 characters.")

    fourcc_builder = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_builder is None:
        fourcc_builder = cv2.VideoWriter.fourcc
    fourcc_code = int((fourcc_builder)(*config.fourcc.upper()))

    cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)

    # Keep exposure and white balance automatic.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    # Lock focus to a fixed value (0 is typically infinity on UVC webcams).
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, config.focus_value)


def open_camera(index: int, config: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    configure_camera(cap, config)
    return cap


def log_camera_info(label: str, cap: cv2.VideoCapture) -> None:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"{label} camera: {width}x{height} @ {fps:.1f} FPS, FOURCC={fourcc}")


def warmup_cameras(cameras: Iterable[cv2.VideoCapture], warmup_frames: int) -> None:
    for _ in range(max(0, warmup_frames)):
        for cap in cameras:
            cap.grab()


def drop_frames(cameras: Iterable[cv2.VideoCapture], frame_count: int) -> None:
    for _ in range(max(0, frame_count)):
        for cap in cameras:
            cap.grab()
