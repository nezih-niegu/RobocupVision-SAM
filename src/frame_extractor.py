from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


@dataclass
class ExtractedFrame:
    timestamp_sec: float
    frame_index: int
    image: np.ndarray


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


def resize_if_needed(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames_at_interval(
    video_path: Path,
    interval_sec: float,
    max_image_side: int = 1280,
) -> list[ExtractedFrame]:
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[ExtractedFrame] = []
    frame_index = 0

    while frame_index * interval_sec <= info.duration_sec + 1e-6:
        timestamp = round(frame_index * interval_sec, 3)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_if_needed(rgb, max_image_side)
        frames.append(
            ExtractedFrame(
                timestamp_sec=timestamp,
                frame_index=frame_index,
                image=rgb,
            )
        )
        frame_index += 1

    cap.release()
    return frames


def save_frame_jpeg(image_rgb: np.ndarray, output_path: Path, quality: int = 90) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
