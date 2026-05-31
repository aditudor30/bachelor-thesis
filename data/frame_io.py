"""Lazy video frame I/O helpers.

These functions open a video only long enough to read metadata or one requested
frame. They never preload a full video into memory.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def list_video_files(videos_dir: Path) -> List[Path]:
    """List video files under a camera video directory."""
    directory = Path(videos_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    files = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def infer_camera_id_from_video_path(path: Path) -> str:
    """Infer camera id from a video filename stem."""
    return Path(path).stem


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Read one 0-based video frame and return it as RGB."""
    if frame_idx < 0:
        raise IndexError("frame_idx must be non-negative")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise IOError("Could not open video: %s" % video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise IOError("Could not read frame %d from video: %s" % (frame_idx, video_path))
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def safe_read_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Read one video frame, returning None instead of raising on failure."""
    try:
        return read_video_frame(video_path, frame_idx)
    except Exception:
        return None


def get_video_frame_count(video_path: Path) -> Optional[int]:
    """Return video frame count when OpenCV can report it."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        value = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if value <= 0:
            return None
        return value
    finally:
        capture.release()


def get_video_fps(video_path: Path) -> Optional[float]:
    """Return video FPS when OpenCV can report it."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        value = float(capture.get(cv2.CAP_PROP_FPS))
        if value <= 0.0:
            return None
        return value
    finally:
        capture.release()


def get_video_resolution(video_path: Path) -> Optional[Tuple[int, int]]:
    """Return video resolution as (width, height) when available."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return None
        return (width, height)
    finally:
        capture.release()

