"""Dataclasses for YOLO labels, detections, and export records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class YoloLabel:
    """One YOLO normalized label row."""

    class_id: int
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float


@dataclass
class Detection2D:
    """Common 2D detection record used after YOLO inference."""

    scene_id: int
    scene_name: str
    split: str
    camera_id: str
    frame_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]
    source: str


@dataclass
class YoloExportRecord:
    """One exported image/label pair in a YOLO dataset."""

    image_path: Path
    label_path: Path
    scene_name: str
    split: str
    camera_id: str
    frame_id: int
    num_objects: int

