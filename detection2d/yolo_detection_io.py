"""CSV and MOT-like I/O for YOLO 2D detections."""

import csv
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.detection2d.yolo_label_utils import xyxy_to_xywh
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D


CSV_FIELDS = [
    "scene_id",
    "scene_name",
    "split",
    "camera_id",
    "frame_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "w",
    "h",
    "source",
]


def write_detections_csv(detections: List[Detection2D], path: Path) -> None:
    """Write detections to a common CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for det in detections:
            writer.writerow(_detection_to_row(det))


def read_detections_csv(path: Path) -> List[Detection2D]:
    """Read detections from the common CSV format."""
    detections = []
    if not path.exists():
        return detections
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            detections.append(_row_to_detection(row))
    return detections


def detection_to_mot_row(det: Detection2D) -> List[Any]:
    """Convert detection to MOT-like row without track id."""
    x, y, width, height = xyxy_to_xywh(det.bbox_xyxy)
    return [
        int(det.frame_id),
        -1,
        float(x),
        float(y),
        float(width),
        float(height),
        float(det.confidence),
        -1,
        -1,
        -1,
    ]


def write_mot_like_detections(detections: List[Detection2D], path: Path) -> None:
    """Write detections in a simple MOT-like format without header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for det in detections:
        row = detection_to_mot_row(det)
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _detection_to_row(det: Detection2D) -> Dict[str, Any]:
    x1, y1, x2, y2 = det.bbox_xyxy
    _x, _y, width, height = xyxy_to_xywh(det.bbox_xyxy)
    return {
        "scene_id": det.scene_id,
        "scene_name": det.scene_name,
        "split": det.split,
        "camera_id": det.camera_id,
        "frame_id": det.frame_id,
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": width,
        "h": height,
        "source": det.source,
    }


def _row_to_detection(row: Dict[str, str]) -> Detection2D:
    bbox_xyxy = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
    bbox_xywh = xyxy_to_xywh(bbox_xyxy)
    return Detection2D(
        scene_id=int(row["scene_id"]),
        scene_name=row["scene_name"],
        split=row["split"],
        camera_id=row["camera_id"],
        frame_id=int(row["frame_id"]),
        class_id=int(row["class_id"]),
        class_name=row["class_name"],
        confidence=float(row["confidence"]),
        bbox_xyxy=bbox_xyxy,
        bbox_xywh=bbox_xywh,
        source=row.get("source", "yolo"),
    )

