"""YOLO bbox conversion and label-file utilities."""

from pathlib import Path
from typing import List, Optional, Tuple

from deep_oc_sort_3d.detection2d.yolo_types import YoloLabel


def xyxy_to_xywh(
    bbox_xyxy: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Convert xyxy bbox to top-left xywh."""
    x1, y1, x2, y2 = bbox_xyxy
    xmin = min(float(x1), float(x2))
    xmax = max(float(x1), float(x2))
    ymin = min(float(y1), float(y2))
    ymax = max(float(y1), float(y2))
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def xywh_to_xyxy(
    bbox_xywh: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Convert top-left xywh bbox to xyxy."""
    x, y, width, height = bbox_xywh
    return (float(x), float(y), float(x) + float(width), float(y) + float(height))


def clip_xyxy_to_image(
    bbox_xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Clip an xyxy bbox to image bounds, returning None if it becomes empty."""
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0.0, min(float(x1), float(x2)))
    right = min(float(width), max(float(x1), float(x2)))
    top = max(0.0, min(float(y1), float(y2)))
    bottom = min(float(height), max(float(y1), float(y2)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def xyxy_to_yolo_norm(
    bbox_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert xyxy pixel bbox to normalized YOLO xywh center format."""
    clipped = clip_xyxy_to_image(bbox_xyxy, image_width, image_height)
    if clipped is None:
        return None
    x, y, width, height = xyxy_to_xywh(clipped)
    if width <= 0.0 or height <= 0.0:
        return None
    x_center = (x + width * 0.5) / float(image_width)
    y_center = (y + height * 0.5) / float(image_height)
    return (
        _clamp01(x_center),
        _clamp01(y_center),
        _clamp01(width / float(image_width)),
        _clamp01(height / float(image_height)),
    )


def yolo_norm_to_xyxy(
    yolo_box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """Convert normalized YOLO xywh center format to xyxy pixel bbox."""
    x_center, y_center, width, height = yolo_box
    box_width = float(width) * float(image_width)
    box_height = float(height) * float(image_height)
    cx = float(x_center) * float(image_width)
    cy = float(y_center) * float(image_height)
    return (
        cx - box_width * 0.5,
        cy - box_height * 0.5,
        cx + box_width * 0.5,
        cy + box_height * 0.5,
    )


def write_yolo_label_file(labels: List[YoloLabel], label_path: Path) -> None:
    """Write YOLO labels to a txt file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for label in labels:
        lines.append(
            "%d %.6f %.6f %.6f %.6f"
            % (
                int(label.class_id),
                float(label.x_center_norm),
                float(label.y_center_norm),
                float(label.width_norm),
                float(label.height_norm),
            )
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_label_file(label_path: Path) -> List[YoloLabel]:
    """Read YOLO labels from a txt file. Empty/missing files return an empty list."""
    if not label_path.exists() or not label_path.is_file():
        return []
    labels = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return labels
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            label = YoloLabel(
                class_id=int(float(parts[0])),
                x_center_norm=float(parts[1]),
                y_center_norm=float(parts[2]),
                width_norm=float(parts[3]),
                height_norm=float(parts[4]),
            )
        except ValueError:
            continue
        labels.append(label)
    return labels


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

