"""Visualization helpers for YOLO labels and detections."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.detection2d.yolo_label_utils import yolo_norm_to_xyxy
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D, YoloLabel


DEFAULT_CLASS_NAMES = {
    0: "Person",
    1: "Forklift",
    2: "PalletTruck",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "NovaCarter",
}


def draw_yolo_labels_on_image(
    image_rgb: np.ndarray,
    labels: List[YoloLabel],
    class_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Draw YOLO labels on an RGB image."""
    names = class_names if class_names is not None else DEFAULT_CLASS_NAMES
    out = image_rgb.copy()
    height, width = out.shape[:2]
    for label in labels:
        bbox = yolo_norm_to_xyxy(
            (label.x_center_norm, label.y_center_norm, label.width_norm, label.height_norm),
            width,
            height,
        )
        _draw_box(out, bbox, "%s:%d" % (names.get(label.class_id, "cls"), label.class_id), (255, 0, 0))
    return out


def draw_detections_on_image(
    image_rgb: np.ndarray,
    detections: List[Detection2D],
) -> np.ndarray:
    """Draw common detection records on an RGB image."""
    out = image_rgb.copy()
    for det in detections:
        label = "%s %.2f" % (det.class_name, det.confidence)
        _draw_box(out, det.bbox_xyxy, label, (0, 255, 0))
    return out


def make_image_grid(
    images: List[np.ndarray],
    cols: int = 4,
    cell_size: Tuple[int, int] = (320, 240),
) -> np.ndarray:
    """Make a simple RGB image grid from variable-size RGB images."""
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    cols = max(int(cols), 1)
    rows = int(np.ceil(float(len(images)) / float(cols)))
    cell_w, cell_h = cell_size
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        resized = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        grid[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = resized
    return grid


def _draw_box(
    image: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

