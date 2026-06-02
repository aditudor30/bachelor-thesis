"""Visualization helpers for local tracks."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord


def draw_tracks_on_frame(
    image: np.ndarray,
    records_for_frame: List[LocalTrackRecord],
    show_gt: bool = True,
) -> np.ndarray:
    """Draw local track bboxes and compact labels on an RGB image."""
    out = image.copy()
    for record in records_for_frame:
        color = color_from_track_id(record.local_track_id)
        label = _format_label(record, show_gt)
        _draw_box(out, record.bbox_xyxy, label, color)
    return out


def color_from_track_id(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic RGB color for a local track id."""
    value = int(track_id) * 2654435761
    return (
        int((value >> 0) & 255),
        int((value >> 8) & 255),
        int((value >> 16) & 255),
    )


def save_track_visualization(image: np.ndarray, records_for_frame: List[LocalTrackRecord], output_path: Path) -> None:
    """Draw tracks and save a PNG."""
    drawn = draw_tracks_on_frame(image, records_for_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))


def _format_label(record: LocalTrackRecord, show_gt: bool) -> str:
    label = "T%d %s %.2f %s" % (
        int(record.local_track_id),
        record.class_name,
        float(record.confidence),
        record.track_state,
    )
    if show_gt and record.matched_gt_object_id is not None:
        label += " GT=%d" % int(record.matched_gt_object_id)
    return label


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
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
