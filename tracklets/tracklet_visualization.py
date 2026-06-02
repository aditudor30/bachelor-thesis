"""Visualization helpers for local tracklets."""

from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from deep_oc_sort_3d.tracking.track_visualization import color_from_track_id
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


def draw_tracklet_trajectory_on_frame(
    image: np.ndarray,
    tracklet: LocalTracklet,
    frame_id: Optional[int] = None,
) -> np.ndarray:
    """Draw a single tracklet trajectory and optional bbox at frame_id."""
    out = image.copy()
    color = color_from_track_id(tracklet.local_track_id)
    points = []
    for item in tracklet.trajectory_2d:
        _frame, x1, y1, x2, y2 = item
        points.append((int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))))
        if frame_id is not None and int(_frame) == int(frame_id):
            _draw_bbox(out, (x1, y1, x2, y2), color, "T%d %s" % (tracklet.local_track_id, tracklet.class_name))
    for index in range(1, len(points)):
        cv2.line(out, points[index - 1], points[index], color, 2)
    return out


def draw_tracklet_bbox_sequence(image: np.ndarray, records_or_tracklet: Any) -> np.ndarray:
    """Draw all bboxes from a LocalTracklet or a list of record-like objects."""
    out = image.copy()
    if isinstance(records_or_tracklet, LocalTracklet):
        tracklet = records_or_tracklet
        color = color_from_track_id(tracklet.local_track_id)
        for frame_id, x1, y1, x2, y2 in tracklet.trajectory_2d:
            _draw_bbox(out, (x1, y1, x2, y2), color, "T%d f%d" % (tracklet.local_track_id, frame_id))
        return out
    for record in records_or_tracklet:
        color = color_from_track_id(record.local_track_id)
        _draw_bbox(out, record.bbox_xyxy, color, "T%d f%d" % (record.local_track_id, record.frame_id))
    return out


def plot_tracklet_bev_trajectory(tracklet: LocalTracklet, output_path: Path) -> None:
    """Save a BEV trajectory plot for one tracklet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not tracklet.trajectory_3d:
        _write_blank_image(output_path, "No 3D trajectory")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _plot_bev_with_cv2(tracklet, output_path)
        return
    xs = [item[1] for item in tracklet.trajectory_3d]
    ys = [item[2] for item in tracklet.trajectory_3d]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o")
    plt.title("Tracklet %d %s" % (tracklet.local_track_id, tracklet.class_name))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def visualize_tracklets_on_frame(
    image: np.ndarray,
    tracklets: List[LocalTracklet],
    frame_id: int,
) -> np.ndarray:
    """Draw all tracklets visible on frame_id."""
    out = image.copy()
    for tracklet in tracklets:
        if _has_frame(tracklet, frame_id):
            out = draw_tracklet_trajectory_on_frame(out, tracklet, frame_id=frame_id)
    return out


def _draw_bbox(image: np.ndarray, bbox: Any, color: Any, label: str) -> None:
    x1, y1, x2, y2 = bbox
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _has_frame(tracklet: LocalTracklet, frame_id: int) -> bool:
    for item in tracklet.trajectory_2d:
        if int(item[0]) == int(frame_id):
            return True
    return False


def _write_blank_image(path: Path, text: str) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), image)


def _plot_bev_with_cv2(tracklet: LocalTracklet, output_path: Path) -> None:
    xs = np.asarray([item[1] for item in tracklet.trajectory_3d], dtype=float)
    ys = np.asarray([item[2] for item in tracklet.trajectory_3d], dtype=float)
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    if xs.size == 0:
        _write_blank_image(output_path, "No 3D trajectory")
        return
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    points = []
    for x, y in zip(xs, ys):
        px = int(40 + (float(x) - x_min) / x_span * 560)
        py = int(600 - (float(y) - y_min) / y_span * 560)
        points.append((px, py))
    color = color_from_track_id(tracklet.local_track_id)
    for index in range(1, len(points)):
        cv2.line(image, points[index - 1], points[index], color, 2)
    for point in points:
        cv2.circle(image, point, 3, color, -1)
    cv2.imwrite(str(output_path), image)
