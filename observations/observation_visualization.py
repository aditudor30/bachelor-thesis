"""Visualization helpers for standardized 3D observations."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.observations.observation_types import Observation3D


def draw_observations_on_image(
    image_rgb: np.ndarray,
    observations: List[Observation3D],
    show_center_3d: bool = True,
) -> np.ndarray:
    """Draw Observation3D boxes and compact metadata on an RGB image."""
    out = image_rgb.copy()
    for obs in observations:
        color = (0, 255, 0) if obs.matched_gt else (255, 190, 0)
        label = _format_observation_label(obs)
        _draw_box(out, obs.bbox_xyxy, label, color)
        if show_center_3d and obs.center_3d is not None:
            center_text = _format_center_3d(obs.center_3d)
            _draw_text(out, center_text, (int(round(obs.bbox_xyxy[0])), int(round(obs.bbox_xyxy[3])) + 16), color)
    return out


def filter_observations(
    observations: List[Observation3D],
    camera_id: str,
    frame_id: int,
) -> List[Observation3D]:
    """Filter observations for one camera and frame."""
    return [obs for obs in observations if obs.camera_id == camera_id and obs.frame_id == int(frame_id)]


def _format_observation_label(obs: Observation3D) -> str:
    gt_text = "GT"
    if not obs.matched_gt:
        gt_text = "noGT"
    object_text = ""
    if obs.object_id is not None:
        object_text = " id=%d" % obs.object_id
    return "%s %.2f %s%s" % (obs.class_name, obs.confidence, gt_text, object_text)


def _format_center_3d(center_3d: np.ndarray) -> str:
    values = np.asarray(center_3d, dtype=float).reshape(-1)
    if values.size < 3:
        return "xyz=?"
    return "xyz=(%.1f, %.1f, %.1f)" % (values[0], values[1], values[2])


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
    _draw_text(image, label, p1, color)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    x, y = origin
    y = max(16, y)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

