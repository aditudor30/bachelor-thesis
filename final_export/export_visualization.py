"""Visualization helpers for final frame-level global exports."""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.final_export.generic_export import read_global_frame_records_file
from deep_oc_sort_3d.final_export.global_frame_types import GlobalFrameRecord


def draw_global_frame_records_on_frame(
    image: np.ndarray,
    records_for_frame: List[GlobalFrameRecord],
) -> np.ndarray:
    """Draw global frame records on an RGB image."""
    output = image.copy()
    for record in records_for_frame:
        color = _color_from_global_id(record.global_track_id)
        label = _label(record)
        _draw_box(output, record.bbox_xyxy, label, color)
    return output


def visualize_global_export_frame(
    root: Path,
    records_file: Path,
    split: str,
    scene: str,
    camera_id: str,
    frame_id: int,
    output_path: Path,
) -> None:
    """Read one RGB frame and save global export visualization."""
    records = read_global_frame_records_file(records_file)
    records_for_frame = [record for record in records if record.frame_id == int(frame_id)]
    scene_paths = get_scene_paths(root, split, scene)
    video_path = _find_video_path(scene_paths.videos_dir, camera_id)
    if video_path is None:
        raise FileNotFoundError("Missing video for %s %s" % (scene, camera_id))
    image = safe_read_video_frame(video_path, int(frame_id))
    if image is None:
        raise IOError("Could not read frame %d from %s" % (int(frame_id), video_path))
    drawn = draw_global_frame_records_on_frame(image, records_for_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))


def _find_video_path(videos_dir: object, camera_id: str):
    if videos_dir is None:
        return None
    for path in list_video_files(Path(videos_dir)):
        if path.stem == camera_id:
            return path
    return None


def _color_from_global_id(global_track_id: object):
    if global_track_id is None:
        return (255, 190, 0)
    value = int(global_track_id) * 2654435761
    return (
        int((value >> 0) & 255),
        int((value >> 8) & 255),
        int((value >> 16) & 255),
    )


def _label(record: GlobalFrameRecord) -> str:
    global_id = "None" if record.global_track_id is None else str(record.global_track_id)
    label = "G%s %s %.2f" % (global_id, record.class_name, float(record.confidence))
    if record.matched_gt_object_id is not None:
        label += " GT=%d" % int(record.matched_gt_object_id)
    return label


def _draw_box(image: np.ndarray, bbox_xyxy, label: str, color) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
