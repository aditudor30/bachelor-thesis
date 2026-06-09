"""Lazy frame/crop loading for ReID merge visual panels."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import infer_camera_id_from_video_path, list_video_files, safe_read_video_frame
from deep_oc_sort_3d.reid_visual_decision.visual_decision_config import dataset_root_from_config, frame_records_root_from_config
from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import (
    frame_record_csv_files,
    parse_track_key_or_empty,
    read_csv_rows,
    safe_float,
    safe_int,
)


def load_visual_evidence(event: Dict[str, Any], config: Dict[str, Any], max_crops: int = 4) -> Dict[str, Any]:
    """Load representative crops and context metadata for both merge fragments."""
    key_a = parse_track_key_or_empty(event.get("fragment_a_id"))
    key_b = parse_track_key_or_empty(event.get("fragment_b_id"))
    rows_a = load_track_rows(key_a, config)
    rows_b = load_track_rows(key_b, config)
    crops_a = load_representative_crops(rows_a, config, max_crops=max_crops)
    crops_b = load_representative_crops(rows_b, config, max_crops=max_crops)
    return {
        "rows_a": rows_a,
        "rows_b": rows_b,
        "crops_a": crops_a,
        "crops_b": crops_b,
        "num_rows_a": len(rows_a),
        "num_rows_b": len(rows_b),
        "num_crops_a": len([item for item in crops_a if item.get("crop") is not None]),
        "num_crops_b": len([item for item in crops_b if item.get("crop") is not None]),
    }


def load_track_rows(key: Tuple[str, str, str, str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load frame_global_records for one track key."""
    if "" in key:
        return []
    root = frame_records_root_from_config(config)
    subset, scene_name, class_id, global_track_id = key
    files = frame_record_csv_files(root, subsets=[subset], scenes=[scene_name])
    rows: List[Dict[str, Any]] = []
    for path in files:
        csv_rows, _fields = read_csv_rows(path)
        camera_id = camera_id_from_frame_record_path(path)
        for row in csv_rows:
            if str(row.get("class_id", "")) != str(class_id):
                continue
            if str(row.get("global_track_id", "")) != str(global_track_id):
                continue
            copied = dict(row)
            copied.setdefault("subset", subset)
            copied.setdefault("scene_name", scene_name)
            copied.setdefault("camera_id", camera_id)
            copied.setdefault("source_csv", str(path))
            rows.append(copied)
    rows.sort(key=lambda item: (safe_int(item.get("frame_id"), 0) or 0, str(item.get("camera_id", ""))))
    return rows


def camera_id_from_frame_record_path(path: Path) -> str:
    """Infer camera id from a frame_global_records filename."""
    name = Path(path).name
    suffix = "_global_records.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(path).stem


def load_representative_crops(rows: List[Dict[str, Any]], config: Dict[str, Any], max_crops: int) -> List[Dict[str, Any]]:
    """Load representative crops for a row group."""
    selected = select_representative_rows(rows, max_crops)
    output = []
    cache: Dict[str, Optional[np.ndarray]] = {}
    for row in selected:
        frame = load_frame_for_row(row, config, cache)
        crop = crop_from_row(frame, row)
        output.append({"row": row, "frame": frame, "crop": crop, "bbox": bbox_from_row(row)})
    return output


def select_representative_rows(rows: List[Dict[str, Any]], max_crops: int) -> List[Dict[str, Any]]:
    """Pick first/middle/last/high-confidence rows without loading all frames."""
    if not rows or max_crops <= 0:
        return []
    if len(rows) <= max_crops:
        return list(rows)
    indices = set([0, len(rows) // 2, len(rows) - 1])
    ranked = sorted(range(len(rows)), key=lambda idx: safe_float(rows[idx].get("confidence"), 0.0) or 0.0, reverse=True)
    for idx in ranked:
        indices.add(idx)
        if len(indices) >= max_crops:
            break
    return [rows[idx] for idx in sorted(indices)[:max_crops]]


def load_frame_for_row(row: Dict[str, Any], config: Dict[str, Any], cache: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Load the RGB frame for a record row."""
    subset = str(row.get("subset", ""))
    split = subset_to_split(subset)
    scene_name = str(row.get("scene_name", ""))
    camera_id = str(row.get("camera_id", ""))
    frame_id = safe_int(row.get("frame_id"), None)
    if frame_id is None:
        return None
    key = "%s|%s|%s|%s" % (split, scene_name, camera_id, frame_id)
    if key in cache:
        return cache[key]
    video_path = find_video_path(dataset_root_from_config(config), split, scene_name, camera_id)
    if video_path is None:
        cache[key] = None
        return None
    frame = safe_read_video_frame(video_path, frame_id)
    cache[key] = frame
    return frame


def subset_to_split(subset: str) -> str:
    """Map pipeline subset names to dataset split names."""
    if subset in ("official_val", "val"):
        return "val"
    if subset in ("internal_holdout", "train"):
        return "train"
    if subset == "test":
        return "test"
    return "train"


def find_video_path(root: Path, split: str, scene_name: str, camera_id: str) -> Optional[Path]:
    """Find the video path for a scene/camera."""
    scene_paths = get_scene_paths(Path(root), split, scene_name)
    if scene_paths.videos_dir is None:
        return None
    for path in list_video_files(scene_paths.videos_dir):
        if infer_camera_id_from_video_path(path) == camera_id:
            return path
    return None


def bbox_from_row(row: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Extract bbox xyxy from a CSV row."""
    values = [
        safe_float(row.get("x1"), None),
        safe_float(row.get("y1"), None),
        safe_float(row.get("x2"), None),
        safe_float(row.get("y2"), None),
    ]
    if any(value is None for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def crop_from_row(frame: Optional[np.ndarray], row: Dict[str, Any]) -> Optional[np.ndarray]:
    """Crop an RGB frame using the row bbox."""
    if frame is None:
        return None
    bbox = bbox_from_row(row)
    if bbox is None:
        return None
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(round(min(bbox[0], bbox[2])))))
    y1 = max(0, min(height - 1, int(round(min(bbox[1], bbox[3])))))
    x2 = max(0, min(width, int(round(max(bbox[0], bbox[2])))))
    y2 = max(0, min(height, int(round(max(bbox[1], bbox[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()
