"""Audit GT 2D bbox scale, visibility, and difficulty for YOLO planning."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths, scene_name_to_id
from deep_oc_sort_3d.data.frame_io import (
    get_video_resolution,
    infer_camera_id_from_video_path,
    list_video_files,
)
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.detection2d.yolo_bbox_difficulty import classify_bbox_difficulty
from deep_oc_sort_3d.detection2d.yolo_dataset_exporter import DEFAULT_CLASS_MAPPING
from deep_oc_sort_3d.detection2d.yolo_label_utils import clip_xyxy_to_image


@dataclass
class BBoxAuditRecord:
    """One visible GT bbox audit row."""

    split: str
    scene_name: str
    scene_id: int
    camera_id: str
    frame_id: int
    object_id: int
    class_name: str
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    width_px: float
    height_px: float
    area_px: float
    aspect_ratio: float
    image_width: int
    image_height: int
    width_norm: float
    height_norm: float
    area_norm: float
    center_x_norm: float
    center_y_norm: float
    difficulty: str


def audit_gt_bboxes(
    root: Union[str, Path],
    split: str,
    scenes: List[str],
    camera_id: Optional[str] = None,
    frame_stride: int = 1,
    max_frames_per_scene: Optional[int] = None,
    difficulty_config: Optional[Dict[str, Any]] = None,
) -> List[BBoxAuditRecord]:
    """Collect scale/difficulty records for visible GT boxes."""
    if split == "test":
        raise ValueError("BBox audit requires ground truth and only supports train/val.")
    root_path = Path(root)
    selected_camera = _normalize_camera_id(camera_id)
    stride = max(int(frame_stride), 1)
    records = []
    for scene_name in scenes:
        scene_paths = get_scene_paths(root_path, split, scene_name)
        if scene_paths.ground_truth_path is None or not scene_paths.ground_truth_path.exists():
            print("warning: missing ground_truth.json for %s %s" % (split, scene_name))
            continue
        gt_objects = load_ground_truth_json(scene_paths.ground_truth_path)
        image_sizes = _resolve_image_sizes(scene_paths)
        scene_id = scene_name_to_id(scene_name)
        if scene_id is None:
            scene_id = -1
        for obj in gt_objects:
            if not _frame_selected(obj.frame_id, max_frames_per_scene, stride):
                continue
            for cam_id, bbox in obj.visible_bboxes_2d.items():
                if selected_camera is not None and cam_id != selected_camera:
                    continue
                image_width, image_height = _image_size_for_camera(image_sizes, cam_id, bbox)
                record = _record_from_bbox(
                    split=split,
                    scene_name=scene_name,
                    scene_id=scene_id,
                    camera_id=cam_id,
                    obj=obj,
                    bbox=bbox,
                    image_width=image_width,
                    image_height=image_height,
                    difficulty_config=difficulty_config,
                )
                if record is not None:
                    records.append(record)
    return records


def records_to_dataframe(records: List[BBoxAuditRecord]) -> Any:
    """Return a pandas DataFrame when pandas is installed, otherwise dictionaries."""
    rows = [asdict(record) for record in records]
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)


def save_bbox_audit_csv(records: List[BBoxAuditRecord], output_path: Path) -> None:
    """Save bbox audit records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BBoxAuditRecord.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def summarize_bbox_audit(records: List[BBoxAuditRecord]) -> Dict[str, Any]:
    """Summarize bbox scale and difficulty records."""
    summary = {
        "total_boxes": len(records),
        "count_per_class": _count_by(records, "class_name"),
        "count_per_split": _count_by(records, "split"),
        "count_per_scene": _count_by(records, "scene_name"),
        "count_per_camera": _count_by(records, "camera_id"),
        "count_per_difficulty": _count_by(records, "difficulty"),
        "per_class_stats": {},
        "top_scenes_per_class": {},
        "top_cameras_per_class": {},
        "warnings": [],
    }
    for class_name in sorted(set(record.class_name for record in records)):
        class_records = [record for record in records if record.class_name == class_name]
        summary["per_class_stats"][class_name] = _class_stats(class_records)
        summary["top_scenes_per_class"][class_name] = _top_counts(class_records, "scene_name")
        summary["top_cameras_per_class"][class_name] = _top_counts(class_records, "camera_id")
    for class_name in DEFAULT_CLASS_MAPPING.keys():
        if summary["count_per_class"].get(class_name, 0) == 0:
            summary["warnings"].append("missing class: %s" % class_name)
    return summary


def load_bbox_audit_csv(path: Path) -> List[Dict[str, Any]]:
    """Load audit CSV rows as dictionaries with numeric values parsed where useful."""
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(_parse_audit_row(row))
    return rows


def save_summary_json(summary: Dict[str, Any], output_path: Path) -> None:
    """Save summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _record_from_bbox(
    split: str,
    scene_name: str,
    scene_id: int,
    camera_id: str,
    obj: GroundTruthObject,
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    difficulty_config: Optional[Dict[str, Any]],
) -> Optional[BBoxAuditRecord]:
    clipped = clip_xyxy_to_image(bbox, image_width, image_height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    if width <= 0.0 or height <= 0.0:
        return None
    area = width * height
    image_area = float(max(int(image_width), 1) * max(int(image_height), 1))
    width_norm = width / float(max(int(image_width), 1))
    height_norm = height / float(max(int(image_height), 1))
    area_norm = area / image_area
    center_x_norm = (x1 + width * 0.5) / float(max(int(image_width), 1))
    center_y_norm = (y1 + height * 0.5) / float(max(int(image_height), 1))
    aspect_ratio = width / max(height, 1e-9)
    difficulty = classify_bbox_difficulty(width, height, area_norm, difficulty_config)
    class_name = str(obj.object_type)
    return BBoxAuditRecord(
        split=split,
        scene_name=scene_name,
        scene_id=int(scene_id),
        camera_id=str(camera_id),
        frame_id=int(obj.frame_id),
        object_id=int(obj.object_id),
        class_name=class_name,
        class_id=_class_id_for_name(class_name),
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        width_px=float(width),
        height_px=float(height),
        area_px=float(area),
        aspect_ratio=float(aspect_ratio),
        image_width=int(image_width),
        image_height=int(image_height),
        width_norm=float(width_norm),
        height_norm=float(height_norm),
        area_norm=float(area_norm),
        center_x_norm=float(center_x_norm),
        center_y_norm=float(center_y_norm),
        difficulty=difficulty,
    )


def _resolve_image_sizes(scene_paths: Any) -> Dict[str, Tuple[int, int]]:
    sizes = {}
    if scene_paths.videos_dir is not None:
        for video_path in list_video_files(scene_paths.videos_dir):
            camera_id = infer_camera_id_from_video_path(video_path)
            resolution = get_video_resolution(video_path)
            if resolution is not None:
                sizes[camera_id] = resolution
    if scene_paths.calibration_path is not None and scene_paths.calibration_path.exists():
        calibrations = load_calibration_json(scene_paths.calibration_path)
        for camera_id, calibration in calibrations.items():
            if camera_id in sizes:
                continue
            if calibration.frame_width is not None and calibration.frame_height is not None:
                sizes[camera_id] = (int(calibration.frame_width), int(calibration.frame_height))
    return sizes


def _image_size_for_camera(
    image_sizes: Dict[str, Tuple[int, int]],
    camera_id: str,
    bbox: Tuple[float, float, float, float],
) -> Tuple[int, int]:
    if camera_id in image_sizes:
        return image_sizes[camera_id]
    x1, y1, x2, y2 = bbox
    width = int(max(max(float(x1), float(x2)), 1.0))
    height = int(max(max(float(y1), float(y2)), 1.0))
    return (width, height)


def _frame_selected(frame_id: int, max_frames_per_scene: Optional[int], frame_stride: int) -> bool:
    if max_frames_per_scene is not None and int(frame_id) >= int(max_frames_per_scene):
        return False
    return int(frame_id) % max(int(frame_stride), 1) == 0


def _normalize_camera_id(camera_id: Optional[str]) -> Optional[str]:
    if camera_id is None:
        return None
    if str(camera_id).lower() == "all":
        return None
    return str(camera_id)


def _class_id_for_name(class_name: str) -> int:
    if class_name in DEFAULT_CLASS_MAPPING:
        return int(DEFAULT_CLASS_MAPPING[class_name])
    lower = {}
    for name, class_id in DEFAULT_CLASS_MAPPING.items():
        lower[name.lower()] = int(class_id)
    return int(lower.get(class_name.lower(), -1))


def _count_by(records: List[BBoxAuditRecord], field_name: str) -> Dict[str, int]:
    counts = {}
    for record in records:
        key = str(getattr(record, field_name))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_counts(records: List[BBoxAuditRecord], field_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    counts = _count_by(records, field_name)
    rows = []
    for key, count in counts.items():
        rows.append({"name": key, "count": int(count)})
    return sorted(rows, key=lambda item: item["count"], reverse=True)[:limit]


def _class_stats(records: List[BBoxAuditRecord]) -> Dict[str, Any]:
    area_norm = np.asarray([record.area_norm for record in records], dtype=float)
    width_px = np.asarray([record.width_px for record in records], dtype=float)
    height_px = np.asarray([record.height_px for record in records], dtype=float)
    aspect = np.asarray([record.aspect_ratio for record in records], dtype=float)
    return {
        "area_norm_mean": _mean(area_norm),
        "area_norm_median": _median(area_norm),
        "area_norm_p25": _percentile(area_norm, 25),
        "area_norm_p75": _percentile(area_norm, 75),
        "width_px_mean": _mean(width_px),
        "width_px_median": _median(width_px),
        "height_px_mean": _mean(height_px),
        "height_px_median": _median(height_px),
        "aspect_ratio_median": _median(aspect),
        "difficulty_counts": _difficulty_counts(records),
    }


def _difficulty_counts(records: List[BBoxAuditRecord]) -> Dict[str, int]:
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for record in records:
        counts[record.difficulty] = counts.get(record.difficulty, 0) + 1
    return counts


def _mean(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _median(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.median(values))


def _percentile(values: np.ndarray, percentile: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _parse_audit_row(row: Dict[str, str]) -> Dict[str, Any]:
    parsed = dict(row)
    int_fields = ["scene_id", "frame_id", "object_id", "class_id", "image_width", "image_height"]
    float_fields = [
        "x1",
        "y1",
        "x2",
        "y2",
        "width_px",
        "height_px",
        "area_px",
        "aspect_ratio",
        "width_norm",
        "height_norm",
        "area_norm",
        "center_x_norm",
        "center_y_norm",
    ]
    for field in int_fields:
        if field in parsed and parsed[field] != "":
            parsed[field] = int(float(parsed[field]))
    for field in float_fields:
        if field in parsed and parsed[field] != "":
            parsed[field] = float(parsed[field])
    return parsed

