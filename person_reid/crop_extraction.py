"""RGB crop extraction for Person ReID diagnostics."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.person_reid.reid_types import PersonCropRecord
from deep_oc_sort_3d.person_reid.reid_utils import (
    count_by,
    progress_iter,
    read_csv_rows,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
)


def extract_person_reid_crops_from_config(config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Extract Person RGB crops from frame_global_records."""
    root = Path(str(config.get("reid_person", {}).get("output_root", "output/reid_person/baseline_v2_pseudo3d_fullcam")))
    paths = config.get("paths", {})
    dataset_root = Path(str(paths.get("dataset_root", "/path/to/MTMC_Tracking_2026")))
    frame_root = Path(str(paths.get("frame_global_records_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam/frame_global_records")))
    crop_config = config.get("crops", {})
    diagnostics = config.get("diagnostics", {})
    subsets = _optional_list(diagnostics.get("subsets"))
    scenes = _optional_list(diagnostics.get("scenes"))
    metadata_path = root / "crops" / "person_crops.csv"
    if metadata_path.exists() and not overwrite:
        rows, _fields = read_csv_rows(metadata_path)
        return {"status": "skipped_existing", "crop_records": len(rows), "metadata_path": str(metadata_path)}
    files = _frame_record_files(frame_root, subsets, scenes)
    all_crop_rows: List[Dict[str, Any]] = []
    summary_rows = []
    for csv_path in progress_iter(files, show_progress, "extract Person ReID crops", "file"):
        rows, _fields = read_csv_rows(csv_path)
        file_summary, crop_rows = _extract_from_frame_csv(
            csv_path,
            rows,
            dataset_root,
            root / "crops",
            crop_config,
            int(config.get("reid_person", {}).get("class_id", 0)),
        )
        summary_rows.append(file_summary)
        all_crop_rows.extend(crop_rows)
    fieldnames = person_crop_csv_fields()
    write_csv_rows(all_crop_rows, metadata_path, fieldnames)
    write_csv_rows(summary_rows, root / "crops" / "crop_extraction_by_file.csv")
    summary = summarize_crop_extraction(all_crop_rows, summary_rows)
    summary["metadata_path"] = str(metadata_path)
    write_json(summary, root / "summaries" / "crop_extraction_summary.json")
    return summary


def crop_image_xyxy(
    image: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    padding_ratio: float = 0.0,
) -> Optional[np.ndarray]:
    """Crop an RGB image by padded xyxy bbox."""
    if image is None:
        return None
    height, width = image.shape[:2]
    bbox = expand_and_clip_bbox(bbox_xyxy, width, height, padding_ratio)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def expand_and_clip_bbox(
    bbox_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Expand and clamp bbox to image boundaries."""
    x1, y1, x2, y2 = bbox_xyxy
    width = float(x2) - float(x1)
    height = float(y2) - float(y1)
    if width <= 0.0 or height <= 0.0:
        return None
    pad_x = width * float(padding_ratio)
    pad_y = height * float(padding_ratio)
    left = max(0, min(int(round(float(x1) - pad_x)), int(image_width) - 1))
    top = max(0, min(int(round(float(y1) - pad_y)), int(image_height) - 1))
    right = max(0, min(int(round(float(x2) + pad_x)), int(image_width)))
    bottom = max(0, min(int(round(float(y2) + pad_y)), int(image_height)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def person_crop_csv_fields() -> List[str]:
    """Return crop metadata CSV fields."""
    return [
        "crop_id",
        "subset",
        "split",
        "scene_name",
        "camera_id",
        "frame_id",
        "local_track_id",
        "global_track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_path",
        "matched_gt_object_id",
        "source_csv",
    ]


def crop_record_from_row(row: Dict[str, Any]) -> PersonCropRecord:
    """Create a crop record from CSV row."""
    return PersonCropRecord(
        crop_id=str(row.get("crop_id", "")),
        subset=str(row.get("subset", "")),
        split=str(row.get("split", "")),
        scene_name=str(row.get("scene_name", "")),
        camera_id=str(row.get("camera_id", "")),
        frame_id=safe_int(row.get("frame_id"), -1) or -1,
        local_track_id=safe_int(row.get("local_track_id"), None),
        global_track_id=safe_int(row.get("global_track_id"), None),
        class_id=safe_int(row.get("class_id"), -1) or -1,
        class_name=str(row.get("class_name", "")),
        confidence=safe_float(row.get("confidence"), 0.0) or 0.0,
        bbox_xyxy=(
            safe_float(row.get("x1"), 0.0) or 0.0,
            safe_float(row.get("y1"), 0.0) or 0.0,
            safe_float(row.get("x2"), 0.0) or 0.0,
            safe_float(row.get("y2"), 0.0) or 0.0,
        ),
        crop_path=str(row.get("crop_path", "")),
        matched_gt_object_id=safe_int(row.get("matched_gt_object_id"), None),
        source_csv=str(row.get("source_csv", "")),
    )


def summarize_crop_extraction(crop_rows: List[Dict[str, Any]], file_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize crop extraction."""
    total_person = sum([int(row.get("person_records", 0)) for row in file_rows])
    invalid = sum([int(row.get("invalid_bbox_count", 0)) for row in file_rows])
    missing = sum([int(row.get("missing_frame_count", 0)) for row in file_rows])
    return {
        "status": "ok",
        "files": len(file_rows),
        "total_person_records": total_person,
        "crop_records": len(crop_rows),
        "crop_success_rate": float(len(crop_rows)) / float(total_person) if total_person else None,
        "invalid_bbox_count": invalid,
        "missing_frame_count": missing,
        "per_subset": count_by(crop_rows, "subset"),
        "per_scene": count_by(crop_rows, "scene_name"),
        "per_camera": count_by(crop_rows, "camera_id"),
    }


def _extract_from_frame_csv(
    csv_path: Path,
    rows: List[Dict[str, Any]],
    dataset_root: Path,
    crop_root: Path,
    config: Dict[str, Any],
    class_id: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    person_rows = [row for row in rows if safe_int(row.get("class_id"), -1) == class_id]
    selected = _sample_rows_by_track(person_rows, int(config.get("max_crops_per_track", 8)), str(config.get("sampling", "uniform")))
    crop_rows = []
    invalid_bbox = 0
    missing_frame = 0
    video_path = _find_video_path(dataset_root, _first_value(rows, "split"), _first_value(rows, "scene_name"), _first_value(rows, "camera_id"))
    frame_cache: Dict[int, Optional[np.ndarray]] = {}
    for row in selected:
        frame_id = safe_int(row.get("frame_id"), -1) or -1
        if video_path is None:
            missing_frame += 1
            continue
        if frame_id not in frame_cache:
            frame_cache[frame_id] = safe_read_video_frame(video_path, frame_id)
        image = frame_cache[frame_id]
        if image is None:
            missing_frame += 1
            continue
        bbox = _bbox(row)
        if bbox is None:
            invalid_bbox += 1
            continue
        min_width = float(config.get("min_width", 8))
        min_height = float(config.get("min_height", 16))
        if (bbox[2] - bbox[0]) < min_width or (bbox[3] - bbox[1]) < min_height:
            invalid_bbox += 1
            continue
        crop = crop_image_xyxy(image, bbox, padding_ratio=float(config.get("padding_ratio", 0.10)))
        if crop is None:
            invalid_bbox += 1
            continue
        width = int(config.get("output_size", {}).get("width", 128))
        height = int(config.get("output_size", {}).get("height", 256))
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        crop_id = _crop_id(row)
        crop_path = crop_root / str(row.get("subset", "unknown")) / str(row.get("scene_name", "")) / str(row.get("camera_id", "")) / ("%s.jpg" % crop_id)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        crop_rows.append(_crop_metadata_row(row, crop_id, str(crop_path), str(csv_path), bbox))
    summary = {
        "source_csv": str(csv_path),
        "subset": _first_value(rows, "subset"),
        "scene_name": _first_value(rows, "scene_name"),
        "camera_id": _first_value(rows, "camera_id"),
        "person_records": len(person_rows),
        "sampled_records": len(selected),
        "crop_records": len(crop_rows),
        "invalid_bbox_count": invalid_bbox,
        "missing_frame_count": missing_frame,
        "video_path": "" if video_path is None else str(video_path),
    }
    return summary, crop_rows


def _sample_rows_by_track(rows: List[Dict[str, Any]], max_crops: int, sampling: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("subset", "")),
            str(row.get("scene_name", "")),
            str(row.get("camera_id", "")),
            str(row.get("global_track_id", "")),
            str(row.get("local_track_id", "")),
        )
        groups.setdefault(key, []).append(row)
    selected = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda item: safe_int(item.get("frame_id"), -1) or -1)
        if len(ordered) <= max_crops:
            selected.extend(ordered)
        elif sampling == "highest_confidence":
            top = sorted(ordered, key=lambda item: safe_float(item.get("confidence"), 0.0) or 0.0, reverse=True)[:max_crops]
            selected.extend(sorted(top, key=lambda item: safe_int(item.get("frame_id"), -1) or -1))
        else:
            indices = _uniform_indices(len(ordered), max_crops)
            selected.extend([ordered[index] for index in indices])
    return selected


def _uniform_indices(length: int, count: int) -> List[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    values = np.linspace(0, length - 1, int(count))
    return sorted(set([int(round(value)) for value in values]))[:count]


def _frame_record_files(root: Path, subsets: Optional[List[str]], scenes: Optional[List[str]]) -> List[Path]:
    files = sorted(root.rglob("*_global_records.csv")) if root.exists() else []
    if subsets is not None:
        subset_set = set([str(item) for item in subsets])
        files = [path for path in files if _subset_from_frame_path(path) in subset_set]
    if scenes is not None:
        scene_set = set([str(item) for item in scenes])
        files = [path for path in files if path.parent.name in scene_set]
    return files


def _subset_from_frame_path(path: Path) -> str:
    parts = list(path.parts)
    if "frame_global_records" in parts:
        index = parts.index("frame_global_records")
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return ""


def _find_video_path(dataset_root: Path, split: str, scene_name: str, camera_id: str) -> Optional[Path]:
    videos_dir = dataset_root / str(split) / str(scene_name) / "videos"
    for video_path in list_video_files(videos_dir):
        if video_path.stem == str(camera_id):
            return video_path
    return None


def _bbox(row: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    values = [safe_float(row.get("x1"), None), safe_float(row.get("y1"), None), safe_float(row.get("x2"), None), safe_float(row.get("y2"), None)]
    if any(value is None for value in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _crop_id(row: Dict[str, Any]) -> str:
    return "%s__%s__%s__g%s__l%s__f%06d" % (
        str(row.get("subset", "")),
        str(row.get("scene_name", "")),
        str(row.get("camera_id", "")),
        str(row.get("global_track_id", "")),
        str(row.get("local_track_id", "")),
        safe_int(row.get("frame_id"), -1) or -1,
    )


def _crop_metadata_row(row: Dict[str, Any], crop_id: str, crop_path: str, source_csv: str, bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    return {
        "crop_id": crop_id,
        "subset": row.get("subset", ""),
        "split": row.get("split", ""),
        "scene_name": row.get("scene_name", ""),
        "camera_id": row.get("camera_id", ""),
        "frame_id": row.get("frame_id", ""),
        "local_track_id": row.get("local_track_id", ""),
        "global_track_id": row.get("global_track_id", ""),
        "class_id": row.get("class_id", ""),
        "class_name": row.get("class_name", ""),
        "confidence": row.get("confidence", ""),
        "x1": bbox[0],
        "y1": bbox[1],
        "x2": bbox[2],
        "y2": bbox[3],
        "crop_path": crop_path,
        "matched_gt_object_id": row.get("matched_gt_object_id", ""),
        "source_csv": source_csv,
    }


def _first_value(rows: List[Dict[str, Any]], key: str) -> str:
    if not rows:
        return ""
    return str(rows[0].get(key, ""))


def _optional_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]
