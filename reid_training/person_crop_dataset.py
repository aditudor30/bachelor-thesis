"""Build a SmartSpaces Person ReID crop dataset from GT boxes."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.data.dataset_structure import scene_name_to_id
from deep_oc_sort_3d.data.frame_io import get_video_resolution, infer_camera_id_from_video_path, list_video_files, safe_read_video_frame
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.reid_training.reid_dataset_config import dataset_root_from_config, output_root_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_io import (
    count_by,
    progress_iter,
    write_csv_rows,
    write_json,
    write_jsonl,
    write_text_lines,
)
from deep_oc_sort_3d.training.target_builder import DEFAULT_CLASS_MAPPING


CROP_METADATA_FIELDS = [
    "crop_id",
    "split",
    "source_split",
    "scene_name",
    "scene_id",
    "camera_id",
    "frame_id",
    "class_id",
    "class_name",
    "object_id",
    "identity_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "image_width",
    "image_height",
    "crop_width",
    "crop_height",
    "crop_path",
    "source_video_path",
    "gt_path",
    "is_valid_crop",
    "invalid_reason",
]


def build_person_reid_dataset_from_config(config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Build Person crop metadata and image files from train/val GT."""
    output_root = output_root_from_config(config)
    metadata_path = output_root / "metadata" / "all_crops.csv"
    skip_existing = bool(config.get("person_reid_dataset", {}).get("skip_existing", False))
    if metadata_path.exists() and not overwrite and not skip_existing:
        return {"status": "skipped_existing", "metadata_path": str(metadata_path)}

    dataset_root = dataset_root_from_config(config)
    valid_candidates: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    scene_specs = _scene_specs(config)
    for split_key, split_name, scene_name in progress_iter(scene_specs, show_progress, "collect Person GT candidates", "scene"):
        scene_valid, scene_invalid = collect_scene_person_candidates(dataset_root, split_key, split_name, scene_name, config)
        valid_candidates.extend(scene_valid)
        invalid_rows.extend(scene_invalid)

    selected = sample_candidates_per_identity(valid_candidates, int(config.get("crop_extraction", {}).get("max_crops_per_identity", 200)))
    assign_reid_splits(selected, config)
    assign_reid_splits(invalid_rows, config)
    saved_rows = save_selected_crops(selected, output_root, config, show_progress=show_progress, overwrite=overwrite, skip_existing=skip_existing)
    all_rows = saved_rows + invalid_rows
    write_metadata_outputs(all_rows, output_root)
    summary = {
        "status": "ok",
        "output_root": str(output_root),
        "candidate_valid_before_sampling": len(valid_candidates),
        "invalid_before_saving": len(invalid_rows),
        "selected_after_identity_sampling": len(selected),
        "metadata_rows": len(all_rows),
        "valid_crops": len([row for row in all_rows if str(row.get("is_valid_crop")) == "1"]),
        "invalid_crops": len([row for row in all_rows if str(row.get("is_valid_crop")) != "1"]),
        "per_split": count_by([row for row in all_rows if str(row.get("is_valid_crop")) == "1"], "split"),
        "per_scene": count_by([row for row in all_rows if str(row.get("is_valid_crop")) == "1"], "scene_name"),
        "per_camera": count_by([row for row in all_rows if str(row.get("is_valid_crop")) == "1"], "camera_id"),
    }
    write_json(summary, output_root / "diagnostics" / "crop_extraction_summary.json")
    return summary


def identity_id(scene_name: str, object_id: int) -> str:
    """Build conservative scene-scoped identity id."""
    return "%s_%s" % (str(scene_name), str(int(object_id)))


def collect_scene_person_candidates(
    dataset_root: Path,
    split_key: str,
    split_name: str,
    scene_name: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect raw crop candidates for one scene without reading video frames."""
    gt_path = dataset_root / split_name / scene_name / "ground_truth.json"
    if not gt_path.exists():
        return [], [_invalid_scene_row(split_key, split_name, scene_name, gt_path, "missing_ground_truth")]
    gt_objects = load_ground_truth_json(gt_path)
    video_index = _video_index(dataset_root / split_name / scene_name / "videos")
    video_sizes = {camera_id: _video_size(path) for camera_id, path in video_index.items()}
    crop_cfg = config.get("crop_extraction", {})
    frame_stride = max(1, int(crop_cfg.get("frame_stride", 5)))
    class_name = str(config.get("selection", {}).get("class_name", "Person"))
    class_id = int(config.get("selection", {}).get("class_id", 0))
    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    for obj in gt_objects:
        if not _is_target_person(obj, class_id, class_name):
            continue
        if int(obj.frame_id) % frame_stride != 0:
            continue
        for camera_id, bbox in obj.visible_bboxes_2d.items():
            row = _candidate_row(obj, bbox, split_key, split_name, scene_name, camera_id, gt_path, video_index.get(camera_id), video_sizes.get(camera_id, ("", "")))
            invalid_reason = validate_bbox_size(row, crop_cfg)
            if row.get("source_video_path", "") == "":
                invalid_reason = "missing_video"
            if invalid_reason:
                row["is_valid_crop"] = "0"
                row["invalid_reason"] = invalid_reason
                invalid_rows.append(row)
            else:
                valid_rows.append(row)
    return valid_rows, invalid_rows


def validate_bbox_size(row: Dict[str, Any], crop_cfg: Dict[str, Any]) -> str:
    """Return invalid reason for a bbox, or empty string if valid."""
    width = float(row.get("bbox_width", 0.0))
    height = float(row.get("bbox_height", 0.0))
    area = float(row.get("bbox_area", 0.0))
    if width <= 0.0 or height <= 0.0:
        return "invalid_bbox"
    if width < float(crop_cfg.get("min_bbox_width", 12)):
        return "bbox_too_narrow"
    if height < float(crop_cfg.get("min_bbox_height", 24)):
        return "bbox_too_short"
    if area < float(crop_cfg.get("min_bbox_area", 500)):
        return "bbox_area_too_small"
    return ""


def sample_candidates_per_identity(rows: List[Dict[str, Any]], max_crops_per_identity: int) -> List[Dict[str, Any]]:
    """Uniformly sample candidates per identity in temporal order."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("identity_id", "")), []).append(row)
    selected: List[Dict[str, Any]] = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda item: (int(item.get("frame_id", -1)), str(item.get("camera_id", ""))))
        if len(ordered) <= int(max_crops_per_identity):
            selected.extend(ordered)
        else:
            indices = uniform_indices(len(ordered), int(max_crops_per_identity))
            selected.extend([ordered[index] for index in indices])
    return selected


def uniform_indices(length: int, count: int) -> List[int]:
    """Return uniformly spaced integer indices."""
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    values = np.linspace(0, length - 1, int(count))
    output = sorted(set([int(round(value)) for value in values]))
    return output[: int(count)]


def assign_reid_splits(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    """Assign output ReID split in-place."""
    split_strategy = str(config.get("split", {}).get("split_strategy", "scene_split"))
    if split_strategy == "scene_split":
        for row in rows:
            row["split"] = "train" if row.get("source_split") == "train" else "val"
        return

    if split_strategy != "identity_split_train_only":
        raise ValueError("Unsupported split_strategy: %s" % split_strategy)

    identities = sorted(set([str(row.get("identity_id", "")) for row in rows if row.get("source_split") == "train"]))
    fraction = float(config.get("split", {}).get("train_identity_fraction", 0.8))
    cutoff = int(round(float(len(identities)) * fraction))
    train_ids = set(identities[:cutoff])
    val_ids = set(identities[cutoff:])
    include_val_as_eval_only = bool(config.get("crop_extraction", {}).get("include_val_as_eval_only", True))
    for row in rows:
        identity = str(row.get("identity_id", ""))
        if row.get("source_split") != "train":
            row["split"] = "val" if include_val_as_eval_only else "unused_val_scene"
        elif identity in train_ids:
            row["split"] = "train"
        elif identity in val_ids:
            row["split"] = "val"
        else:
            row["split"] = "unused"


def save_selected_crops(
    rows: List[Dict[str, Any]],
    output_root: Path,
    config: Dict[str, Any],
    show_progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> List[Dict[str, Any]]:
    """Read frames lazily, crop bboxes, save images, and return metadata rows."""
    crop_cfg = config.get("crop_extraction", {})
    jpeg_quality = int(crop_cfg.get("jpeg_quality", 95))
    padding_ratio = float(crop_cfg.get("padding_ratio", 0.05))
    saved_rows: List[Dict[str, Any]] = []
    current_key: Optional[Tuple[str, int]] = None
    current_image: Optional[np.ndarray] = None
    ordered = sorted(rows, key=lambda item: (str(item.get("source_video_path", "")), int(item.get("frame_id", -1)), str(item.get("identity_id", ""))))
    seen_crop_ids = set()
    for row in progress_iter(ordered, show_progress, "save Person ReID crops", "crop"):
        output_row = dict(row)
        if str(output_row.get("split", "")) not in ("train", "val"):
            output_row["is_valid_crop"] = "0"
            output_row["invalid_reason"] = "split_not_used"
            saved_rows.append(output_row)
            continue
        crop_id = str(output_row.get("crop_id", ""))
        if crop_id in seen_crop_ids:
            output_row["is_valid_crop"] = "0"
            output_row["invalid_reason"] = "duplicate_crop_id"
            saved_rows.append(output_row)
            continue
        seen_crop_ids.add(crop_id)
        video_path = str(output_row.get("source_video_path", ""))
        frame_id = int(output_row.get("frame_id", -1))
        key = (video_path, frame_id)
        if key != current_key:
            current_image = safe_read_video_frame(Path(video_path), frame_id) if video_path else None
            current_key = key
        if current_image is None:
            output_row["is_valid_crop"] = "0"
            output_row["invalid_reason"] = "missing_frame"
            saved_rows.append(output_row)
            continue
        crop, clipped = crop_image_xyxy(current_image, _bbox_from_row(output_row), padding_ratio=padding_ratio)
        if crop is None or clipped is None:
            output_row["is_valid_crop"] = "0"
            output_row["invalid_reason"] = "invalid_crop_after_clipping"
            saved_rows.append(output_row)
            continue
        crop_path = _crop_path(output_root, output_row)
        if crop_path.exists() and not overwrite and skip_existing:
            pass
        else:
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                output_row["is_valid_crop"] = "0"
                output_row["invalid_reason"] = "write_failed"
                saved_rows.append(output_row)
                continue
        output_row["bbox_x1"] = clipped[0]
        output_row["bbox_y1"] = clipped[1]
        output_row["bbox_x2"] = clipped[2]
        output_row["bbox_y2"] = clipped[3]
        output_row["bbox_width"] = clipped[2] - clipped[0]
        output_row["bbox_height"] = clipped[3] - clipped[1]
        output_row["bbox_area"] = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        output_row["image_width"] = int(current_image.shape[1])
        output_row["image_height"] = int(current_image.shape[0])
        output_row["crop_width"] = int(crop.shape[1])
        output_row["crop_height"] = int(crop.shape[0])
        output_row["crop_path"] = str(crop_path)
        output_row["is_valid_crop"] = "1"
        output_row["invalid_reason"] = ""
        saved_rows.append(output_row)
    return saved_rows


def crop_image_xyxy(
    image: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    padding_ratio: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
    """Crop an RGB image by padded xyxy bbox."""
    height, width = image.shape[:2]
    clipped = expand_and_clip_bbox(bbox_xyxy, width, height, padding_ratio)
    if clipped is None:
        return None, None
    x1, y1, x2, y2 = clipped
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    return crop.copy(), (float(x1), float(y1), float(x2), float(y2))


def expand_and_clip_bbox(
    bbox_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Expand and clip bbox to image limits."""
    x1, y1, x2, y2 = bbox_xyxy
    left = float(min(x1, x2))
    right = float(max(x1, x2))
    top = float(min(y1, y2))
    bottom = float(max(y1, y2))
    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        return None
    pad_x = width * float(padding_ratio)
    pad_y = height * float(padding_ratio)
    left_i = max(0, min(int(round(left - pad_x)), int(image_width) - 1))
    top_i = max(0, min(int(round(top - pad_y)), int(image_height) - 1))
    right_i = max(0, min(int(round(right + pad_x)), int(image_width)))
    bottom_i = max(0, min(int(round(bottom + pad_y)), int(image_height)))
    if right_i <= left_i or bottom_i <= top_i:
        return None
    return (left_i, top_i, right_i, bottom_i)


def write_metadata_outputs(rows: List[Dict[str, Any]], output_root: Path) -> None:
    """Write crop metadata and split summary files."""
    valid_rows = [row for row in rows if str(row.get("is_valid_crop", "")) == "1"]
    metadata_root = output_root / "metadata"
    write_csv_rows(rows, metadata_root / "all_crops.csv", CROP_METADATA_FIELDS)
    write_jsonl(rows, metadata_root / "all_crops.jsonl")
    train_rows = [row for row in valid_rows if row.get("split") == "train"]
    val_rows = [row for row in valid_rows if row.get("split") == "val"]
    write_csv_rows(train_rows, metadata_root / "train_split.csv", CROP_METADATA_FIELDS)
    write_csv_rows(val_rows, metadata_root / "val_split.csv", CROP_METADATA_FIELDS)
    write_identity_camera_scene_tables(valid_rows, metadata_root)
    write_text_lines(sorted(set([str(row.get("identity_id", "")) for row in train_rows])), metadata_root / "train_identities.txt")
    write_text_lines(sorted(set([str(row.get("identity_id", "")) for row in val_rows])), metadata_root / "val_identities.txt")


def write_identity_camera_scene_tables(rows: List[Dict[str, Any]], metadata_root: Path) -> None:
    """Write identity, camera, and scene summary CSV files."""
    identities = _summary_rows(rows, "identity_id")
    cameras = _summary_rows(rows, "camera_id")
    scenes = _summary_rows(rows, "scene_name")
    write_csv_rows(identities, metadata_root / "identities.csv")
    write_csv_rows(cameras, metadata_root / "cameras.csv")
    write_csv_rows(scenes, metadata_root / "scenes.csv")


def _summary_rows(rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(field, "")), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                field: key,
                "num_crops": len(values),
                "num_scenes": len(set([str(row.get("scene_name", "")) for row in values])),
                "num_cameras": len(set([str(row.get("camera_id", "")) for row in values])),
                "num_identities": len(set([str(row.get("identity_id", "")) for row in values])),
            }
        )
    return output


def _scene_specs(config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    specs = []
    splits = config.get("selection", {}).get("splits", {})
    for key in ["train", "val"]:
        split_cfg = splits.get(key, {})
        split_name = str(split_cfg.get("split_name", key))
        for scene_name in split_cfg.get("scenes", []):
            specs.append((key, split_name, str(scene_name)))
    return specs


def _is_target_person(obj: GroundTruthObject, class_id: int, class_name: str) -> bool:
    object_type = str(obj.object_type)
    mapped_id = int(DEFAULT_CLASS_MAPPING.get(object_type, -1))
    return object_type == class_name or mapped_id == int(class_id)


def _video_index(videos_dir: Path) -> Dict[str, Path]:
    output = {}
    for path in list_video_files(videos_dir):
        output[infer_camera_id_from_video_path(path)] = path
    return output


def _candidate_row(
    obj: GroundTruthObject,
    bbox: Tuple[float, float, float, float],
    split_key: str,
    source_split: str,
    scene_name: str,
    camera_id: str,
    gt_path: Path,
    video_path: Optional[Path],
    video_size: Tuple[Any, Any],
) -> Dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    scene_id = scene_name_to_id(scene_name)
    identity = identity_id(scene_name, int(obj.object_id))
    crop_id = "%s_%s_%06d_%d" % (scene_name, camera_id, int(obj.frame_id), int(obj.object_id))
    image_width, image_height = video_size
    return {
        "crop_id": crop_id,
        "split": split_key,
        "source_split": source_split,
        "scene_name": scene_name,
        "scene_id": -1 if scene_id is None else int(scene_id),
        "camera_id": camera_id,
        "frame_id": int(obj.frame_id),
        "class_id": int(DEFAULT_CLASS_MAPPING.get(str(obj.object_type), -1)),
        "class_name": str(obj.object_type),
        "object_id": int(obj.object_id),
        "identity_id": identity,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "image_width": image_width,
        "image_height": image_height,
        "crop_width": "",
        "crop_height": "",
        "crop_path": "",
        "source_video_path": "" if video_path is None else str(video_path),
        "gt_path": str(gt_path),
        "is_valid_crop": "1",
        "invalid_reason": "",
    }


def _invalid_scene_row(split_key: str, source_split: str, scene_name: str, gt_path: Path, reason: str) -> Dict[str, Any]:
    return {
        "crop_id": "%s_missing_gt" % scene_name,
        "split": split_key,
        "source_split": source_split,
        "scene_name": scene_name,
        "scene_id": scene_name_to_id(scene_name),
        "camera_id": "",
        "frame_id": -1,
        "class_id": 0,
        "class_name": "Person",
        "object_id": -1,
        "identity_id": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "bbox_width": "",
        "bbox_height": "",
        "bbox_area": "",
        "image_width": "",
        "image_height": "",
        "crop_width": "",
        "crop_height": "",
        "crop_path": "",
        "source_video_path": "",
        "gt_path": str(gt_path),
        "is_valid_crop": "0",
        "invalid_reason": reason,
    }


def _video_size(video_path: Optional[Path]) -> Tuple[Any, Any]:
    if video_path is None:
        return ("", "")
    resolution = get_video_resolution(video_path)
    if resolution is None:
        return ("", "")
    return resolution


def _bbox_from_row(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(row.get("bbox_x1", 0.0)),
        float(row.get("bbox_y1", 0.0)),
        float(row.get("bbox_x2", 0.0)),
        float(row.get("bbox_y2", 0.0)),
    )


def _crop_path(output_root: Path, row: Dict[str, Any]) -> Path:
    filename = "%s_%s_%06d_%s.jpg" % (
        row.get("scene_name", ""),
        row.get("camera_id", ""),
        int(row.get("frame_id", -1)),
        row.get("object_id", ""),
    )
    return output_root / "crops" / str(row.get("split", "")) / str(row.get("identity_id", "")) / filename
