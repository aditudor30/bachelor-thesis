"""Lazy RGB crop extraction helpers for ReID embeddings."""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.reid.reid_types import ReIDCropSample


def clamp_bbox_xyxy(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp an xyxy bbox to image bounds and return integer coordinates."""
    x1, y1, x2, y2 = bbox
    x1_i = max(0, min(int(round(float(x1))), int(image_width) - 1))
    y1_i = max(0, min(int(round(float(y1))), int(image_height) - 1))
    x2_i = max(0, min(int(round(float(x2))), int(image_width)))
    y2_i = max(0, min(int(round(float(y2))), int(image_height)))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return (x1_i, y1_i, x2_i, y2_i)


def expand_bbox(
    bbox,
    padding_ratio: float,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Expand an xyxy bbox by a padding ratio and clamp to image bounds."""
    x1, y1, x2, y2 = bbox
    width = float(x2) - float(x1)
    height = float(y2) - float(y1)
    if width <= 0.0 or height <= 0.0:
        return None
    pad_x = width * float(padding_ratio)
    pad_y = height * float(padding_ratio)
    expanded = (float(x1) - pad_x, float(y1) - pad_y, float(x2) + pad_x, float(y2) + pad_y)
    return clamp_bbox_xyxy(expanded, image_width, image_height)


def crop_image_by_bbox(image, bbox_xyxy) -> Optional[np.ndarray]:
    """Crop an RGB image by an xyxy bbox."""
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim < 2:
        return None
    height, width = arr.shape[:2]
    clamped = clamp_bbox_xyxy(bbox_xyxy, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def sample_frame_records_for_track(
    records: List[Any],
    max_crops: int = 8,
    strategy: str = "uniform",
) -> List[Any]:
    """Sample frame records from one tracklet for crop extraction."""
    ordered = sorted(records, key=lambda item: _record_int(item, "frame_id", -1))
    if len(ordered) <= int(max_crops):
        return ordered
    if strategy == "highest_confidence":
        selected = sorted(ordered, key=lambda item: _record_float(item, "confidence", 0.0), reverse=True)[: int(max_crops)]
        return sorted(selected, key=lambda item: _record_int(item, "frame_id", -1))
    if strategy == "first_middle_last":
        indices = [0, len(ordered) // 2, len(ordered) - 1]
        extra = max(0, int(max_crops) - len(indices))
        if extra > 0:
            uniform = _uniform_indices(len(ordered), extra)
            indices.extend(uniform)
        return [ordered[index] for index in sorted(set(indices))[: int(max_crops)]]
    indices = _uniform_indices(len(ordered), int(max_crops))
    return [ordered[index] for index in indices]


def extract_crops_for_tracklet(
    records: List[Any],
    frame_loader: Callable[[int], Optional[np.ndarray]],
    max_crops: int,
    padding_ratio: float,
    resize: Tuple[int, int],
    save_crops: bool,
    crop_output_dir: Optional[Path],
) -> Tuple[List[np.ndarray], List[ReIDCropSample]]:
    """Extract RGB crops for one tracklet using a lazy frame loader."""
    sampled = sample_frame_records_for_track(records, max_crops=max_crops, strategy="uniform")
    crops = []
    samples = []
    for record in sampled:
        frame_id = _record_int(record, "frame_id", -1)
        image = frame_loader(frame_id)
        if image is None:
            continue
        bbox = _record_bbox(record)
        if bbox is None:
            continue
        height, width = image.shape[:2]
        expanded = expand_bbox(bbox, padding_ratio, width, height)
        if expanded is None:
            continue
        crop = crop_image_by_bbox(image, expanded)
        if crop is None:
            continue
        resized = cv2.resize(crop, (int(resize[0]), int(resize[1])), interpolation=cv2.INTER_LINEAR)
        crop_path = None
        sample = _record_to_crop_sample(record, expanded, crop_path)
        if save_crops and crop_output_dir is not None:
            crop_output_dir.mkdir(parents=True, exist_ok=True)
            crop_path = str(crop_output_dir / _crop_filename(sample))
            cv2.imwrite(crop_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            sample.crop_path = crop_path
        crops.append(resized)
        samples.append(sample)
    return crops, samples


def _uniform_indices(length: int, count: int) -> List[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    values = np.linspace(0, length - 1, count)
    return sorted(set([int(round(value)) for value in values]))[:count]


def _record_to_crop_sample(record: Any, bbox, crop_path: Optional[str]) -> ReIDCropSample:
    return ReIDCropSample(
        subset=str(_record_value(record, "subset", "")),
        split=str(_record_value(record, "split", "")),
        scene_name=str(_record_value(record, "scene_name", "")),
        camera_id=str(_record_value(record, "camera_id", "")),
        frame_id=_record_int(record, "frame_id", -1),
        local_track_id=_record_int(record, "local_track_id", -1),
        global_track_id=_record_optional_int(record, "global_track_id"),
        candidate_id=_record_optional_str(record, "candidate_id"),
        class_id=_record_int(record, "class_id", -1),
        class_name=str(_record_value(record, "class_name", "")),
        bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        crop_path=crop_path,
        confidence=_record_float(record, "confidence", 0.0),
        source=str(_record_value(record, "source", "")),
    )


def _crop_filename(sample: ReIDCropSample) -> str:
    candidate = sample.candidate_id if sample.candidate_id not in (None, "") else "track_%s" % sample.local_track_id
    return "%s_%s_%s_%s_frame_%06d.jpg" % (
        sample.subset,
        sample.scene_name,
        sample.camera_id,
        str(candidate),
        int(sample.frame_id),
    )


def _record_bbox(record: Any) -> Optional[Tuple[float, float, float, float]]:
    try:
        return (
            _record_float(record, "x1", 0.0),
            _record_float(record, "y1", 0.0),
            _record_float(record, "x2", 0.0),
            _record_float(record, "y2", 0.0),
        )
    except (TypeError, ValueError):
        return None


def _record_value(record: Any, key: str, default: Any) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _record_int(record: Any, key: str, default: int) -> int:
    value = _record_value(record, key, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _record_optional_int(record: Any, key: str) -> Optional[int]:
    value = _record_value(record, key, None)
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _record_optional_str(record: Any, key: str) -> Optional[str]:
    value = _record_value(record, key, None)
    if value in (None, ""):
        return None
    return str(value)


def _record_float(record: Any, key: str, default: float) -> float:
    value = _record_value(record, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

