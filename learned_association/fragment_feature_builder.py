"""Normalize fragment observations into compact aggregate features."""

import math
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.pair_dataset_io import (
    parse_list,
    safe_float,
    safe_int,
)


def scene_name_to_id(scene_name: str) -> Optional[int]:
    """Extract the numeric warehouse id."""
    try:
        return int(scene_name.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return None


def normalize_fragment_record(
    record: Dict[str, Any],
    source_name: str,
    split: Optional[str] = None,
    scene_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a candidate or tracklet dictionary to the Step 20A schema."""
    # The configured train/val scene split is authoritative. Existing pipeline
    # folders may call the same data "internal_holdout" or "official_val".
    split_value = str(split or record.get("split") or record.get("subset") or "")
    scene_value = str(record.get("scene_name") or scene_name or "")
    camera_id = str(record.get("camera_id") or record.get("camera") or "")
    fragment_id = _first_text(
        record,
        ("fragment_id", "candidate_id", "tracklet_id", "global_track_id", "local_track_id"),
    )
    if not fragment_id:
        fragment_id = "%s__%s__%s" % (scene_value, camera_id, id(record))

    trajectory_2d = _parse_trajectory(record.get("trajectory_2d_sampled") or record.get("trajectory_2d"))
    trajectory_3d = _parse_trajectory(record.get("trajectory_3d_sampled") or record.get("trajectory_3d"))
    observations = build_observations(trajectory_2d, trajectory_3d)
    frame_ids = [safe_int(item.get("frame_id")) for item in observations]
    frame_ids = [value for value in frame_ids if value is not None]
    frame_start = safe_int(
        record.get("frame_start") if record.get("frame_start") is not None else record.get("start_frame"),
        min(frame_ids) if frame_ids else None,
    )
    frame_end = safe_int(
        record.get("frame_end") if record.get("frame_end") is not None else record.get("end_frame"),
        max(frame_ids) if frame_ids else frame_start,
    )
    length = safe_int(
        record.get("length") or record.get("num_observations"),
        len(observations) if observations else _duration(frame_start, frame_end),
    )

    bbox_values = [item["bbox_xyxy"] for item in observations if item.get("bbox_xyxy") is not None]
    point_values = [item["center_3d"] for item in observations if item.get("center_3d") is not None]
    if not bbox_values:
        bbox_values = _record_bboxes(record)
    bbox_stats = summarize_bboxes(bbox_values)
    motion_stats = summarize_points(point_values, frame_ids_from_observations(observations))
    motion_stats.update(_record_point_features(record, motion_stats))
    gt_counts = _gt_counts(record)
    mean_confidence = safe_float(record.get("mean_confidence"), 0.0)
    median_confidence = safe_float(record.get("median_confidence"), mean_confidence)
    min_confidence = safe_float(record.get("min_confidence"), mean_confidence)

    normalized = {
        "fragment_id": str(fragment_id),
        "source": source_name,
        "split": split_value,
        "scene_name": scene_value,
        "scene_id": safe_int(record.get("scene_id"), scene_name_to_id(scene_value)),
        "class_id": safe_int(record.get("class_id"), 0),
        "class_name": str(record.get("class_name") or "Person"),
        "camera_id": camera_id,
        "cameras": record.get("cameras") or ([camera_id] if camera_id else []),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "duration": _duration(frame_start, frame_end),
        "num_observations": length or 0,
        "mean_confidence": mean_confidence,
        "median_confidence": median_confidence,
        "min_confidence": min_confidence,
        "bbox_area_mean": _prefer(record, "bbox_area_mean", bbox_stats.get("bbox_area_mean")),
        "bbox_area_median": _prefer(record, "bbox_area_median", bbox_stats.get("bbox_area_median")),
        "bbox_height_mean": _prefer(record, "bbox_height_mean", bbox_stats.get("bbox_height_mean")),
        "bbox_width_mean": _prefer(record, "bbox_width_mean", bbox_stats.get("bbox_width_mean")),
        "center_x_mean": _prefer(record, "center_x_mean", motion_stats.get("center_x_mean")),
        "center_y_mean": _prefer(record, "center_y_mean", motion_stats.get("center_y_mean")),
        "center_z_mean": _prefer(record, "center_z_mean", motion_stats.get("center_z_mean")),
        "center_x_start": _prefer(record, "center_x_start", motion_stats.get("center_x_start")),
        "center_y_start": _prefer(record, "center_y_start", motion_stats.get("center_y_start")),
        "center_z_start": _prefer(record, "center_z_start", motion_stats.get("center_z_start")),
        "center_x_end": _prefer(record, "center_x_end", motion_stats.get("center_x_end")),
        "center_y_end": _prefer(record, "center_y_end", motion_stats.get("center_y_end")),
        "center_z_end": _prefer(record, "center_z_end", motion_stats.get("center_z_end")),
        "velocity_x": _prefer(record, "velocity_x", motion_stats.get("velocity_x")),
        "velocity_y": _prefer(record, "velocity_y", motion_stats.get("velocity_y")),
        "velocity_z": _prefer(record, "velocity_z", motion_stats.get("velocity_z")),
        "speed_mean": _prefer(record, "speed_mean", motion_stats.get("speed_mean")),
        "step_p95": _prefer(record, "step_p95", motion_stats.get("step_p95")),
        "step_max": _prefer(record, "step_max", motion_stats.get("step_max")),
        "global_track_id": record.get("global_track_id"),
        "local_track_id": record.get("local_track_id"),
        "tracklet_id": record.get("tracklet_id"),
        "candidate_id": record.get("candidate_id"),
        "gt_identity_id": "unknown",
        "gt_object_id": None,
        "gt_match_count": 0,
        "gt_match_ratio": 0.0,
        "gt_purity": 0.0,
        "valid_for_pairs": False,
        "invalid_reason": "gt_not_matched",
        "embedding_available": False,
        "embedding_index": None,
        "fragment_quality": record.get("quality_flag") or record.get("fragment_quality") or "unknown",
        "_observations": observations,
        "_embedding": None,
        "_pipeline_gt_object_id": record.get("majority_gt_object_id"),
        "_pipeline_gt_purity": safe_float(record.get("gt_purity")),
        "_pipeline_gt_match_count": (
            sum(gt_counts.values())
            if gt_counts
            else safe_int(record.get("gt_match_count"))
        ),
        "_pipeline_gt_counts": gt_counts,
    }
    return normalized


def aggregate_observation_rows(
    rows: Sequence[Dict[str, Any]], source_name: str, split: str, scene_name: str
) -> List[Dict[str, Any]]:
    """Aggregate frame-level rows into local/global fragment records."""
    groups = {}  # type: Dict[Tuple[str, str], List[Dict[str, Any]]]
    for row in rows:
        if not _is_person(row):
            continue
        camera_id = str(row.get("camera_id") or row.get("camera") or "")
        track_id = _first_text(row, ("global_track_id", "local_track_id", "track_id", "object_id"))
        if not track_id:
            continue
        groups.setdefault((camera_id, track_id), []).append(row)

    fragments = []  # type: List[Dict[str, Any]]
    for (camera_id, track_id), group_rows in groups.items():
        ordered = sorted(group_rows, key=lambda item: safe_int(item.get("frame_id"), 0) or 0)
        trajectory_2d = []
        trajectory_3d = []
        confidences = []
        for row in ordered:
            frame_id = safe_int(row.get("frame_id"))
            bbox = _bbox_from_row(row)
            point = _point_from_row(row)
            if frame_id is not None and bbox is not None:
                trajectory_2d.append([frame_id] + bbox)
            if frame_id is not None and point is not None:
                trajectory_3d.append([frame_id] + point)
            confidence = safe_float(row.get("confidence"))
            if confidence is not None:
                confidences.append(confidence)
        base = dict(ordered[0])
        base.update(
            {
                "fragment_id": "%s__%s__%s__%s" % (source_name, scene_name, camera_id, track_id),
                "split": split,
                "scene_name": scene_name,
                "camera_id": camera_id,
                "frame_start": safe_int(ordered[0].get("frame_id")),
                "frame_end": safe_int(ordered[-1].get("frame_id")),
                "length": len(ordered),
                "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "median_confidence": float(np.median(confidences)) if confidences else 0.0,
                "min_confidence": min(confidences) if confidences else 0.0,
                "trajectory_2d": trajectory_2d,
                "trajectory_3d": trajectory_3d,
            }
        )
        fragments.append(normalize_fragment_record(base, source_name, split, scene_name))
    return fragments


def build_observations(
    trajectory_2d: Sequence[Sequence[Any]], trajectory_3d: Sequence[Sequence[Any]]
) -> List[Dict[str, Any]]:
    """Merge sampled 2D and 3D trajectories by frame id."""
    by_frame = {}  # type: Dict[int, Dict[str, Any]]
    for row in trajectory_2d:
        if len(row) < 5:
            continue
        frame_id = safe_int(row[0])
        bbox = [safe_float(value) for value in row[1:5]]
        if frame_id is not None and all(value is not None for value in bbox):
            by_frame.setdefault(frame_id, {"frame_id": frame_id})["bbox_xyxy"] = bbox
    for row in trajectory_3d:
        if len(row) < 4:
            continue
        frame_id = safe_int(row[0])
        point = [safe_float(value) for value in row[1:4]]
        if frame_id is not None and all(value is not None for value in point):
            by_frame.setdefault(frame_id, {"frame_id": frame_id})["center_3d"] = point
    return [by_frame[key] for key in sorted(by_frame.keys())]


def summarize_bboxes(bboxes: Sequence[Sequence[float]]) -> Dict[str, Optional[float]]:
    """Compute simple bbox statistics."""
    widths = [max(0.0, float(box[2]) - float(box[0])) for box in bboxes if len(box) >= 4]
    heights = [max(0.0, float(box[3]) - float(box[1])) for box in bboxes if len(box) >= 4]
    areas = [width * height for width, height in zip(widths, heights)]
    return {
        "bbox_area_mean": _mean(areas),
        "bbox_area_median": float(median(areas)) if areas else None,
        "bbox_height_mean": _mean(heights),
        "bbox_width_mean": _mean(widths),
    }


def summarize_points(
    points: Sequence[Sequence[float]], frame_ids: Sequence[int]
) -> Dict[str, Optional[float]]:
    """Compute center, velocity and step-distance statistics."""
    valid = [np.asarray(point[:3], dtype=np.float64) for point in points if len(point) >= 3]
    if not valid:
        return {}
    array = np.stack(valid, axis=0)
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1) if len(array) > 1 else np.asarray([])
    frame_delta = max(1, (frame_ids[-1] - frame_ids[0]) if len(frame_ids) >= 2 else len(array) - 1)
    velocity = (array[-1] - array[0]) / float(frame_delta)
    return {
        "center_x_mean": float(np.mean(array[:, 0])),
        "center_y_mean": float(np.mean(array[:, 1])),
        "center_z_mean": float(np.mean(array[:, 2])),
        "center_x_start": float(array[0, 0]),
        "center_y_start": float(array[0, 1]),
        "center_z_start": float(array[0, 2]),
        "center_x_end": float(array[-1, 0]),
        "center_y_end": float(array[-1, 1]),
        "center_z_end": float(array[-1, 2]),
        "velocity_x": float(velocity[0]),
        "velocity_y": float(velocity[1]),
        "velocity_z": float(velocity[2]),
        "speed_mean": float(np.mean(steps)) if len(steps) else 0.0,
        "step_p95": float(np.percentile(steps, 95)) if len(steps) else 0.0,
        "step_max": float(np.max(steps)) if len(steps) else 0.0,
    }


def frame_ids_from_observations(observations: Sequence[Dict[str, Any]]) -> List[int]:
    """Return frame ids for observations that contain 3D points."""
    result = []
    for item in observations:
        if item.get("center_3d") is None:
            continue
        frame_id = safe_int(item.get("frame_id"))
        if frame_id is not None:
            result.append(frame_id)
    return result


def _parse_trajectory(value: Any) -> List[List[Any]]:
    parsed = parse_list(value)
    return [list(item) for item in parsed if isinstance(item, (list, tuple))]


def _record_bboxes(record: Dict[str, Any]) -> List[List[float]]:
    """Read compact candidate bbox summaries when trajectories are unavailable."""
    result = []
    for key in ("bbox_start", "bbox_end", "bbox_mean"):
        values = parse_list(record.get(key))
        parsed = [safe_float(value) for value in values[:4]]
        if len(parsed) == 4 and all(value is not None for value in parsed):
            result.append([float(value) for value in parsed if value is not None])
    return result


def _record_point_features(
    record: Dict[str, Any], existing: Dict[str, Optional[float]]
) -> Dict[str, Optional[float]]:
    """Read compact candidate center and velocity vectors."""
    result = dict(existing)
    vector_fields = {
        "start": ("center_3d_start", "entry_center_3d"),
        "end": ("center_3d_end", "exit_center_3d"),
        "mean": ("center_3d_mean",),
    }
    for suffix, keys in vector_fields.items():
        vector = _first_vector(record, keys)
        if vector is None:
            continue
        for axis, value in zip(("x", "y", "z"), vector):
            result.setdefault("center_%s_%s" % (axis, suffix), value)
    velocity = _first_vector(record, ("mean_velocity_3d", "velocity_3d"))
    if velocity is not None:
        for axis, value in zip(("x", "y", "z"), velocity):
            result.setdefault("velocity_%s" % axis, value)
        result.setdefault("speed_mean", float(np.linalg.norm(np.asarray(velocity))))
    aliases = {
        "step_p95": "p95_step_distance_3d",
        "step_max": "max_step_distance_3d",
        "speed_mean": "mean_speed_3d",
    }
    for target, source in aliases.items():
        value = safe_float(record.get(source))
        if value is not None:
            result.setdefault(target, value)
    return result


def _first_vector(
    record: Dict[str, Any], keys: Sequence[str]
) -> Optional[List[float]]:
    for key in keys:
        values = parse_list(record.get(key))
        parsed = [safe_float(value) for value in values[:3]]
        if len(parsed) == 3 and all(value is not None for value in parsed):
            return [float(value) for value in parsed if value is not None]
    return None


def _gt_counts(record: Dict[str, Any]) -> Dict[str, int]:
    """Read per-GT observation counts from JSONL or compact CSV records."""
    value = record.get("gt_id_counts")
    if not isinstance(value, dict):
        raw = record.get("gt_id_counts_json")
        if isinstance(raw, str) and raw:
            try:
                import json

                value = json.loads(raw)
            except (TypeError, ValueError):
                value = {}
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        count = safe_int(item)
        if count is not None and count > 0:
            result[str(key)] = count
    return result


def _first_text(record: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def _duration(start: Optional[int], end: Optional[int]) -> int:
    if start is None or end is None:
        return 0
    return max(0, end - start + 1)


def _prefer(record: Dict[str, Any], key: str, fallback: Optional[float]) -> Optional[float]:
    return safe_float(record.get(key), fallback)


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / float(len(values))) if values else None


def _is_person(row: Dict[str, Any]) -> bool:
    class_id = safe_int(row.get("class_id"))
    class_name = str(row.get("class_name") or row.get("object_type") or "")
    return class_id == 0 or class_name.lower() == "person"


def _bbox_from_row(row: Dict[str, Any]) -> Optional[List[float]]:
    key_sets = (("x1", "y1", "x2", "y2"), ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    for keys in key_sets:
        values = [safe_float(row.get(key)) for key in keys]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    bbox = parse_list(row.get("bbox_xyxy"))
    if len(bbox) >= 4:
        values = [safe_float(value) for value in bbox[:4]]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    return None


def _point_from_row(row: Dict[str, Any]) -> Optional[List[float]]:
    for keys in (("center_x", "center_y", "center_z"), ("x", "y", "z")):
        values = [safe_float(row.get(key)) for key in keys]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    return None
