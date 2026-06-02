"""CSV and JSONL I/O for LocalTracklet objects."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.tracklets.tracklet_types import (
    LocalTracklet,
    local_tracklet_from_dict,
    local_tracklet_to_dict,
)


CSV_FIELDS = [
    "scene_id",
    "scene_name",
    "split",
    "camera_id",
    "local_track_id",
    "class_id",
    "class_name",
    "start_frame",
    "end_frame",
    "length",
    "mean_confidence",
    "median_confidence",
    "max_confidence",
    "bbox_start",
    "bbox_end",
    "bbox_mean",
    "center_3d_start",
    "center_3d_end",
    "center_3d_mean",
    "center_3d_median",
    "dimensions_3d_mean",
    "yaw_mean",
    "majority_gt_object_id",
    "gt_purity",
    "num_gt_ids",
    "gt_id_counts_json",
    "quality_score",
    "quality_flag",
    "is_valid_for_mtmc",
    "notes",
]


def write_tracklets_csv(tracklets: List[LocalTracklet], path: Path) -> None:
    """Write compact tracklet summaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for tracklet in tracklets:
            writer.writerow(tracklet_to_csv_row(tracklet))


def read_tracklets_csv(path: Path) -> List[LocalTracklet]:
    """Read compact tracklet summaries from CSV."""
    if not path.exists():
        return []
    output = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append(tracklet_from_csv_row(row))
    return output


def write_tracklets_jsonl(tracklets: List[LocalTracklet], path: Path) -> None:
    """Write full tracklets as JSONL, including trajectories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(local_tracklet_to_dict(tracklet), sort_keys=True) for tracklet in tracklets]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_tracklets_jsonl(path: Path) -> List[LocalTracklet]:
    """Read full tracklets from JSONL."""
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        output.append(local_tracklet_from_dict(json.loads(line)))
    return output


def read_tracklets_file(path: Path) -> List[LocalTracklet]:
    """Read tracklets from CSV or JSONL based on suffix."""
    if path.suffix.lower() == ".jsonl":
        return read_tracklets_jsonl(path)
    return read_tracklets_csv(path)


def tracklet_to_csv_row(tracklet: LocalTracklet) -> Dict[str, Any]:
    """Convert a tracklet to a compact CSV row."""
    return {
        "scene_id": tracklet.scene_id,
        "scene_name": tracklet.scene_name,
        "split": tracklet.split,
        "camera_id": tracklet.camera_id,
        "local_track_id": tracklet.local_track_id,
        "class_id": tracklet.class_id,
        "class_name": tracklet.class_name,
        "start_frame": tracklet.start_frame,
        "end_frame": tracklet.end_frame,
        "length": tracklet.length,
        "mean_confidence": tracklet.mean_confidence,
        "median_confidence": tracklet.median_confidence,
        "max_confidence": tracklet.max_confidence,
        "bbox_start": _json_or_empty(tracklet.bbox_start),
        "bbox_end": _json_or_empty(tracklet.bbox_end),
        "bbox_mean": _json_or_empty(tracklet.bbox_mean),
        "center_3d_start": _json_or_empty(tracklet.center_3d_start),
        "center_3d_end": _json_or_empty(tracklet.center_3d_end),
        "center_3d_mean": _json_or_empty(tracklet.center_3d_mean),
        "center_3d_median": _json_or_empty(tracklet.center_3d_median),
        "dimensions_3d_mean": _json_or_empty(tracklet.dimensions_3d_mean),
        "yaw_mean": "" if tracklet.yaw_mean is None else tracklet.yaw_mean,
        "majority_gt_object_id": "" if tracklet.majority_gt_object_id is None else tracklet.majority_gt_object_id,
        "gt_purity": "" if tracklet.gt_purity is None else tracklet.gt_purity,
        "num_gt_ids": tracklet.num_gt_ids,
        "gt_id_counts_json": json.dumps(tracklet.gt_id_counts, sort_keys=True),
        "quality_score": tracklet.quality_score,
        "quality_flag": tracklet.quality_flag,
        "is_valid_for_mtmc": tracklet.is_valid_for_mtmc,
        "notes": tracklet.notes,
    }


def tracklet_from_csv_row(row: Dict[str, str]) -> LocalTracklet:
    """Create a LocalTracklet from a compact CSV row."""
    data = {
        "scene_id": _int(row.get("scene_id"), -1),
        "scene_name": row.get("scene_name", ""),
        "split": row.get("split", ""),
        "camera_id": row.get("camera_id", ""),
        "local_track_id": _int(row.get("local_track_id"), -1),
        "class_id": _int(row.get("class_id"), -1),
        "class_name": row.get("class_name", ""),
        "start_frame": _int(row.get("start_frame"), -1),
        "end_frame": _int(row.get("end_frame"), -1),
        "length": _int(row.get("length"), 0),
        "frame_ids": [],
        "detection_ids": [],
        "mean_confidence": _float(row.get("mean_confidence"), 0.0),
        "median_confidence": _float(row.get("median_confidence"), 0.0),
        "max_confidence": _float(row.get("max_confidence"), 0.0),
        "bbox_start": _json_value(row.get("bbox_start")),
        "bbox_end": _json_value(row.get("bbox_end")),
        "bbox_mean": _json_value(row.get("bbox_mean")),
        "center_3d_start": _json_value(row.get("center_3d_start")),
        "center_3d_end": _json_value(row.get("center_3d_end")),
        "center_3d_mean": _json_value(row.get("center_3d_mean")),
        "center_3d_median": _json_value(row.get("center_3d_median")),
        "dimensions_3d_mean": _json_value(row.get("dimensions_3d_mean")),
        "yaw_mean": _optional_float(row.get("yaw_mean")),
        "trajectory_2d": [],
        "trajectory_3d": [],
        "majority_gt_object_id": _optional_int(row.get("majority_gt_object_id")),
        "gt_purity": _optional_float(row.get("gt_purity")),
        "num_gt_ids": _int(row.get("num_gt_ids"), 0),
        "gt_id_counts": _json_dict(row.get("gt_id_counts_json")),
        "quality_score": _float(row.get("quality_score"), 0.0),
        "quality_flag": row.get("quality_flag", "invalid"),
        "is_valid_for_mtmc": str(row.get("is_valid_for_mtmc", "")).lower() in ("true", "1", "yes"),
        "notes": row.get("notes", ""),
    }
    return local_tracklet_from_dict(data)


def _json_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]
    elif isinstance(value, tuple):
        value = [float(item) for item in value]
    return json.dumps(value)


def _json_value(value: Any) -> Optional[Any]:
    if value in (None, ""):
        return None
    return json.loads(value)


def _json_dict(value: Any) -> Dict[str, int]:
    if value in (None, ""):
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        return {}
    return {str(key): int(item) for key, item in data.items()}


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(float(value))


def _float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(float(value))
