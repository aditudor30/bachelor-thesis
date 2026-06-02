"""CSV and JSONL I/O for MTMC candidates."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import (
    MTMCTrackletCandidate,
    candidate_from_dict,
    candidate_to_dict,
)


CSV_FIELDS = [
    "scene_id",
    "scene_name",
    "split",
    "subset",
    "camera_id",
    "local_track_id",
    "candidate_id",
    "class_id",
    "class_name",
    "start_frame",
    "end_frame",
    "length",
    "duration",
    "mean_confidence",
    "median_confidence",
    "max_confidence",
    "quality_score",
    "quality_flag",
    "source_tracklet_valid_for_mtmc",
    "is_candidate",
    "reject_reason",
    "has_3d",
    "trajectory_3d_length",
    "entry_center_3d",
    "exit_center_3d",
    "center_3d_mean",
    "mean_velocity_3d",
    "travel_distance_3d",
    "majority_gt_object_id",
    "gt_purity",
    "num_gt_ids",
    "gt_id_counts_json",
    "reid_embedding_path",
    "global_track_id",
]


def write_candidates_csv(candidates: List[MTMCTrackletCandidate], path: Path) -> None:
    """Write compact candidate summaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate_to_csv_row(candidate))


def read_candidates_csv(path: Path) -> List[MTMCTrackletCandidate]:
    """Read compact candidate summaries from CSV."""
    if not path.exists():
        return []
    output = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append(candidate_from_csv_row(row))
    return output


def write_candidates_jsonl(
    candidates: List[MTMCTrackletCandidate],
    path: Path,
    only_candidates: bool = False,
) -> None:
    """Write full candidate records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [item for item in candidates if item.is_candidate] if only_candidates else candidates
    lines = [json.dumps(candidate_to_dict(candidate), sort_keys=True) for candidate in selected]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_candidates_jsonl(path: Path) -> List[MTMCTrackletCandidate]:
    """Read full candidate records from JSONL."""
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        output.append(candidate_from_dict(json.loads(line)))
    return output


def read_candidates_file(path: Path) -> List[MTMCTrackletCandidate]:
    """Read candidates from CSV or JSONL based on suffix."""
    if path.suffix.lower() == ".jsonl":
        return read_candidates_jsonl(path)
    return read_candidates_csv(path)


def candidate_to_csv_row(candidate: MTMCTrackletCandidate) -> Dict[str, Any]:
    """Convert a candidate to a compact CSV row."""
    return {
        "scene_id": candidate.scene_id,
        "scene_name": candidate.scene_name,
        "split": candidate.split,
        "subset": candidate.subset,
        "camera_id": candidate.camera_id,
        "local_track_id": candidate.local_track_id,
        "candidate_id": candidate.candidate_id,
        "class_id": candidate.class_id,
        "class_name": candidate.class_name,
        "start_frame": candidate.start_frame,
        "end_frame": candidate.end_frame,
        "length": candidate.length,
        "duration": candidate.duration,
        "mean_confidence": candidate.mean_confidence,
        "median_confidence": candidate.median_confidence,
        "max_confidence": candidate.max_confidence,
        "quality_score": candidate.quality_score,
        "quality_flag": candidate.quality_flag,
        "source_tracklet_valid_for_mtmc": candidate.source_tracklet_valid_for_mtmc,
        "is_candidate": candidate.is_candidate,
        "reject_reason": "" if candidate.reject_reason is None else candidate.reject_reason,
        "has_3d": candidate.has_3d,
        "trajectory_3d_length": candidate.trajectory_3d_length,
        "entry_center_3d": _json_or_empty(candidate.entry_center_3d),
        "exit_center_3d": _json_or_empty(candidate.exit_center_3d),
        "center_3d_mean": _json_or_empty(candidate.center_3d_mean),
        "mean_velocity_3d": _json_or_empty(candidate.mean_velocity_3d),
        "travel_distance_3d": "" if candidate.travel_distance_3d is None else candidate.travel_distance_3d,
        "majority_gt_object_id": "" if candidate.majority_gt_object_id is None else candidate.majority_gt_object_id,
        "gt_purity": "" if candidate.gt_purity is None else candidate.gt_purity,
        "num_gt_ids": candidate.num_gt_ids,
        "gt_id_counts_json": json.dumps(candidate.gt_id_counts, sort_keys=True),
        "reid_embedding_path": "" if candidate.reid_embedding_path is None else candidate.reid_embedding_path,
        "global_track_id": "" if candidate.global_track_id is None else candidate.global_track_id,
    }


def candidate_from_csv_row(row: Dict[str, str]) -> MTMCTrackletCandidate:
    """Create a candidate from a compact CSV row."""
    data = {
        "candidate_id": row.get("candidate_id", ""),
        "scene_id": _int(row.get("scene_id"), -1),
        "scene_name": row.get("scene_name", ""),
        "split": row.get("split", ""),
        "subset": row.get("subset", ""),
        "camera_id": row.get("camera_id", ""),
        "local_track_id": _int(row.get("local_track_id"), -1),
        "class_id": _int(row.get("class_id"), -1),
        "class_name": row.get("class_name", ""),
        "start_frame": _int(row.get("start_frame"), -1),
        "end_frame": _int(row.get("end_frame"), -1),
        "length": _int(row.get("length"), 0),
        "duration": _int(row.get("duration"), 0),
        "mean_confidence": _float(row.get("mean_confidence"), 0.0),
        "median_confidence": _float(row.get("median_confidence"), 0.0),
        "max_confidence": _float(row.get("max_confidence"), 0.0),
        "quality_score": _float(row.get("quality_score"), 0.0),
        "quality_flag": row.get("quality_flag", ""),
        "source_tracklet_valid_for_mtmc": _bool(row.get("source_tracklet_valid_for_mtmc")),
        "is_candidate": _bool(row.get("is_candidate")),
        "reject_reason": _optional_str(row.get("reject_reason")),
        "bbox_start": None,
        "bbox_end": None,
        "bbox_mean": None,
        "center_3d_start": None,
        "center_3d_end": None,
        "center_3d_mean": _json_value(row.get("center_3d_mean")),
        "center_3d_median": None,
        "trajectory_2d_sampled": [],
        "trajectory_3d_sampled": [],
        "trajectory_3d_length": _int(row.get("trajectory_3d_length"), 0),
        "has_3d": _bool(row.get("has_3d")),
        "entry_frame": _int(row.get("start_frame"), -1),
        "exit_frame": _int(row.get("end_frame"), -1),
        "entry_center_3d": _json_value(row.get("entry_center_3d")),
        "exit_center_3d": _json_value(row.get("exit_center_3d")),
        "mean_velocity_3d": _json_value(row.get("mean_velocity_3d")),
        "travel_distance_3d": _optional_float(row.get("travel_distance_3d")),
        "majority_gt_object_id": _optional_int(row.get("majority_gt_object_id")),
        "gt_purity": _optional_float(row.get("gt_purity")),
        "num_gt_ids": _int(row.get("num_gt_ids"), 0),
        "gt_id_counts": _json_dict(row.get("gt_id_counts_json")),
        "reid_embedding_path": _optional_str(row.get("reid_embedding_path")),
        "reid_embedding": None,
        "global_track_id": _optional_int(row.get("global_track_id")),
    }
    return candidate_from_dict(data)


def _json_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]
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


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _bool(value: Any) -> bool:
    return str(value).lower() in ("true", "1", "yes")
