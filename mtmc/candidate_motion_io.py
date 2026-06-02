"""I/O helpers for candidate motion-quality metrics."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.mtmc.candidate_io import candidate_to_csv_row
from deep_oc_sort_3d.mtmc.candidate_motion_filtering import attach_motion_metrics_to_candidate_dict
from deep_oc_sort_3d.mtmc.candidate_motion_quality import CandidateMotionMetrics
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate


MOTION_FIELDS = [
    "candidate_id",
    "scene_name",
    "split",
    "subset",
    "camera_id",
    "local_track_id",
    "class_id",
    "class_name",
    "length",
    "start_frame",
    "end_frame",
    "num_valid_3d_points",
    "valid_3d_ratio",
    "max_step_distance_3d",
    "mean_step_distance_3d",
    "median_step_distance_3d",
    "p95_step_distance_3d",
    "max_speed_3d",
    "mean_speed_3d",
    "travel_distance_3d_recomputed",
    "straight_line_distance_3d",
    "path_efficiency_3d",
    "travel_distance_per_frame",
    "jump_count",
    "jump_ratio",
    "motion_quality_flag",
    "motion_reject_reason",
    "is_motion_clean",
    "step_distances_3d_json",
]


MOTION_COMPACT_FIELDS = [
    "motion_quality_flag",
    "motion_reject_reason",
    "is_motion_clean",
    "max_step_distance_3d",
    "mean_step_distance_3d",
    "p95_step_distance_3d",
    "max_speed_3d",
    "travel_distance_3d_recomputed",
    "travel_distance_per_frame",
    "jump_count",
    "jump_ratio",
    "num_valid_3d_points",
    "valid_3d_ratio",
]


def write_motion_metrics_csv(metrics: List[CandidateMotionMetrics], path: Path) -> None:
    """Write motion metrics as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOTION_FIELDS)
        writer.writeheader()
        for item in metrics:
            writer.writerow(motion_metrics_to_csv_row(item))


def read_motion_metrics_csv(path: Path) -> List[CandidateMotionMetrics]:
    """Read motion metrics from CSV."""
    if not path.exists():
        return []
    output = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append(motion_metrics_from_csv_row(row))
    return output


def write_motion_metrics_jsonl(metrics: List[CandidateMotionMetrics], path: Path) -> None:
    """Write motion metrics as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(motion_metrics_to_dict(item), sort_keys=True) for item in metrics]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_motion_metrics_jsonl(path: Path) -> List[CandidateMotionMetrics]:
    """Read motion metrics from JSONL."""
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        output.append(motion_metrics_from_dict(json.loads(line)))
    return output


def write_candidates_with_motion_jsonl(
    candidates: List[MTMCTrackletCandidate],
    metrics_by_id: Dict[str, CandidateMotionMetrics],
    path: Path,
) -> None:
    """Write candidate dictionaries augmented with motion fields as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for candidate in candidates:
        metrics = metrics_by_id.get(candidate.candidate_id)
        if metrics is None:
            continue
        lines.append(json.dumps(attach_motion_metrics_to_candidate_dict(candidate, metrics), sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_candidates_with_motion_csv(
    candidates: List[MTMCTrackletCandidate],
    metrics_by_id: Dict[str, CandidateMotionMetrics],
    path: Path,
) -> None:
    """Write compact candidate CSV augmented with motion fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(candidate_to_csv_row(candidates[0]).keys()) + MOTION_COMPACT_FIELDS if candidates else MOTION_COMPACT_FIELDS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            metrics = metrics_by_id.get(candidate.candidate_id)
            row = candidate_to_csv_row(candidate)
            if metrics is not None:
                row.update(_motion_compact_row(metrics))
            writer.writerow(row)


def motion_metrics_to_dict(metrics: CandidateMotionMetrics) -> Dict[str, Any]:
    """Convert metrics to JSON-friendly dictionary."""
    return {
        "candidate_id": metrics.candidate_id,
        "scene_name": metrics.scene_name,
        "split": metrics.split,
        "subset": metrics.subset,
        "camera_id": metrics.camera_id,
        "local_track_id": metrics.local_track_id,
        "class_id": metrics.class_id,
        "class_name": metrics.class_name,
        "length": metrics.length,
        "start_frame": metrics.start_frame,
        "end_frame": metrics.end_frame,
        "num_valid_3d_points": metrics.num_valid_3d_points,
        "valid_3d_ratio": metrics.valid_3d_ratio,
        "max_step_distance_3d": metrics.max_step_distance_3d,
        "mean_step_distance_3d": metrics.mean_step_distance_3d,
        "median_step_distance_3d": metrics.median_step_distance_3d,
        "p95_step_distance_3d": metrics.p95_step_distance_3d,
        "max_speed_3d": metrics.max_speed_3d,
        "mean_speed_3d": metrics.mean_speed_3d,
        "travel_distance_3d_recomputed": metrics.travel_distance_3d_recomputed,
        "straight_line_distance_3d": metrics.straight_line_distance_3d,
        "path_efficiency_3d": metrics.path_efficiency_3d,
        "travel_distance_per_frame": metrics.travel_distance_per_frame,
        "jump_count": metrics.jump_count,
        "jump_ratio": metrics.jump_ratio,
        "motion_quality_flag": metrics.motion_quality_flag,
        "motion_reject_reason": metrics.motion_reject_reason,
        "is_motion_clean": metrics.is_motion_clean,
        "step_distances_3d": [list(item) for item in metrics.step_distances_3d],
    }


def motion_metrics_from_dict(data: Dict[str, Any]) -> CandidateMotionMetrics:
    """Create metrics from dictionary."""
    return CandidateMotionMetrics(
        candidate_id=str(data.get("candidate_id", "")),
        scene_name=str(data.get("scene_name", "")),
        split=str(data.get("split", "")),
        subset=str(data.get("subset", "")),
        camera_id=str(data.get("camera_id", "")),
        local_track_id=int(data.get("local_track_id", -1)),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        length=int(data.get("length", 0)),
        start_frame=int(data.get("start_frame", -1)),
        end_frame=int(data.get("end_frame", -1)),
        num_valid_3d_points=int(data.get("num_valid_3d_points", 0)),
        valid_3d_ratio=float(data.get("valid_3d_ratio", 0.0)),
        max_step_distance_3d=_optional_float(data.get("max_step_distance_3d")),
        mean_step_distance_3d=_optional_float(data.get("mean_step_distance_3d")),
        median_step_distance_3d=_optional_float(data.get("median_step_distance_3d")),
        p95_step_distance_3d=_optional_float(data.get("p95_step_distance_3d")),
        max_speed_3d=_optional_float(data.get("max_speed_3d")),
        mean_speed_3d=_optional_float(data.get("mean_speed_3d")),
        travel_distance_3d_recomputed=_optional_float(data.get("travel_distance_3d_recomputed")),
        straight_line_distance_3d=_optional_float(data.get("straight_line_distance_3d")),
        path_efficiency_3d=_optional_float(data.get("path_efficiency_3d")),
        travel_distance_per_frame=_optional_float(data.get("travel_distance_per_frame")),
        jump_count=int(data.get("jump_count", 0)),
        jump_ratio=float(data.get("jump_ratio", 0.0)),
        motion_quality_flag=str(data.get("motion_quality_flag", "motion_unknown")),
        motion_reject_reason=str(data.get("motion_reject_reason", "")),
        is_motion_clean=_bool(data.get("is_motion_clean", False)),
        step_distances_3d=_step_distances(data.get("step_distances_3d", [])),
    )


def motion_metrics_to_csv_row(metrics: CandidateMotionMetrics) -> Dict[str, Any]:
    """Convert metrics to CSV row."""
    data = motion_metrics_to_dict(metrics)
    data["step_distances_3d_json"] = json.dumps(data.pop("step_distances_3d"))
    return data


def motion_metrics_from_csv_row(row: Dict[str, str]) -> CandidateMotionMetrics:
    """Create metrics from CSV row."""
    data = dict(row)
    data["step_distances_3d"] = json.loads(row.get("step_distances_3d_json") or "[]")
    return motion_metrics_from_dict(data)


def _motion_compact_row(metrics: CandidateMotionMetrics) -> Dict[str, Any]:
    return {
        "motion_quality_flag": metrics.motion_quality_flag,
        "motion_reject_reason": metrics.motion_reject_reason,
        "is_motion_clean": metrics.is_motion_clean,
        "max_step_distance_3d": _empty_if_none(metrics.max_step_distance_3d),
        "mean_step_distance_3d": _empty_if_none(metrics.mean_step_distance_3d),
        "p95_step_distance_3d": _empty_if_none(metrics.p95_step_distance_3d),
        "max_speed_3d": _empty_if_none(metrics.max_speed_3d),
        "travel_distance_3d_recomputed": _empty_if_none(metrics.travel_distance_3d_recomputed),
        "travel_distance_per_frame": _empty_if_none(metrics.travel_distance_per_frame),
        "jump_count": metrics.jump_count,
        "jump_ratio": metrics.jump_ratio,
        "num_valid_3d_points": metrics.num_valid_3d_points,
        "valid_3d_ratio": metrics.valid_3d_ratio,
    }


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _empty_if_none(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _bool(value: Any) -> bool:
    return str(value).lower() in ("true", "1", "yes")


def _step_distances(values: Any) -> List[Tuple[int, int, float, int]]:
    output = []
    for item in values or []:
        if len(item) < 4:
            continue
        output.append((int(item[0]), int(item[1]), float(item[2]), int(item[3])))
    return output
