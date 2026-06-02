"""Motion-quality metrics for MTMC tracklet candidates."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate


@dataclass
class CandidateMotionMetrics:
    """Motion diagnostics computed from one MTMC candidate trajectory."""

    candidate_id: str
    scene_name: str
    split: str
    subset: str
    camera_id: str
    local_track_id: int
    class_id: int
    class_name: str
    length: int
    start_frame: int
    end_frame: int
    num_valid_3d_points: int
    valid_3d_ratio: float
    max_step_distance_3d: Optional[float]
    mean_step_distance_3d: Optional[float]
    median_step_distance_3d: Optional[float]
    p95_step_distance_3d: Optional[float]
    max_speed_3d: Optional[float]
    mean_speed_3d: Optional[float]
    travel_distance_3d_recomputed: Optional[float]
    straight_line_distance_3d: Optional[float]
    path_efficiency_3d: Optional[float]
    travel_distance_per_frame: Optional[float]
    jump_count: int
    jump_ratio: float
    motion_quality_flag: str
    motion_reject_reason: str
    is_motion_clean: bool
    step_distances_3d: List[Tuple[int, int, float, int]]


def default_motion_quality_config() -> Dict[str, Any]:
    """Return default motion-quality thresholds."""
    general = {
        "suspicious_step_m": 3.0,
        "invalid_step_m": 6.0,
        "suspicious_speed_m_per_frame": 3.0,
        "invalid_speed_m_per_frame": 6.0,
        "suspicious_travel_per_frame": 1.5,
        "invalid_travel_per_frame": 3.0,
        "max_jump_ratio_suspicious": 0.10,
        "max_jump_ratio_invalid": 0.25,
    }
    return {
        "require_3d_motion": True,
        "allow_suspicious_as_clean": False,
        "min_valid_3d_points": 3,
        "min_valid_3d_ratio": 0.5,
        "general_thresholds": general,
        "per_class_thresholds": {
            "Person": {"suspicious_step_m": 3.0, "invalid_step_m": 6.0},
            "FourierGR1T2": {"suspicious_step_m": 3.0, "invalid_step_m": 6.0},
            "AgilityDigit": {"suspicious_step_m": 3.0, "invalid_step_m": 6.0},
            "Forklift": {"suspicious_step_m": 4.0, "invalid_step_m": 8.0},
            "PalletTruck": {"suspicious_step_m": 4.0, "invalid_step_m": 8.0},
            "Transporter": {"suspicious_step_m": 4.0, "invalid_step_m": 8.0},
            "NovaCarter": {"suspicious_step_m": 4.0, "invalid_step_m": 8.0},
        },
    }


def extract_3d_trajectory_points(candidate: MTMCTrackletCandidate) -> List[Tuple[int, np.ndarray]]:
    """Extract valid sampled 3D trajectory points from a candidate."""
    points = []
    for item in candidate.trajectory_3d_sampled:
        if len(item) < 4:
            continue
        frame_id = int(item[0])
        center = np.asarray([float(item[1]), float(item[2]), float(item[3])], dtype=float)
        if not np.all(np.isfinite(center)):
            continue
        points.append((frame_id, center))
    return sorted(points, key=lambda item: item[0])


def compute_step_distances_3d(points: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, int, float, int]]:
    """Compute step distances as frame_a, frame_b, distance_m, frame_gap."""
    output = []
    if len(points) < 2:
        return output
    for index in range(1, len(points)):
        frame_a, center_a = points[index - 1]
        frame_b, center_b = points[index]
        frame_gap = max(int(frame_b) - int(frame_a), 1)
        distance = float(np.linalg.norm(center_b - center_a))
        output.append((int(frame_a), int(frame_b), distance, int(frame_gap)))
    return output


def compute_candidate_motion_metrics(
    candidate: MTMCTrackletCandidate,
    config: Optional[Dict[str, Any]] = None,
) -> CandidateMotionMetrics:
    """Compute motion-quality metrics for one candidate."""
    cfg = merge_motion_quality_config(config)
    points = extract_3d_trajectory_points(candidate)
    steps = compute_step_distances_3d(points)
    distances = [float(item[2]) for item in steps]
    speeds = [float(item[2]) / float(max(int(item[3]), 1)) for item in steps]
    valid_ratio = _valid_3d_ratio(candidate, len(points))
    travel = float(sum(distances)) if distances else None
    straight = _straight_line_distance(points)
    duration = max(int(candidate.end_frame) - int(candidate.start_frame) + 1, 1)
    travel_per_frame = None if travel is None else float(travel) / float(duration)
    thresholds = _thresholds_for_class(str(candidate.class_name), cfg)
    jump_count = _jump_count(steps, thresholds)
    jump_ratio = float(jump_count) / float(len(steps)) if steps else 0.0

    metric = CandidateMotionMetrics(
        candidate_id=str(candidate.candidate_id),
        scene_name=str(candidate.scene_name),
        split=str(candidate.split),
        subset=str(candidate.subset),
        camera_id=str(candidate.camera_id),
        local_track_id=int(candidate.local_track_id),
        class_id=int(candidate.class_id),
        class_name=str(candidate.class_name),
        length=int(candidate.length),
        start_frame=int(candidate.start_frame),
        end_frame=int(candidate.end_frame),
        num_valid_3d_points=len(points),
        valid_3d_ratio=float(valid_ratio),
        max_step_distance_3d=_max(distances),
        mean_step_distance_3d=_mean(distances),
        median_step_distance_3d=_median(distances),
        p95_step_distance_3d=_percentile(distances, 95),
        max_speed_3d=_max(speeds),
        mean_speed_3d=_mean(speeds),
        travel_distance_3d_recomputed=travel,
        straight_line_distance_3d=straight,
        path_efficiency_3d=_path_efficiency(straight, travel),
        travel_distance_per_frame=travel_per_frame,
        jump_count=int(jump_count),
        jump_ratio=float(jump_ratio),
        motion_quality_flag="motion_unknown",
        motion_reject_reason="not_classified",
        is_motion_clean=False,
        step_distances_3d=steps,
    )
    flag, is_clean, reason = classify_motion_quality(metric, str(candidate.class_name), cfg)
    metric.motion_quality_flag = flag
    metric.is_motion_clean = bool(is_clean)
    metric.motion_reject_reason = reason
    return metric


def classify_motion_quality(
    metrics: CandidateMotionMetrics,
    class_name: str,
    config: Dict[str, Any],
) -> Tuple[str, bool, str]:
    """Classify motion quality from computed metrics and thresholds."""
    cfg = merge_motion_quality_config(config)
    thresholds = _thresholds_for_class(class_name, cfg)
    require_3d = bool(cfg.get("require_3d_motion", True))
    allow_suspicious = bool(cfg.get("allow_suspicious_as_clean", False))
    min_points = int(cfg.get("min_valid_3d_points", 3))
    min_ratio = float(cfg.get("min_valid_3d_ratio", 0.5))

    if int(metrics.num_valid_3d_points) < min_points:
        return "motion_unknown", not require_3d, "not_enough_3d_points"
    if float(metrics.valid_3d_ratio) < min_ratio:
        return "motion_invalid", False, "low_valid_3d_ratio"

    if _greater_equal(metrics.max_step_distance_3d, thresholds["invalid_step_m"]):
        return "motion_invalid", False, "invalid_step_distance"
    if _greater_equal(metrics.max_speed_3d, thresholds["invalid_speed_m_per_frame"]):
        return "motion_invalid", False, "invalid_speed"
    if _greater_equal(metrics.jump_ratio, thresholds["max_jump_ratio_invalid"]):
        return "motion_invalid", False, "invalid_jump_ratio"
    if _greater_equal(metrics.travel_distance_per_frame, thresholds["invalid_travel_per_frame"]):
        return "motion_invalid", False, "invalid_travel_per_frame"

    if _greater_equal(metrics.max_step_distance_3d, thresholds["suspicious_step_m"]):
        return "motion_suspicious", allow_suspicious, "suspicious_step_distance"
    if _greater_equal(metrics.max_speed_3d, thresholds["suspicious_speed_m_per_frame"]):
        return "motion_suspicious", allow_suspicious, "suspicious_speed"
    if _greater_equal(metrics.jump_ratio, thresholds["max_jump_ratio_suspicious"]):
        return "motion_suspicious", allow_suspicious, "suspicious_jump_ratio"
    if _greater_equal(metrics.travel_distance_per_frame, thresholds["suspicious_travel_per_frame"]):
        return "motion_suspicious", allow_suspicious, "suspicious_travel_per_frame"

    return "motion_good", True, "ok"


def merge_motion_quality_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a partial config over defaults."""
    base = default_motion_quality_config()
    if not config:
        return base
    for key, value in config.items():
        if key in ("general_thresholds", "per_class_thresholds") and isinstance(value, dict):
            merged = dict(base.get(key, {}))
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and isinstance(merged.get(sub_key), dict):
                    item = dict(merged[sub_key])
                    item.update(sub_value)
                    merged[sub_key] = item
                else:
                    merged[sub_key] = sub_value
            base[key] = merged
        else:
            base[key] = value
    return base


def _thresholds_for_class(class_name: str, config: Dict[str, Any]) -> Dict[str, float]:
    general = dict(config.get("general_thresholds", {}))
    per_class = config.get("per_class_thresholds", {})
    if isinstance(per_class, dict) and class_name in per_class:
        general.update(per_class[class_name])
    return {
        "suspicious_step_m": float(general.get("suspicious_step_m", 3.0)),
        "invalid_step_m": float(general.get("invalid_step_m", 6.0)),
        "suspicious_speed_m_per_frame": float(general.get("suspicious_speed_m_per_frame", 3.0)),
        "invalid_speed_m_per_frame": float(general.get("invalid_speed_m_per_frame", 6.0)),
        "suspicious_travel_per_frame": float(general.get("suspicious_travel_per_frame", 1.5)),
        "invalid_travel_per_frame": float(general.get("invalid_travel_per_frame", 3.0)),
        "max_jump_ratio_suspicious": float(general.get("max_jump_ratio_suspicious", 0.10)),
        "max_jump_ratio_invalid": float(general.get("max_jump_ratio_invalid", 0.25)),
    }


def _valid_3d_ratio(candidate: MTMCTrackletCandidate, num_points: int) -> float:
    denom = len(candidate.trajectory_3d_sampled)
    if denom <= 0:
        denom = max(int(candidate.trajectory_3d_length), int(candidate.length), 1)
    return min(float(num_points) / float(max(denom, 1)), 1.0)


def _jump_count(steps: List[Tuple[int, int, float, int]], thresholds: Dict[str, float]) -> int:
    count = 0
    suspicious_step = float(thresholds["suspicious_step_m"])
    suspicious_speed = float(thresholds["suspicious_speed_m_per_frame"])
    for _frame_a, _frame_b, distance, frame_gap in steps:
        speed = float(distance) / float(max(int(frame_gap), 1))
        if float(distance) >= suspicious_step or speed >= suspicious_speed:
            count += 1
    return count


def _straight_line_distance(points: List[Tuple[int, np.ndarray]]) -> Optional[float]:
    if len(points) < 2:
        return None
    return float(np.linalg.norm(points[-1][1] - points[0][1]))


def _path_efficiency(straight: Optional[float], travel: Optional[float]) -> Optional[float]:
    if straight is None or travel is None or travel <= 1e-6:
        return None
    return float(straight) / float(travel)


def _greater_equal(value: Optional[float], threshold: float) -> bool:
    if value is None:
        return False
    return float(value) >= float(threshold)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _percentile(values: List[float], percentile: int) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _max(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))
