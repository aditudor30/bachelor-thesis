"""Pairwise cost functions for global MTMC association."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge


INF_COST = 1e9


def default_global_association_config() -> Dict[str, Any]:
    """Return default global association config."""
    return {
        "class_must_match": True,
        "allow_same_camera_links": False,
        "enable_overlap_association": True,
        "enable_transition_association": False,
        "max_frame_delta_for_overlap": 2,
        "max_mean_overlap_distance": 2.5,
        "max_median_overlap_distance": 2.0,
        "max_entry_exit_distance": 3.0,
        "max_temporal_gap": 60,
        "max_velocity_angle_deg": 90.0,
        "overlap_distance_weight": 1.0,
        "transition_distance_weight": 1.0,
        "temporal_gap_weight": 0.1,
        "velocity_weight": 0.1,
        "confidence_weight": 0.05,
        "cost_threshold": 1.0,
        "include_singletons": True,
        "min_candidates_per_global_track": 1,
        "max_candidates_per_group": None,
    }


def merge_global_association_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge partial config over defaults."""
    merged = default_global_association_config()
    if config:
        merged.update(config)
    return merged


def temporal_overlap(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate) -> int:
    """Return number of overlapping frames."""
    start = max(int(a.start_frame), int(b.start_frame))
    end = min(int(a.end_frame), int(b.end_frame))
    if end < start:
        return 0
    return int(end - start + 1)


def temporal_gap(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate) -> int:
    """Return positive temporal gap or zero when candidates overlap."""
    if temporal_overlap(a, b) > 0:
        return 0
    if int(a.end_frame) < int(b.start_frame):
        return int(b.start_frame) - int(a.end_frame)
    return int(a.start_frame) - int(b.end_frame)


def temporal_relation(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate) -> str:
    """Return temporal relation between two candidates."""
    if temporal_overlap(a, b) > 0:
        return "overlap"
    if int(a.end_frame) < int(b.start_frame):
        return "a_before_b"
    return "b_before_a"


def get_common_or_nearby_3d_points(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    max_frame_delta: int = 2,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Return nearby 3D points for overlap-mode distance checks."""
    points_a = _trajectory_dict(a)
    points_b = _trajectory_dict(b)
    output = []
    for frame_a, center_a in points_a.items():
        best_frame = None
        best_delta = None
        for frame_b in points_b.keys():
            delta = abs(int(frame_a) - int(frame_b))
            if delta <= int(max_frame_delta) and (best_delta is None or delta < best_delta):
                best_delta = delta
                best_frame = frame_b
        if best_frame is not None:
            output.append((int(frame_a), center_a, points_b[best_frame]))
    return output


def compute_overlap_3d_distance_stats(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    max_frame_delta: int = 2,
) -> Dict[str, Any]:
    """Compute 3D distance stats over common/nearby trajectory points."""
    points = get_common_or_nearby_3d_points(a, b, max_frame_delta=max_frame_delta)
    distances = [float(np.linalg.norm(center_a - center_b)) for _frame, center_a, center_b in points]
    return {
        "count": len(distances),
        "mean": _mean(distances),
        "median": _median(distances),
        "min": _min(distances),
        "max": _max(distances),
    }


def compute_entry_exit_distance(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate) -> Optional[float]:
    """Compute transition entry/exit distance."""
    relation = temporal_relation(a, b)
    if relation == "overlap":
        return None
    if relation == "a_before_b":
        if a.exit_center_3d is None or b.entry_center_3d is None:
            return None
        return float(np.linalg.norm(np.asarray(a.exit_center_3d, dtype=float) - np.asarray(b.entry_center_3d, dtype=float)))
    if b.exit_center_3d is None or a.entry_center_3d is None:
        return None
    return float(np.linalg.norm(np.asarray(b.exit_center_3d, dtype=float) - np.asarray(a.entry_center_3d, dtype=float)))


def compute_velocity_angle_difference(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate) -> Optional[float]:
    """Return angle in degrees between mean velocity vectors."""
    if a.mean_velocity_3d is None or b.mean_velocity_3d is None:
        return None
    va = np.asarray(a.mean_velocity_3d, dtype=float).reshape(-1)
    vb = np.asarray(b.mean_velocity_3d, dtype=float).reshape(-1)
    if va.size < 3 or vb.size < 3:
        return None
    norm_a = float(np.linalg.norm(va[:3]))
    norm_b = float(np.linalg.norm(vb[:3]))
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return None
    cos_value = float(np.dot(va[:3], vb[:3]) / (norm_a * norm_b))
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(np.degrees(np.arccos(cos_value)))


def compute_global_association_cost(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    config: Dict[str, Any],
) -> GlobalAssociationEdge:
    """Compute global association edge for two candidates."""
    cfg = merge_global_association_config(config)
    relation = temporal_relation(a, b)
    overlap = temporal_overlap(a, b)
    gap = temporal_gap(a, b)
    stats = compute_overlap_3d_distance_stats(a, b, int(cfg["max_frame_delta_for_overlap"]))
    entry_exit = compute_entry_exit_distance(a, b)
    velocity_angle = compute_velocity_angle_difference(a, b)
    reject = _hard_reject_reason(a, b, cfg, relation, overlap, gap, stats, entry_exit, velocity_angle)
    if reject != "ok":
        return _edge(a, b, relation, overlap, gap, stats, entry_exit, velocity_angle, INF_COST, False, reject)

    if relation == "overlap":
        base_distance = float(stats["mean"])
        norm_distance = base_distance / max(float(cfg["max_mean_overlap_distance"]), 1e-6)
        cost = float(cfg["overlap_distance_weight"]) * norm_distance
    else:
        base_distance = 0.0 if entry_exit is None else float(entry_exit)
        norm_distance = base_distance / max(float(cfg["max_entry_exit_distance"]), 1e-6)
        cost = float(cfg["transition_distance_weight"]) * norm_distance
        cost += float(cfg["temporal_gap_weight"]) * (float(gap) / max(float(cfg["max_temporal_gap"]), 1e-6))

    if velocity_angle is not None:
        cost += float(cfg["velocity_weight"]) * (float(velocity_angle) / max(float(cfg["max_velocity_angle_deg"]), 1e-6))
    confidence = (float(a.mean_confidence) + float(b.mean_confidence)) * 0.5
    cost -= float(cfg["confidence_weight"]) * confidence
    cost = max(0.0, float(cost))
    accepted = cost <= float(cfg["cost_threshold"])
    reason = "ok" if accepted else "cost_above_threshold"
    return _edge(a, b, relation, overlap, gap, stats, entry_exit, velocity_angle, cost, accepted, reason)


def _hard_reject_reason(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    cfg: Dict[str, Any],
    relation: str,
    overlap: int,
    gap: int,
    stats: Dict[str, Any],
    entry_exit: Optional[float],
    velocity_angle: Optional[float],
) -> str:
    if a.scene_name != b.scene_name:
        return "scene_mismatch"
    if bool(cfg["class_must_match"]) and int(a.class_id) != int(b.class_id):
        return "class_mismatch"
    if not bool(cfg["allow_same_camera_links"]) and a.camera_id == b.camera_id:
        return "same_camera_not_allowed"
    if relation == "overlap":
        if not bool(cfg["enable_overlap_association"]):
            return "overlap_disabled"
        if overlap <= 0 or int(stats["count"]) <= 0:
            return "no_3d_overlap"
        if stats["mean"] is None or float(stats["mean"]) > float(cfg["max_mean_overlap_distance"]):
            return "overlap_distance_too_large"
        if stats["median"] is None or float(stats["median"]) > float(cfg["max_median_overlap_distance"]):
            return "overlap_distance_too_large"
    else:
        if not bool(cfg["enable_transition_association"]):
            return "transition_disabled"
        if int(gap) > int(cfg["max_temporal_gap"]):
            return "temporal_gap_too_large"
        if entry_exit is None or float(entry_exit) > float(cfg["max_entry_exit_distance"]):
            return "entry_exit_distance_too_large"
    if velocity_angle is not None and float(velocity_angle) > float(cfg["max_velocity_angle_deg"]):
        return "velocity_incompatible"
    return "ok"


def _edge(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    relation: str,
    overlap: int,
    gap: int,
    stats: Dict[str, Any],
    entry_exit: Optional[float],
    velocity_angle: Optional[float],
    cost: float,
    accepted: bool,
    reason: str,
) -> GlobalAssociationEdge:
    affinity = 0.0 if cost >= INF_COST else 1.0 / (1.0 + float(cost))
    return GlobalAssociationEdge(
        scene_name=str(a.scene_name),
        subset=str(a.subset),
        class_id=int(a.class_id),
        class_name=str(a.class_name),
        candidate_id_a=str(a.candidate_id),
        candidate_id_b=str(b.candidate_id),
        camera_id_a=str(a.camera_id),
        camera_id_b=str(b.camera_id),
        start_frame_a=int(a.start_frame),
        end_frame_a=int(a.end_frame),
        start_frame_b=int(b.start_frame),
        end_frame_b=int(b.end_frame),
        temporal_relation=relation,
        overlap_frames=int(overlap),
        temporal_gap=int(gap),
        mean_3d_distance=stats.get("mean"),
        median_3d_distance=stats.get("median"),
        min_3d_distance=stats.get("min"),
        max_3d_distance=stats.get("max"),
        entry_exit_distance=entry_exit,
        velocity_angle_difference=velocity_angle,
        cost=float(cost),
        affinity=float(affinity),
        accepted=bool(accepted),
        reject_reason=str(reason),
    )


def _trajectory_dict(candidate: MTMCTrackletCandidate) -> Dict[int, np.ndarray]:
    output = {}
    for item in candidate.trajectory_3d_sampled:
        if len(item) < 4:
            continue
        output[int(item[0])] = np.asarray([float(item[1]), float(item[2]), float(item[3])], dtype=float)
    return output


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _min(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(min(values))


def _max(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))
