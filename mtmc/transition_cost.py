"""Cost functions for transition association."""

from typing import Any, Dict, Optional, Tuple

from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge
from deep_oc_sort_3d.mtmc.transition_types import TransitionCandidatePair


INF_COST = 1e9


def default_transition_config() -> Dict[str, Any]:
    """Return default transition diagnostic config."""
    return {
        "class_must_match": True,
        "allow_same_camera_links": False,
        "min_temporal_gap": 1,
        "max_temporal_gap": 120,
        "max_entry_exit_distance": 5.0,
        "max_normalized_distance": 0.5,
        "max_velocity_angle_deg": 120.0,
        "max_expected_position_error": 6.0,
        "entry_exit_distance_weight": 1.0,
        "normalized_distance_weight": 0.5,
        "temporal_gap_weight": 0.01,
        "velocity_angle_weight": 0.1,
        "expected_error_weight": 0.5,
        "confidence_weight": 0.05,
        "transition_cost_threshold": 1.0,
        "max_candidates_per_group": None,
    }


def merge_transition_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge partial transition config over defaults."""
    merged = default_transition_config()
    if config:
        merged.update(config)
    return merged


def compute_transition_cost(
    pair: TransitionCandidatePair,
    config: Dict[str, Any],
) -> Tuple[float, bool, str]:
    """Compute transition cost and threshold decision."""
    cfg = merge_transition_config(config)
    reject = _hard_reject_reason(pair, cfg)
    if reject != "ok":
        return INF_COST, False, reject

    entry_norm = float(pair.entry_exit_distance) / max(float(cfg["max_entry_exit_distance"]), 1e-6)
    normalized_distance = 0.0
    if pair.normalized_entry_exit_distance is not None:
        normalized_distance = float(pair.normalized_entry_exit_distance) / max(float(cfg["max_normalized_distance"]), 1e-6)
    temporal_norm = float(pair.temporal_gap) / max(float(cfg["max_temporal_gap"]), 1e-6)
    velocity_norm = 0.0
    if pair.velocity_angle_difference is not None:
        velocity_norm = float(pair.velocity_angle_difference) / max(float(cfg["max_velocity_angle_deg"]), 1e-6)
    expected_norm = 0.0
    if pair.expected_position_error is not None:
        expected_norm = float(pair.expected_position_error) / max(float(cfg["max_expected_position_error"]), 1e-6)

    cost = 0.0
    cost += float(cfg["entry_exit_distance_weight"]) * entry_norm
    cost += float(cfg["normalized_distance_weight"]) * normalized_distance
    cost += float(cfg["temporal_gap_weight"]) * temporal_norm
    cost += float(cfg["velocity_angle_weight"]) * velocity_norm
    cost += float(cfg["expected_error_weight"]) * expected_norm
    cost -= float(cfg["confidence_weight"]) * float(pair.confidence_pair_mean)
    cost = max(0.0, float(cost))
    accepted = cost <= float(cfg["transition_cost_threshold"])
    reason = "ok" if accepted else "transition_cost_above_threshold"
    return cost, accepted, reason


def transition_pair_to_global_edge(
    pair: TransitionCandidatePair,
    cost: float,
    accepted: bool,
    reject_reason: str,
) -> GlobalAssociationEdge:
    """Convert a transition pair into a 12A-compatible global edge."""
    affinity = 0.0 if cost >= INF_COST else 1.0 / (1.0 + float(cost))
    return GlobalAssociationEdge(
        scene_name=pair.scene_name,
        subset=pair.subset,
        class_id=pair.class_id,
        class_name=pair.class_name,
        candidate_id_a=pair.candidate_id_a,
        candidate_id_b=pair.candidate_id_b,
        camera_id_a=pair.camera_id_a,
        camera_id_b=pair.camera_id_b,
        start_frame_a=pair.start_frame_a,
        end_frame_a=pair.end_frame_a,
        start_frame_b=pair.start_frame_b,
        end_frame_b=pair.end_frame_b,
        temporal_relation=pair.temporal_relation,
        overlap_frames=0,
        temporal_gap=pair.temporal_gap,
        mean_3d_distance=pair.entry_exit_distance,
        median_3d_distance=pair.entry_exit_distance,
        min_3d_distance=pair.entry_exit_distance,
        max_3d_distance=pair.entry_exit_distance,
        entry_exit_distance=pair.entry_exit_distance,
        velocity_angle_difference=pair.velocity_angle_difference,
        cost=float(cost),
        affinity=float(affinity),
        accepted=bool(accepted),
        reject_reason=str(reject_reason),
    )


def _hard_reject_reason(pair: TransitionCandidatePair, cfg: Dict[str, Any]) -> str:
    if pair.temporal_gap < int(cfg["min_temporal_gap"]):
        return "temporal_gap_too_small"
    if pair.temporal_gap > int(cfg["max_temporal_gap"]):
        return "temporal_gap_too_large"
    if pair.entry_exit_distance is None:
        return "missing_entry_exit_distance"
    if float(pair.entry_exit_distance) > float(cfg["max_entry_exit_distance"]):
        return "entry_exit_distance_too_large"
    if pair.normalized_entry_exit_distance is None:
        return "missing_normalized_distance"
    if float(pair.normalized_entry_exit_distance) > float(cfg["max_normalized_distance"]):
        return "normalized_distance_too_large"
    if pair.velocity_angle_difference is not None:
        if float(pair.velocity_angle_difference) > float(cfg["max_velocity_angle_deg"]):
            return "velocity_angle_too_large"
    if pair.expected_position_error is not None:
        if float(pair.expected_position_error) > float(cfg["max_expected_position_error"]):
            return "expected_position_error_too_large"
    return "ok"
