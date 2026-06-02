"""Dataclasses for MTMC transition diagnostics."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TransitionCandidatePair:
    """One non-overlap transition candidate pair."""

    scene_name: str
    subset: str
    split: str
    class_id: int
    class_name: str
    candidate_id_a: str
    candidate_id_b: str
    camera_id_a: str
    camera_id_b: str
    camera_pair: str
    start_frame_a: int
    end_frame_a: int
    start_frame_b: int
    end_frame_b: int
    temporal_relation: str
    temporal_gap: int
    entry_exit_distance: Optional[float]
    normalized_entry_exit_distance: Optional[float]
    velocity_angle_difference: Optional[float]
    velocity_magnitude_difference: Optional[float]
    expected_position_error: Optional[float]
    reverse_expected_position_error: Optional[float]
    confidence_pair_mean: float
    gt_id_a: Optional[int]
    gt_id_b: Optional[int]
    same_gt_object_id: Optional[bool]
    diagnostic_label: str
    transition_cost: Optional[float]
    accepted_by_threshold: bool
    reject_reason: str


def transition_pair_to_dict(pair: TransitionCandidatePair) -> Dict[str, Any]:
    """Convert a transition pair to a JSON-friendly dictionary."""
    return {
        "scene_name": pair.scene_name,
        "subset": pair.subset,
        "split": pair.split,
        "class_id": pair.class_id,
        "class_name": pair.class_name,
        "candidate_id_a": pair.candidate_id_a,
        "candidate_id_b": pair.candidate_id_b,
        "camera_id_a": pair.camera_id_a,
        "camera_id_b": pair.camera_id_b,
        "camera_pair": pair.camera_pair,
        "start_frame_a": pair.start_frame_a,
        "end_frame_a": pair.end_frame_a,
        "start_frame_b": pair.start_frame_b,
        "end_frame_b": pair.end_frame_b,
        "temporal_relation": pair.temporal_relation,
        "temporal_gap": pair.temporal_gap,
        "entry_exit_distance": pair.entry_exit_distance,
        "normalized_entry_exit_distance": pair.normalized_entry_exit_distance,
        "velocity_angle_difference": pair.velocity_angle_difference,
        "velocity_magnitude_difference": pair.velocity_magnitude_difference,
        "expected_position_error": pair.expected_position_error,
        "reverse_expected_position_error": pair.reverse_expected_position_error,
        "confidence_pair_mean": pair.confidence_pair_mean,
        "gt_id_a": pair.gt_id_a,
        "gt_id_b": pair.gt_id_b,
        "same_gt_object_id": pair.same_gt_object_id,
        "diagnostic_label": pair.diagnostic_label,
        "transition_cost": pair.transition_cost,
        "accepted_by_threshold": pair.accepted_by_threshold,
        "reject_reason": pair.reject_reason,
    }


def transition_pair_from_dict(data: Dict[str, Any]) -> TransitionCandidatePair:
    """Create a transition pair from a dictionary."""
    return TransitionCandidatePair(
        scene_name=str(data.get("scene_name", "")),
        subset=str(data.get("subset", "")),
        split=str(data.get("split", "")),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        candidate_id_a=str(data.get("candidate_id_a", "")),
        candidate_id_b=str(data.get("candidate_id_b", "")),
        camera_id_a=str(data.get("camera_id_a", "")),
        camera_id_b=str(data.get("camera_id_b", "")),
        camera_pair=str(data.get("camera_pair", "")),
        start_frame_a=int(data.get("start_frame_a", -1)),
        end_frame_a=int(data.get("end_frame_a", -1)),
        start_frame_b=int(data.get("start_frame_b", -1)),
        end_frame_b=int(data.get("end_frame_b", -1)),
        temporal_relation=str(data.get("temporal_relation", "")),
        temporal_gap=int(data.get("temporal_gap", 0)),
        entry_exit_distance=_optional_float(data.get("entry_exit_distance")),
        normalized_entry_exit_distance=_optional_float(data.get("normalized_entry_exit_distance")),
        velocity_angle_difference=_optional_float(data.get("velocity_angle_difference")),
        velocity_magnitude_difference=_optional_float(data.get("velocity_magnitude_difference")),
        expected_position_error=_optional_float(data.get("expected_position_error")),
        reverse_expected_position_error=_optional_float(data.get("reverse_expected_position_error")),
        confidence_pair_mean=float(data.get("confidence_pair_mean", 0.0)),
        gt_id_a=_optional_int(data.get("gt_id_a")),
        gt_id_b=_optional_int(data.get("gt_id_b")),
        same_gt_object_id=_optional_bool(data.get("same_gt_object_id")),
        diagnostic_label=str(data.get("diagnostic_label", "unknown_gt")),
        transition_cost=_optional_float(data.get("transition_cost")),
        accepted_by_threshold=_bool(data.get("accepted_by_threshold", False)),
        reject_reason=str(data.get("reject_reason", "")),
    )


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(float(value))


def _optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    return _bool(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")
