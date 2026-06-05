"""Transition candidate diagnostics for MTMC association."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_association_cost import (
    compute_velocity_angle_difference,
    temporal_gap,
    temporal_overlap,
    temporal_relation,
)
from deep_oc_sort_3d.mtmc.transition_cost import (
    apply_transition_per_class_overrides,
    compute_transition_cost,
    merge_transition_config,
)
from deep_oc_sort_3d.mtmc.transition_types import TransitionCandidatePair


def build_transition_candidate_pairs(
    candidates: List[MTMCTrackletCandidate],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> List[TransitionCandidatePair]:
    """Build non-overlap transition candidate pairs with blocking."""
    cfg = merge_transition_config(config)
    groups = _group_candidates(candidates, cfg)
    pairs = []
    group_items = sorted(groups.items(), key=lambda item: item[0])
    for key, indices in _progress_iter(group_items, show_progress, "transition groups", "group"):
        _unused_key = key
        indices = _limit_group(indices, cfg)
        for index_a, index_b in _candidate_index_pairs(candidates, indices, cfg):
            pair = compute_transition_pair_metrics(candidates[index_a], candidates[index_b], cfg)
            pairs.append(pair)
    return pairs


def compute_transition_pair_metrics(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    config: Dict[str, Any],
) -> TransitionCandidatePair:
    """Compute transition diagnostics for two candidates."""
    cfg = merge_transition_config(config)
    relation = temporal_relation(a, b)
    gap = temporal_gap(a, b)
    entry_exit = _entry_exit_distance(a, b, relation)
    normalized = None if entry_exit is None else float(entry_exit) / max(float(gap), 1.0)
    velocity_angle = compute_velocity_angle_difference(a, b)
    velocity_magnitude_difference = _velocity_magnitude_difference(a, b)
    expected_error = _expected_position_error(a, b, relation, gap)
    reverse_expected_error = _reverse_expected_position_error(a, b, relation, gap)
    gt_id_a = a.majority_gt_object_id
    gt_id_b = b.majority_gt_object_id
    same_gt = None
    if gt_id_a is not None and gt_id_b is not None:
        same_gt = int(gt_id_a) == int(gt_id_b)
    label = label_transition_pair_diagnostic(gt_id_a, gt_id_b)
    camera_pair = _camera_pair(a.camera_id, b.camera_id)
    base_pair = TransitionCandidatePair(
        scene_name=str(a.scene_name),
        subset=str(a.subset),
        split=str(a.split),
        class_id=int(a.class_id),
        class_name=str(a.class_name),
        candidate_id_a=str(a.candidate_id),
        candidate_id_b=str(b.candidate_id),
        camera_id_a=str(a.camera_id),
        camera_id_b=str(b.camera_id),
        camera_pair=camera_pair,
        start_frame_a=int(a.start_frame),
        end_frame_a=int(a.end_frame),
        start_frame_b=int(b.start_frame),
        end_frame_b=int(b.end_frame),
        temporal_relation=relation,
        temporal_gap=int(gap),
        entry_exit_distance=entry_exit,
        normalized_entry_exit_distance=normalized,
        velocity_angle_difference=velocity_angle,
        velocity_magnitude_difference=velocity_magnitude_difference,
        expected_position_error=expected_error,
        reverse_expected_position_error=reverse_expected_error,
        confidence_pair_mean=(float(a.mean_confidence) + float(b.mean_confidence)) * 0.5,
        gt_id_a=gt_id_a,
        gt_id_b=gt_id_b,
        same_gt_object_id=same_gt,
        diagnostic_label=label,
        transition_cost=None,
        accepted_by_threshold=False,
        reject_reason="not_scored",
    )
    cost, accepted, reason = compute_transition_cost(base_pair, cfg)
    base_pair.transition_cost = cost
    base_pair.accepted_by_threshold = accepted
    base_pair.reject_reason = reason
    return base_pair


def label_transition_pair_diagnostic(
    gt_id_a: Optional[int],
    gt_id_b: Optional[int],
) -> str:
    """Return GT-only diagnostic label for a transition pair."""
    if gt_id_a is None or gt_id_b is None:
        return "unknown_gt"
    if int(gt_id_a) == int(gt_id_b):
        return "true_transition"
    return "false_transition"


def summarize_transition_pairs(pairs: List[TransitionCandidatePair]) -> Dict[str, Any]:
    """Summarize transition pair diagnostics."""
    accepted = [pair for pair in pairs if pair.accepted_by_threshold]
    true_pairs = [pair for pair in pairs if pair.diagnostic_label == "true_transition"]
    false_pairs = [pair for pair in pairs if pair.diagnostic_label == "false_transition"]
    unknown_pairs = [pair for pair in pairs if pair.diagnostic_label == "unknown_gt"]
    accepted_true = [pair for pair in accepted if pair.diagnostic_label == "true_transition"]
    accepted_false = [pair for pair in accepted if pair.diagnostic_label == "false_transition"]
    return {
        "total_pairs": len(pairs),
        "true_transition": len(true_pairs),
        "false_transition": len(false_pairs),
        "unknown_gt": len(unknown_pairs),
        "accepted_by_threshold": len(accepted),
        "accepted_true": len(accepted_true),
        "accepted_false": len(accepted_false),
        "precision_diagnostic": _ratio(len(accepted_true), len(accepted_true) + len(accepted_false)),
        "recall_proxy_diagnostic": _ratio(len(accepted_true), len(true_pairs)),
        "per_class_counts": _count_by(pairs, "class_name"),
        "per_class_accepted": _count_by(accepted, "class_name"),
        "per_camera_pair_counts": _count_by(pairs, "camera_pair"),
        "per_camera_pair_accepted": _count_by(accepted, "camera_pair"),
        "reject_reasons": _count_by(pairs, "reject_reason"),
        "temporal_gap_stats": _stats([pair.temporal_gap for pair in pairs]),
        "entry_exit_distance_stats": _stats([pair.entry_exit_distance for pair in pairs if pair.entry_exit_distance is not None]),
        "normalized_distance_stats": _stats(
            [pair.normalized_entry_exit_distance for pair in pairs if pair.normalized_entry_exit_distance is not None]
        ),
        "expected_position_error_stats": _stats(
            [pair.expected_position_error for pair in pairs if pair.expected_position_error is not None]
        ),
        "top_camera_pairs_by_true_transition": _top_camera_pairs(true_pairs),
        "top_camera_pairs_by_false_transition": _top_camera_pairs(false_pairs),
    }


def _group_candidates(candidates: List[MTMCTrackletCandidate], cfg: Dict[str, Any]) -> Dict[Tuple[str, str, int], List[int]]:
    groups = {}
    for index, candidate in enumerate(candidates):
        if not candidate.is_candidate:
            continue
        if bool(cfg["class_must_match"]):
            key = (str(candidate.subset), str(candidate.scene_name), int(candidate.class_id))
        else:
            key = (str(candidate.subset), str(candidate.scene_name), -1)
        groups.setdefault(key, []).append(index)
    return groups


def _limit_group(indices: List[int], cfg: Dict[str, Any]) -> List[int]:
    max_candidates = cfg.get("max_candidates_per_group")
    if max_candidates is None:
        return indices
    return indices[: int(max_candidates)]


def _candidate_index_pairs(
    candidates: List[MTMCTrackletCandidate],
    indices: List[int],
    cfg: Dict[str, Any],
) -> List[Tuple[int, int]]:
    camera_groups = {}
    for index in indices:
        camera_groups.setdefault(candidates[index].camera_id, []).append(index)
    camera_ids = sorted(camera_groups.keys())
    pairs = []
    for outer_pos, camera_a in enumerate(camera_ids):
        for camera_b in camera_ids[outer_pos + 1 :]:
            if not bool(cfg["allow_same_camera_links"]) and camera_a == camera_b:
                continue
            pairs.extend(_camera_pair_indices(candidates, camera_groups[camera_a], camera_groups[camera_b], cfg))
    return pairs


def _camera_pair_indices(
    candidates: List[MTMCTrackletCandidate],
    indices_a: List[int],
    indices_b: List[int],
    cfg: Dict[str, Any],
) -> List[Tuple[int, int]]:
    output = []
    sorted_a = sorted(indices_a, key=lambda index: candidates[index].start_frame)
    sorted_b = sorted(indices_b, key=lambda index: candidates[index].start_frame)
    for index_a in sorted_a:
        candidate_a = candidates[index_a]
        cfg_a = apply_transition_per_class_overrides(cfg, candidate_a.class_id, candidate_a.class_name)
        for index_b in sorted_b:
            candidate_b = candidates[index_b]
            cfg_b = apply_transition_per_class_overrides(cfg, candidate_b.class_id, candidate_b.class_name)
            min_gap = min(int(cfg_a["min_temporal_gap"]), int(cfg_b["min_temporal_gap"]))
            max_gap = max(int(cfg_a["max_temporal_gap"]), int(cfg_b["max_temporal_gap"]))
            if temporal_overlap(candidate_a, candidate_b) > 0:
                continue
            gap = temporal_gap(candidate_a, candidate_b)
            if gap < min_gap:
                continue
            if gap > max_gap:
                continue
            if candidate_a.entry_center_3d is None or candidate_a.exit_center_3d is None:
                continue
            if candidate_b.entry_center_3d is None or candidate_b.exit_center_3d is None:
                continue
            output.append((index_a, index_b))
    return output


def _entry_exit_distance(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate, relation: str) -> Optional[float]:
    if relation == "a_before_b":
        return _distance(a.exit_center_3d, b.entry_center_3d)
    if relation == "b_before_a":
        return _distance(b.exit_center_3d, a.entry_center_3d)
    return None


def _expected_position_error(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    relation: str,
    gap: int,
) -> Optional[float]:
    if relation == "a_before_b":
        return _forward_prediction_error(a.exit_center_3d, a.mean_velocity_3d, b.entry_center_3d, gap)
    if relation == "b_before_a":
        return _forward_prediction_error(b.exit_center_3d, b.mean_velocity_3d, a.entry_center_3d, gap)
    return None


def _reverse_expected_position_error(
    a: MTMCTrackletCandidate,
    b: MTMCTrackletCandidate,
    relation: str,
    gap: int,
) -> Optional[float]:
    if relation == "a_before_b":
        return _backward_prediction_error(b.entry_center_3d, b.mean_velocity_3d, a.exit_center_3d, gap)
    if relation == "b_before_a":
        return _backward_prediction_error(a.entry_center_3d, a.mean_velocity_3d, b.exit_center_3d, gap)
    return None


def _forward_prediction_error(start: Any, velocity: Any, target: Any, gap: int) -> Optional[float]:
    if start is None or velocity is None or target is None:
        return None
    start_arr = np.asarray(start, dtype=float).reshape(-1)[:3]
    velocity_arr = np.asarray(velocity, dtype=float).reshape(-1)[:3]
    if start_arr.size < 3 or velocity_arr.size < 3:
        return None
    predicted = start_arr + velocity_arr * float(gap)
    return _distance(predicted, target)


def _backward_prediction_error(start: Any, velocity: Any, target: Any, gap: int) -> Optional[float]:
    if start is None or velocity is None or target is None:
        return None
    start_arr = np.asarray(start, dtype=float).reshape(-1)[:3]
    velocity_arr = np.asarray(velocity, dtype=float).reshape(-1)[:3]
    if start_arr.size < 3 or velocity_arr.size < 3:
        return None
    predicted = start_arr - velocity_arr * float(gap)
    return _distance(predicted, target)


def _velocity_magnitude_difference(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate) -> Optional[float]:
    if a.mean_velocity_3d is None or b.mean_velocity_3d is None:
        return None
    va = np.asarray(a.mean_velocity_3d, dtype=float).reshape(-1)[:3]
    vb = np.asarray(b.mean_velocity_3d, dtype=float).reshape(-1)[:3]
    if va.size < 3 or vb.size < 3:
        return None
    return abs(float(np.linalg.norm(va)) - float(np.linalg.norm(vb)))


def _distance(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    left_arr = np.asarray(left, dtype=float).reshape(-1)[:3]
    right_arr = np.asarray(right, dtype=float).reshape(-1)[:3]
    if left_arr.size < 3 or right_arr.size < 3:
        return None
    return float(np.linalg.norm(left_arr - right_arr))


def _camera_pair(camera_a: str, camera_b: str) -> str:
    values = sorted([str(camera_a), str(camera_b)])
    return "%s__%s" % (values[0], values[1])


def _count_by(pairs: List[TransitionCandidatePair], field: str) -> Dict[str, int]:
    counts = {}
    for pair in pairs:
        key = str(getattr(pair, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_camera_pairs(pairs: List[TransitionCandidatePair]) -> List[Dict[str, Any]]:
    counts = _count_by(pairs, "camera_pair")
    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [{"camera_pair": key, "count": value} for key, value in items[:10]]


def _stats(values: List[Any]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p75": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    total = len(values) if hasattr(values, "__len__") else None
    for index, value in enumerate(values):
        if total is None:
            print("%s: item %d" % (desc, index + 1))
        elif index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print("%s: item %d/%d" % (desc, index + 1, total))
        yield value
