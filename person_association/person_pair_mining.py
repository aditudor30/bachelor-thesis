"""Mine Person-only fragment pairs for conservative association experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import math

from deep_oc_sort_3d.person_association.person_association_io import (
    TrackKey,
    frame_record_csv_files,
    generic_csv_files,
    optional_list,
    progress_iter,
    read_csv_rows,
    row_track_key,
    safe_float,
    safe_int,
    serialize_track_key,
    write_csv_rows,
    write_json,
)


@dataclass
class PersonTrackFragment:
    """A Person global-track fragment summarized from final frame records."""

    key: TrackKey
    subset: str
    scene_name: str
    class_id: int
    class_name: str
    global_track_id: str
    row_count: int
    cameras: List[str]
    frames: Set[int]
    start_frame: Optional[int]
    end_frame: Optional[int]
    entry_center_3d: Optional[Tuple[float, float, float]]
    exit_center_3d: Optional[Tuple[float, float, float]]
    velocity_3d: Optional[Tuple[float, float, float]]
    mean_confidence: float
    max_confidence: float
    matched_gt_counts: Dict[str, int]


def load_person_fragments_from_final_export(
    final_export_root: Path,
    config: Dict[str, Any],
    show_progress: bool = True,
) -> List[PersonTrackFragment]:
    """Load Person fragments from frame records, falling back to generic rows."""
    class_id = int(config.get("class_id", 0))
    subsets = optional_list(config.get("subsets"))
    scenes = optional_list(config.get("scenes"))
    frame_root = final_export_root / "frame_global_records"
    if frame_root.exists():
        rows = _load_rows_from_frame_records(frame_root, subsets, scenes, show_progress)
    else:
        rows = _load_rows_from_generic(final_export_root / "generic_tracking_export", subsets, scenes, show_progress)
    groups: Dict[TrackKey, List[Dict[str, Any]]] = {}
    for row in rows:
        if safe_int(row.get("class_id"), -1) != class_id:
            continue
        key = row_track_key(row)
        if key[3] in ("", "None", "nan"):
            continue
        groups.setdefault(key, []).append(row)
    fragments = []
    for key, group_rows in groups.items():
        fragment = fragment_from_rows(key, group_rows)
        if fragment is not None:
            fragments.append(fragment)
    return sorted(fragments, key=lambda item: (item.subset, item.scene_name, item.start_frame or -1, item.global_track_id))


def fragment_from_rows(key: TrackKey, rows: List[Dict[str, Any]]) -> Optional[PersonTrackFragment]:
    """Create one fragment summary from rows."""
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: safe_int(row.get("frame_id"), 0) or 0)
    frames = sorted([frame for frame in [safe_int(row.get("frame_id"), None) for row in ordered] if frame is not None])
    frame_set = set(frames)
    cameras = sorted(set([str(row.get("camera_id", "")) for row in ordered if str(row.get("camera_id", ""))]))
    confidences = [safe_float(row.get("confidence"), 0.0) or 0.0 for row in ordered]
    entry = _center_3d(ordered[0])
    exit_center = _center_3d(ordered[-1])
    velocity = _velocity(entry, exit_center, frames[0] if frames else None, frames[-1] if frames else None)
    gt_counts = {}
    for row in ordered:
        gt_id = row.get("matched_gt_object_id")
        if gt_id in (None, ""):
            continue
        key_gt = str(gt_id)
        gt_counts[key_gt] = gt_counts.get(key_gt, 0) + 1
    return PersonTrackFragment(
        key=key,
        subset=str(key[0]),
        scene_name=str(key[1]),
        class_id=safe_int(ordered[0].get("class_id"), 0) or 0,
        class_name=str(ordered[0].get("class_name", "Person")),
        global_track_id=str(key[3]),
        row_count=len(ordered),
        cameras=cameras,
        frames=frame_set,
        start_frame=frames[0] if frames else None,
        end_frame=frames[-1] if frames else None,
        entry_center_3d=entry,
        exit_center_3d=exit_center,
        velocity_3d=velocity,
        mean_confidence=float(sum(confidences)) / float(len(confidences)) if confidences else 0.0,
        max_confidence=max(confidences) if confidences else 0.0,
        matched_gt_counts=gt_counts,
    )


def mine_person_candidate_pairs(
    fragments: List[PersonTrackFragment],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Mine Person fragment pairs.

    This compatibility wrapper returns only rows. New callers should use
    mine_person_candidate_pairs_with_summary so rejected-pair counts can be
    summarized without materializing every rejected pair in memory.
    """
    rows, _summary = mine_person_candidate_pairs_with_summary(fragments, config, show_progress=show_progress)
    return rows


def mine_person_candidate_pairs_with_summary(
    fragments: List[PersonTrackFragment],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Mine Person fragment pairs with memory-safe rejected-pair accounting."""
    groups: Dict[Tuple[str, str], List[PersonTrackFragment]] = {}
    for fragment in fragments:
        groups.setdefault((fragment.subset, fragment.scene_name), []).append(fragment)
    rows = []
    store_rejected = bool(config.get("store_rejected_pairs", config.get("include_rejected", False)))
    max_temporal_gap = int(config.get("max_temporal_gap", 45))
    total_pairs = 0
    kept_pairs = 0
    rejected_pairs = 0
    reject_reasons: Dict[str, int] = {}
    gt_diagnostic: Dict[str, int] = {}
    for group_key in progress_iter(sorted(groups.keys()), show_progress, "mine Person scenes", "scene"):
        scene_fragments = sorted(groups[group_key], key=lambda item: (item.start_frame or -1, item.global_track_id))
        for left_index, left in enumerate(scene_fragments):
            for right in scene_fragments[left_index + 1 :]:
                if _can_break_on_temporal_gap(left, right, max_temporal_gap):
                    break
                row = pair_diagnostics(left, right, config)
                total_pairs += 1
                reason = str(row.get("reject_reason", "unknown"))
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                label = str(row.get("same_gt_diagnostic", "unknown"))
                gt_diagnostic[label] = gt_diagnostic.get(label, 0) + 1
                if row.get("candidate_status") == "ok":
                    kept_pairs += 1
                    rows.append(row)
                else:
                    rejected_pairs += 1
                    if store_rejected:
                        rows.append(row)
    summary = {
        "total_pairs": total_pairs,
        "kept_pairs": kept_pairs,
        "rejected_pairs": rejected_pairs,
        "stored_rows": len(rows),
        "store_rejected_pairs": store_rejected,
        "reject_reasons": reject_reasons,
        "gt_diagnostic": gt_diagnostic,
    }
    return rows, summary


def pair_diagnostics(
    left: PersonTrackFragment,
    right: PersonTrackFragment,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute diagnostics and basic gates for a fragment pair."""
    ordered_left, ordered_right = _temporal_order(left, right)
    temporal_gap = _temporal_gap(ordered_left, ordered_right)
    overlap = _temporal_overlap(left, right)
    entry_exit_distance = _entry_exit_distance(ordered_left, ordered_right)
    expected_error = _expected_position_error(ordered_left, ordered_right)
    angle = _velocity_angle(ordered_left.velocity_3d, ordered_right.velocity_3d)
    min_conf = min(left.mean_confidence, right.mean_confidence)
    same_gt = _same_gt_label(left, right)
    row = {
        "subset": left.subset,
        "scene_name": left.scene_name,
        "class_id": left.class_id,
        "class_name": left.class_name,
        "track_a": serialize_track_key(left.key),
        "track_b": serialize_track_key(right.key),
        "global_track_id_a": left.global_track_id,
        "global_track_id_b": right.global_track_id,
        "cameras_a": ";".join(left.cameras),
        "cameras_b": ";".join(right.cameras),
        "start_a": left.start_frame,
        "end_a": left.end_frame,
        "start_b": right.start_frame,
        "end_b": right.end_frame,
        "rows_a": left.row_count,
        "rows_b": right.row_count,
        "mean_confidence_a": left.mean_confidence,
        "mean_confidence_b": right.mean_confidence,
        "max_confidence_a": left.max_confidence,
        "max_confidence_b": right.max_confidence,
        "temporal_gap": temporal_gap,
        "temporal_overlap": overlap,
        "entry_exit_distance_3d": entry_exit_distance,
        "expected_position_error": expected_error,
        "velocity_angle": angle,
        "min_mean_confidence": min_conf,
        "same_gt_diagnostic": same_gt,
    }
    reason = _reject_reason(row, left, right, config)
    row["candidate_status"] = "ok" if reason == "ok" else "rejected"
    row["reject_reason"] = reason
    return row


def summarize_candidate_pairs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize mined candidate pairs."""
    reject_reasons = {}
    same_gt = {}
    for row in rows:
        reason = str(row.get("reject_reason", "unknown"))
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        label = str(row.get("same_gt_diagnostic", "unknown"))
        same_gt[label] = same_gt.get(label, 0) + 1
    kept = [row for row in rows if row.get("candidate_status") == "ok"]
    return {
        "total_pairs": len(rows),
        "kept_pairs": len(kept),
        "rejected_pairs": len(rows) - len(kept),
        "reject_reasons": reject_reasons,
        "gt_diagnostic": same_gt,
    }


def write_candidate_pairs(
    rows: List[Dict[str, Any]],
    output_csv: Path,
    summary_json: Optional[Path] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Write candidate pairs and optional summary."""
    fields = [
        "subset",
        "scene_name",
        "class_id",
        "class_name",
        "track_a",
        "track_b",
        "global_track_id_a",
        "global_track_id_b",
        "cameras_a",
        "cameras_b",
        "start_a",
        "end_a",
        "start_b",
        "end_b",
        "rows_a",
        "rows_b",
        "mean_confidence_a",
        "mean_confidence_b",
        "max_confidence_a",
        "max_confidence_b",
        "temporal_gap",
        "temporal_overlap",
        "entry_exit_distance_3d",
        "expected_position_error",
        "velocity_angle",
        "min_mean_confidence",
        "same_gt_diagnostic",
        "candidate_status",
        "reject_reason",
    ]
    write_csv_rows(rows, output_csv, fields)
    if summary_json is not None:
        write_json(summary if summary is not None else summarize_candidate_pairs(rows), summary_json)


def _can_break_on_temporal_gap(left: PersonTrackFragment, right: PersonTrackFragment, max_temporal_gap: int) -> bool:
    """Return True when later fragments will only be farther in time."""
    if left.end_frame is None or right.start_frame is None:
        return False
    return int(right.start_frame) - int(left.end_frame) > int(max_temporal_gap)


def _load_rows_from_frame_records(root: Path, subsets: Optional[List[str]], scenes: Optional[List[str]], show_progress: bool) -> List[Dict[str, Any]]:
    rows = []
    for path in progress_iter(frame_record_csv_files(root, subsets=subsets, scenes=scenes), show_progress, "load frame records", "file"):
        file_rows, _fields = read_csv_rows(path)
        rows.extend(file_rows)
    return rows


def _load_rows_from_generic(root: Path, subsets: Optional[List[str]], scenes: Optional[List[str]], show_progress: bool) -> List[Dict[str, Any]]:
    rows = []
    for path in progress_iter(generic_csv_files(root, subsets=subsets, scenes=scenes), show_progress, "load generic exports", "file"):
        subset = path.parent.name
        file_rows, _fields = read_csv_rows(path)
        for row in file_rows:
            copied = dict(row)
            copied["subset"] = subset
            rows.append(copied)
    return rows


def _reject_reason(row: Dict[str, Any], left: PersonTrackFragment, right: PersonTrackFragment, config: Dict[str, Any]) -> str:
    allow_same_camera = bool(config.get("allow_same_camera", False))
    if not allow_same_camera and set(left.cameras).intersection(set(right.cameras)):
        return "same_camera_not_allowed"
    if bool(config.get("require_no_temporal_conflict", True)) and int(row.get("temporal_overlap") or 0) > 0:
        return "temporal_overlap"
    gap = safe_int(row.get("temporal_gap"), None)
    if gap is None or gap > int(config.get("max_temporal_gap", 45)):
        return "temporal_gap_too_large"
    distance = safe_float(row.get("entry_exit_distance_3d"), None)
    if distance is None or distance > float(config.get("max_entry_exit_distance", 2.0)):
        return "entry_exit_distance_too_large"
    expected = safe_float(row.get("expected_position_error"), None)
    max_expected = config.get("max_expected_position_error")
    if max_expected is not None and expected is not None and expected > float(max_expected):
        return "expected_position_error_too_large"
    angle = safe_float(row.get("velocity_angle"), None)
    max_angle = config.get("max_velocity_angle")
    if max_angle is not None and angle is not None and angle > float(max_angle):
        return "velocity_angle_too_large"
    min_conf = safe_float(row.get("min_mean_confidence"), 0.0) or 0.0
    if min_conf < float(config.get("min_mean_confidence", 0.0)):
        return "confidence_too_low"
    return "ok"


def _center_3d(row: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    values = [safe_float(row.get("center_x"), None), safe_float(row.get("center_y"), None), safe_float(row.get("center_z"), None)]
    if any(value is None for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _velocity(
    entry: Optional[Tuple[float, float, float]],
    exit_center: Optional[Tuple[float, float, float]],
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> Optional[Tuple[float, float, float]]:
    if entry is None or exit_center is None or start_frame is None or end_frame is None:
        return None
    dt = max(1.0, float(end_frame - start_frame))
    return (
        (exit_center[0] - entry[0]) / dt,
        (exit_center[1] - entry[1]) / dt,
        (exit_center[2] - entry[2]) / dt,
    )


def _temporal_order(left: PersonTrackFragment, right: PersonTrackFragment) -> Tuple[PersonTrackFragment, PersonTrackFragment]:
    if left.start_frame is None or right.start_frame is None:
        return left, right
    if left.start_frame <= right.start_frame:
        return left, right
    return right, left


def _temporal_gap(left: PersonTrackFragment, right: PersonTrackFragment) -> Optional[int]:
    if left.end_frame is None or right.start_frame is None:
        return None
    if left.end_frame < right.start_frame:
        return int(right.start_frame) - int(left.end_frame)
    return 0


def _temporal_overlap(left: PersonTrackFragment, right: PersonTrackFragment) -> int:
    return len(left.frames.intersection(right.frames))


def _entry_exit_distance(left: PersonTrackFragment, right: PersonTrackFragment) -> Optional[float]:
    return _distance(left.exit_center_3d, right.entry_center_3d)


def _expected_position_error(left: PersonTrackFragment, right: PersonTrackFragment) -> Optional[float]:
    if left.exit_center_3d is None or left.velocity_3d is None or right.entry_center_3d is None:
        return None
    gap = _temporal_gap(left, right)
    if gap is None:
        return None
    predicted = (
        left.exit_center_3d[0] + left.velocity_3d[0] * float(gap),
        left.exit_center_3d[1] + left.velocity_3d[1] * float(gap),
        left.exit_center_3d[2] + left.velocity_3d[2] * float(gap),
    )
    return _distance(predicted, right.entry_center_3d)


def _distance(left: Optional[Tuple[float, float, float]], right: Optional[Tuple[float, float, float]]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(sum([(left[index] - right[index]) ** 2 for index in range(3)]) ** 0.5)


def _velocity_angle(left: Optional[Tuple[float, float, float]], right: Optional[Tuple[float, float, float]]) -> Optional[float]:
    if left is None or right is None:
        return None
    norm_left = _norm(left)
    norm_right = _norm(right)
    if norm_left <= 1e-9 or norm_right <= 1e-9:
        return None
    dot = sum([left[index] * right[index] for index in range(3)]) / (norm_left * norm_right)
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def _norm(value: Tuple[float, float, float]) -> float:
    return float(sum([item * item for item in value]) ** 0.5)


def _same_gt_label(left: PersonTrackFragment, right: PersonTrackFragment) -> str:
    if not left.matched_gt_counts or not right.matched_gt_counts:
        return "unknown_gt"
    left_ids = set(left.matched_gt_counts.keys())
    right_ids = set(right.matched_gt_counts.keys())
    if left_ids.intersection(right_ids):
        return "true_match"
    return "false_match"
