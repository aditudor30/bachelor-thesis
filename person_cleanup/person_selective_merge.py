"""Conservative Person-only merge diagnostics and optional ID remapping."""

from typing import Any, Dict, List, Set, Tuple

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import safe_float, safe_int, track_key


TrackKey = Tuple[str, str, str, str]


def build_safe_person_merge_mapping(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[Dict[TrackKey, str], List[Dict[str, Any]]]:
    """Build optional old-track-key to new-id mapping for safe Person merges."""
    if not bool(config.get("enabled", False)):
        return {}, []
    stats = _track_summaries(rows, int(config.get("class_id", 0)))
    candidates = []
    keys = sorted(stats.keys())
    for outer, key_a in enumerate(keys):
        for key_b in keys[outer + 1 :]:
            pair = _safe_merge_pair(stats[key_a], stats[key_b], config)
            if pair.get("accepted"):
                candidates.append(pair)
    if bool(config.get("diagnostic_only", True)) or not bool(config.get("apply_merges", False)):
        return {}, candidates
    mapping = _mapping_from_pairs(candidates)
    mapping = _remove_conflicting_mappings(rows, mapping)
    return mapping, candidates


def apply_person_merge_mapping(rows: List[Dict[str, Any]], mapping: Dict[TrackKey, str]) -> List[Dict[str, Any]]:
    """Apply a Person-only global id remapping to rows."""
    if not mapping:
        return [dict(row) for row in rows]
    output = []
    for row in rows:
        copied = dict(row)
        key = track_key(copied)
        if key in mapping:
            copied["global_track_id"] = mapping[key]
        output.append(copied)
    return output


def _track_summaries(rows: List[Dict[str, Any]], class_id: int) -> Dict[TrackKey, Dict[str, Any]]:
    groups = {}
    for row in rows:
        if safe_int(row.get("class_id"), -1) != class_id:
            continue
        key = track_key(row)
        if key[3] in ("", "None"):
            continue
        groups.setdefault(key, []).append(row)
    return {key: _summary_for_group(key, group_rows) for key, group_rows in groups.items()}


def _summary_for_group(key: TrackKey, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = [safe_int(row.get("frame_id"), None) for row in rows]
    frames = sorted([frame for frame in frames if frame is not None])
    confs = [safe_float(row.get("confidence"), 0.0) or 0.0 for row in rows]
    cameras = sorted(set([str(row.get("camera_id", "")) for row in rows]))
    ordered = sorted(rows, key=lambda row: safe_int(row.get("frame_id"), 0) or 0)
    return {
        "key": key,
        "rows": len(rows),
        "frames": set(frames),
        "start_frame": frames[0] if frames else None,
        "end_frame": frames[-1] if frames else None,
        "entry_center": _center(ordered[0]) if ordered else None,
        "exit_center": _center(ordered[-1]) if ordered else None,
        "cameras": cameras,
        "mean_confidence": float(sum(confs) / float(len(confs))) if confs else 0.0,
    }


def _safe_merge_pair(left: Dict[str, Any], right: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    key_a = left["key"]
    key_b = right["key"]
    row = {
        "candidate_a": "|".join(key_a),
        "candidate_b": "|".join(key_b),
        "accepted": False,
        "reason": "not_checked",
    }
    if key_a[0] != key_b[0] or key_a[1] != key_b[1] or key_a[2] != key_b[2]:
        row["reason"] = "subset_scene_class_mismatch"
        return row
    if not bool(config.get("allow_same_camera", False)):
        if set(left["cameras"]).intersection(set(right["cameras"])):
            row["reason"] = "same_camera_not_allowed"
            return row
    if bool(config.get("require_no_temporal_overlap", True)):
        if left["frames"].intersection(right["frames"]):
            row["reason"] = "temporal_overlap"
            return row
    gap = _temporal_gap(left, right)
    if gap is None or gap > int(config.get("max_temporal_gap", 30)):
        row["reason"] = "temporal_gap_too_large"
        row["temporal_gap"] = gap
        return row
    distance = _entry_exit_distance(left, right)
    row["temporal_gap"] = gap
    row["entry_exit_distance"] = distance
    if distance is None or distance > float(config.get("max_entry_exit_distance", 1.5)):
        row["reason"] = "entry_exit_distance_too_large"
        return row
    min_conf = float(config.get("min_confidence", 0.05))
    if min(float(left["mean_confidence"]), float(right["mean_confidence"])) < min_conf:
        row["reason"] = "confidence_too_low"
        return row
    row["accepted"] = True
    row["reason"] = "ok"
    return row


def _mapping_from_pairs(pairs: List[Dict[str, Any]]) -> Dict[TrackKey, str]:
    parent = {}
    for pair in pairs:
        if not pair.get("accepted"):
            continue
        key_a = tuple(str(pair["candidate_a"]).split("|"))
        key_b = tuple(str(pair["candidate_b"]).split("|"))
        parent.setdefault(key_a, key_a)
        parent.setdefault(key_b, key_b)
        _union(parent, key_a, key_b)
    groups = {}
    for key in list(parent.keys()):
        groups.setdefault(_find(parent, key), []).append(key)
    mapping = {}
    for members in groups.values():
        new_id = sorted([member[3] for member in members], key=lambda value: int(float(value)) if str(value).replace(".", "", 1).isdigit() else str(value))[0]
        for member in members:
            mapping[member] = str(new_id)
    return mapping


def _remove_conflicting_mappings(rows: List[Dict[str, Any]], mapping: Dict[TrackKey, str]) -> Dict[TrackKey, str]:
    seen = set()
    conflict_new_ids = set()
    for row in rows:
        key = track_key(row)
        new_id = mapping.get(key, key[3])
        duplicate_key = (key[0], key[1], str(row.get("class_id", "")), str(new_id), str(row.get("frame_id", "")))
        if duplicate_key in seen:
            conflict_new_ids.add(str(new_id))
        seen.add(duplicate_key)
    return {key: value for key, value in mapping.items() if str(value) not in conflict_new_ids}


def _find(parent: Dict[Any, Any], key: Any) -> Any:
    if parent[key] != key:
        parent[key] = _find(parent, parent[key])
    return parent[key]


def _union(parent: Dict[Any, Any], left: Any, right: Any) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _temporal_gap(left: Dict[str, Any], right: Dict[str, Any]) -> Any:
    if left["end_frame"] is None or right["start_frame"] is None:
        return None
    if left["end_frame"] < right["start_frame"]:
        return int(right["start_frame"]) - int(left["end_frame"])
    if right["end_frame"] < left["start_frame"]:
        return int(left["start_frame"]) - int(right["end_frame"])
    return 0


def _entry_exit_distance(left: Dict[str, Any], right: Dict[str, Any]) -> Any:
    if left["end_frame"] is None or right["start_frame"] is None:
        return None
    if left["end_frame"] < right["start_frame"]:
        return _distance(left["exit_center"], right["entry_center"])
    return _distance(right["exit_center"], left["entry_center"])


def _center(row: Dict[str, Any]) -> Any:
    values = [safe_float(row.get("center_x"), None), safe_float(row.get("center_y"), None), safe_float(row.get("center_z"), None)]
    if any(value is None for value in values):
        return None
    return values


def _distance(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return None
    return sum([(float(left[index]) - float(right[index])) ** 2 for index in range(3)]) ** 0.5

