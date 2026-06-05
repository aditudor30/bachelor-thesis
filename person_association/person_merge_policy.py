"""Conservative Person-only merge policies for association experiments."""

from typing import Any, Dict, List, Optional, Set, Tuple

from deep_oc_sort_3d.person_association.person_association_io import (
    TrackKey,
    parse_track_key,
    row_track_key,
    safe_float,
    safe_int,
    serialize_track_key,
)


def build_person_merge_mapping(
    scored_pairs: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[Dict[TrackKey, str], List[Dict[str, Any]]]:
    """Build a Person-only old track key -> new global id mapping."""
    selected_edges, audit_rows = select_merge_edges(scored_pairs, config)
    if not bool(config.get("apply_merges", True)):
        return {}, audit_rows
    mapping = mapping_from_edges(selected_edges)
    if bool(config.get("prevent_duplicate_frame_keys", True)):
        mapping, conflict_rows = remove_conflicting_mappings(all_rows, mapping)
        audit_rows.extend(conflict_rows)
    return mapping, audit_rows


def select_merge_edges(
    scored_pairs: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter scored pair rows into merge edges with audit reasons."""
    edges = []
    audit = []
    max_score = float(config.get("max_pair_score", 0.45))
    min_conf = float(config.get("min_mean_confidence", 0.03))
    reject_known_false = bool(config.get("reject_known_false_gt", True))
    for row in scored_pairs:
        decision = dict(row)
        reason = "ok"
        if str(row.get("candidate_status")) != "ok":
            reason = str(row.get("reject_reason", "candidate_rejected"))
        elif str(row.get("score_status")) != "scored":
            reason = "not_scored"
        elif safe_float(row.get("pair_score"), None) is None or float(row.get("pair_score")) > max_score:
            reason = "pair_score_too_high"
        elif (safe_float(row.get("min_mean_confidence"), 0.0) or 0.0) < min_conf:
            reason = "confidence_too_low"
        elif reject_known_false and row.get("same_gt_diagnostic") == "false_match":
            reason = "known_false_gt_diagnostic"
        decision["merge_selected"] = reason == "ok"
        decision["merge_reject_reason"] = reason
        audit.append(decision)
        if reason == "ok":
            edges.append(decision)
    return edges, audit


def mapping_from_edges(edges: List[Dict[str, Any]]) -> Dict[TrackKey, str]:
    """Create a union-find mapping from selected edges."""
    parent: Dict[TrackKey, TrackKey] = {}
    for row in edges:
        key_a = parse_track_key(row.get("track_a"))
        key_b = parse_track_key(row.get("track_b"))
        if "" in key_a or "" in key_b:
            continue
        parent.setdefault(key_a, key_a)
        parent.setdefault(key_b, key_b)
        _union(parent, key_a, key_b)
    groups: Dict[TrackKey, List[TrackKey]] = {}
    for key in list(parent.keys()):
        groups.setdefault(_find(parent, key), []).append(key)
    mapping: Dict[TrackKey, str] = {}
    for members in groups.values():
        if len(members) <= 1:
            continue
        new_id = _canonical_global_id([member[3] for member in members])
        for member in members:
            mapping[member] = new_id
    return mapping


def apply_person_merge_mapping(rows: List[Dict[str, Any]], mapping: Dict[TrackKey, str]) -> List[Dict[str, Any]]:
    """Apply mapping to Person rows only; non-Person is preserved."""
    if not mapping:
        return [dict(row) for row in rows]
    output = []
    for row in rows:
        copied = dict(row)
        key = row_track_key(copied)
        if safe_int(copied.get("class_id"), -1) == 0 and key in mapping:
            copied["global_track_id"] = mapping[key]
        output.append(copied)
    return output


def remove_conflicting_mappings(
    rows: List[Dict[str, Any]],
    mapping: Dict[TrackKey, str],
) -> Tuple[Dict[TrackKey, str], List[Dict[str, Any]]]:
    """Remove mapped components that would create duplicate frame/global keys."""
    if not mapping:
        return {}, []
    seen: Set[Tuple[str, str, str, str, str]] = set()
    conflicting_new_ids: Set[str] = set()
    conflict_rows = []
    for row in rows:
        key = row_track_key(row)
        new_id = mapping.get(key, key[3])
        duplicate_key = (
            key[0],
            str(row.get("scene_name", "")),
            str(row.get("class_id", "")),
            str(new_id),
            str(row.get("frame_id", "")),
        )
        if duplicate_key in seen:
            conflicting_new_ids.add(str(new_id))
            conflict_rows.append(
                {
                    "merge_selected": False,
                    "merge_reject_reason": "duplicate_frame_key_after_mapping",
                    "new_global_track_id": str(new_id),
                    "duplicate_key": "|".join(duplicate_key),
                }
            )
        seen.add(duplicate_key)
    clean_mapping = {key: value for key, value in mapping.items() if str(value) not in conflicting_new_ids}
    return clean_mapping, conflict_rows


def summarize_merge_audit(audit_rows: List[Dict[str, Any]], mapping: Dict[TrackKey, str]) -> Dict[str, Any]:
    """Summarize merge decisions."""
    reasons: Dict[str, int] = {}
    selected = 0
    for row in audit_rows:
        reason = str(row.get("merge_reject_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
        if row.get("merge_selected") in (True, "True", "true", "1"):
            selected += 1
    return {
        "candidate_rows": len(audit_rows),
        "selected_edges_before_conflict_filter": selected,
        "mapping_size": len(mapping),
        "merged_components": len(set(mapping.values())),
        "reject_reasons": reasons,
    }


def mapping_rows(mapping: Dict[TrackKey, str]) -> List[Dict[str, Any]]:
    """Serialize mapping to rows."""
    rows = []
    for key, new_id in sorted(mapping.items()):
        rows.append(
            {
                "old_track_key": serialize_track_key(key),
                "subset": key[0],
                "scene_name": key[1],
                "class_id": key[2],
                "old_global_track_id": key[3],
                "new_global_track_id": new_id,
            }
        )
    return rows


def _find(parent: Dict[TrackKey, TrackKey], key: TrackKey) -> TrackKey:
    if parent[key] != key:
        parent[key] = _find(parent, parent[key])
    return parent[key]


def _union(parent: Dict[TrackKey, TrackKey], left: TrackKey, right: TrackKey) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _canonical_global_id(values: List[str]) -> str:
    def _sort_key(value: str) -> Tuple[int, Any]:
        try:
            return (0, int(float(value)))
        except (TypeError, ValueError):
            return (1, str(value))

    return str(sorted(values, key=_sort_key)[0])

