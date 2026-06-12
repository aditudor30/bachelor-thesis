"""Conservative union-find merging for learned Person association edges."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from deep_oc_sort_3d.person_association.person_association_io import TrackKey, parse_track_key
from deep_oc_sort_3d.learned_association_application.scorer_association_io import safe_float, safe_int


MERGE_DECISION_FIELDS = (
    "merge_id", "variant_name", "pair_id", "scene_name", "fragment_a_id", "fragment_b_id",
    "global_track_a_before", "global_track_b_before", "global_track_after", "camera_a", "camera_b",
    "temporal_gap", "spatial_distance", "reid_similarity", "mlp_score", "accepted", "rejection_reason",
)


class ConservativeUnionFind:
    """Union-find that retains component fragment metadata."""

    def __init__(self, max_component_size: int = 20) -> None:
        self.parent = {}  # type: Dict[TrackKey, TrackKey]
        self.members = {}  # type: Dict[TrackKey, Set[TrackKey]]
        self.metadata = {}  # type: Dict[TrackKey, List[Dict[str, Any]]]
        self.max_component_size = int(max_component_size)

    def add(self, key: TrackKey, metadata: Dict[str, Any]) -> None:
        """Register one graph node."""
        if key in self.parent:
            return
        self.parent[key] = key
        self.members[key] = {key}
        self.metadata[key] = [dict(metadata)]

    def find(self, key: TrackKey) -> TrackKey:
        """Return the compressed component root."""
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def try_union(self, left: TrackKey, right: TrackKey) -> Tuple[bool, str]:
        """Union components only when size and temporal-camera constraints hold."""
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False, "already_same_component"
        new_size = len(self.members[root_left]) + len(self.members[root_right])
        if new_size > self.max_component_size:
            return False, "max_component_size"
        if _component_temporal_conflict(self.metadata[root_left], self.metadata[root_right]):
            return False, "component_same_camera_temporal_conflict"
        canonical, merged = _ordered_roots(root_left, root_right)
        self.parent[merged] = canonical
        self.members[canonical].update(self.members.pop(merged))
        self.metadata[canonical].extend(self.metadata.pop(merged))
        return True, "ok"

    def mapping(self) -> Dict[TrackKey, str]:
        """Return old keys mapped to canonical global ids for merged components."""
        result = {}
        groups = {}  # type: Dict[TrackKey, List[TrackKey]]
        for key in self.parent.keys():
            groups.setdefault(self.find(key), []).append(key)
        for values in groups.values():
            if len(values) <= 1:
                continue
            global_id = _canonical_global_id([value[3] for value in values])
            for value in values:
                result[value] = global_id
        return result


def build_conservative_merge_mapping(
    rows: List[Dict[str, Any]],
    variant_name: str,
    config: Dict[str, Any],
) -> Tuple[Dict[TrackKey, str], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply variant gates followed by score-ordered conservative graph merging."""
    constraints = config.get("constraints", {})
    graph = ConservativeUnionFind(int(constraints.get("max_component_size", 20)))
    accepted = []  # type: List[Dict[str, Any]]
    rejected = []  # type: List[Dict[str, Any]]
    ordered = sorted(rows, key=lambda row: float(safe_float(row.get("mlp_score"), 0.0) or 0.0), reverse=True)
    for index, row in enumerate(ordered):
        key_a = pair_track_key(row, "a")
        key_b = pair_track_key(row, "b")
        decision = _decision_row(row, variant_name, index)
        reason = variant_rejection_reason(row, variant_name, config)
        if reason != "ok":
            decision["accepted"] = 0
            decision["rejection_reason"] = reason
            rejected.append(decision)
            continue
        if key_a is None or key_b is None:
            decision["accepted"] = 0
            decision["rejection_reason"] = "invalid_track_key"
            rejected.append(decision)
            continue
        graph.add(key_a, fragment_metadata(row, "a"))
        graph.add(key_b, fragment_metadata(row, "b"))
        merged, graph_reason = graph.try_union(key_a, key_b)
        decision["accepted"] = int(merged)
        decision["rejection_reason"] = graph_reason
        if merged:
            decision["global_track_after"] = graph.mapping().get(key_a, _canonical_global_id([key_a[3], key_b[3]]))
            accepted.append(decision)
        else:
            rejected.append(decision)
    mapping = graph.mapping()
    summary = {
        "variant_name": variant_name,
        "candidate_pairs": len(rows),
        "accepted_edges": len(accepted),
        "accepted_edges_with_reid": len([row for row in accepted if safe_float(row.get("reid_similarity"), None) is not None]),
        "rejected_edges": len(rejected),
        "mapping_size": len(mapping),
        "merged_components": len(set(mapping.values())),
        "reject_reasons": _count_reasons(rejected),
    }
    return mapping, accepted, rejected, summary


def variant_rejection_reason(row: Dict[str, Any], variant_name: str, config: Dict[str, Any]) -> str:
    """Apply the configured MLP/ReID/geometry gates for one variant."""
    thresholds = config.get("thresholds", {})
    constraints = config.get("constraints", {})
    mlp = safe_float(row.get("mlp_score"), None)
    reid = safe_float(row.get("reid_similarity"), None)
    spatial = safe_float(row.get("spatial_distance"), None)
    temporal = safe_float(row.get("temporal_gap"), None)
    if str(row.get("class_name", "Person")).lower() != "person" and safe_int(row.get("class_id"), -1) != 0:
        return "non_person"
    if str(row.get("global_track_a", "")) == str(row.get("global_track_b", "")):
        return "same_global_track"
    if not bool(safe_int(row.get("valid_for_merge"), 0)):
        return str(row.get("rejection_reason") or "hard_constraint_failed")
    mlp_threshold = float(thresholds.get("mlp_very_strict", 0.85)) if variant_name == "mlp_very_strict_085" else float(thresholds.get("mlp_strict", 0.77))
    if mlp is None or mlp < mlp_threshold:
        return "mlp_score_below_threshold"
    if reid is None and bool(config.get("candidate_scoring", {}).get("require_valid_reid", True)):
        return "missing_reid"
    if variant_name in ("mlp_reid_gate_080", "mlp_geometry_safe", "mlp_combined_export_compact"):
        if reid is None or reid < float(thresholds.get("reid_gate_080", 0.80)):
            return "reid_below_080"
    if variant_name == "mlp_reid_gate_085":
        if reid is None or reid < float(thresholds.get("reid_gate_085", 0.85)):
            return "reid_below_085"
    safe_geometry = variant_name in ("mlp_geometry_safe", "mlp_combined_export_compact")
    max_spatial = float(constraints.get("max_spatial_distance_safe" if safe_geometry else "max_spatial_distance_default", 8.0 if safe_geometry else 12.0))
    max_temporal = float(constraints.get("max_temporal_gap_safe" if safe_geometry else "max_temporal_gap_default", 150 if safe_geometry else 300))
    if spatial is not None and spatial > max_spatial:
        return "spatial_distance_too_large"
    if safe_geometry and spatial is None:
        return "missing_geometry"
    if temporal is not None and temporal > max_temporal:
        return "temporal_gap_too_large"
    if bool(constraints.get("forbid_same_camera_temporal_overlap", True)) and safe_int(row.get("same_camera_temporal_conflict"), 0):
        return "same_camera_temporal_conflict"
    return "ok"


def pair_track_key(row: Dict[str, Any], suffix: str) -> Optional[TrackKey]:
    """Read a stable track key from a scored pair."""
    fragment_value = row.get("fragment_%s_id" % suffix)
    if fragment_value:
        parsed = parse_track_key(fragment_value)
        if "" not in parsed:
            return parsed
    subset = str(row.get("subset", ""))
    scene = str(row.get("scene_name", ""))
    class_id = str(safe_int(row.get("class_id"), 0))
    global_id = str(row.get("global_track_%s" % suffix, ""))
    if not subset or not scene or not global_id:
        return None
    return (subset, scene, class_id, global_id)


def fragment_metadata(row: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Extract interval and camera metadata for component conflict checks."""
    camera_text = str(row.get("camera_%s" % suffix, ""))
    return {
        "cameras": set([value for value in camera_text.split(";") if value]),
        "frame_start": safe_int(row.get("frame_start_%s" % suffix), None),
        "frame_end": safe_int(row.get("frame_end_%s" % suffix), None),
    }


def _component_temporal_conflict(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]]) -> bool:
    for item_a in left:
        for item_b in right:
            if not item_a.get("cameras", set()).intersection(item_b.get("cameras", set())):
                continue
            if _intervals_overlap(item_a.get("frame_start"), item_a.get("frame_end"), item_b.get("frame_start"), item_b.get("frame_end")):
                return True
    return False


def _intervals_overlap(start_a: Any, end_a: Any, start_b: Any, end_b: Any) -> bool:
    if None in (start_a, end_a, start_b, end_b):
        return False
    return max(int(start_a), int(start_b)) <= min(int(end_a), int(end_b))


def _decision_row(row: Dict[str, Any], variant_name: str, index: int) -> Dict[str, Any]:
    return {
        "merge_id": "%s_%08d" % (variant_name, index),
        "variant_name": variant_name,
        "pair_id": row.get("pair_id", ""),
        "scene_name": row.get("scene_name", ""),
        "fragment_a_id": row.get("fragment_a_id", ""),
        "fragment_b_id": row.get("fragment_b_id", ""),
        "global_track_a_before": row.get("global_track_a", ""),
        "global_track_b_before": row.get("global_track_b", ""),
        "global_track_after": "",
        "camera_a": row.get("camera_a", ""),
        "camera_b": row.get("camera_b", ""),
        "temporal_gap": row.get("temporal_gap", ""),
        "spatial_distance": row.get("spatial_distance", ""),
        "reid_similarity": row.get("reid_similarity", ""),
        "mlp_score": row.get("mlp_score", ""),
    }


def _ordered_roots(left: TrackKey, right: TrackKey) -> Tuple[TrackKey, TrackKey]:
    return (left, right) if _track_sort_key(left) <= _track_sort_key(right) else (right, left)


def _track_sort_key(key: TrackKey) -> Tuple[str, str, str, Tuple[int, Any]]:
    return (key[0], key[1], key[2], _global_sort_key(key[3]))


def _canonical_global_id(values: Sequence[str]) -> str:
    return str(sorted([str(value) for value in values], key=_global_sort_key)[0])


def _global_sort_key(value: str) -> Tuple[int, Any]:
    try:
        return (0, int(float(value)))
    except (TypeError, ValueError):
        return (1, str(value))


def _count_reasons(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    result = {}
    for row in rows:
        reason = str(row.get("rejection_reason", "unknown"))
        result[reason] = result.get(reason, 0) + 1
    return result
