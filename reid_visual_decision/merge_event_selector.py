"""Select representative ReID merge events for visual inspection."""

from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import safe_float, safe_int


def select_events_for_review(
    events: List[Dict[str, Any]],
    max_events: int,
    threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """Select a diverse subset of merge events for manual visual review."""
    if max_events <= 0:
        return []
    buckets = bucket_events(events, threshold)
    selected: List[Dict[str, Any]] = []
    order = ["known_same_gt", "known_different_gt", "high_similarity", "near_threshold", "large_gap", "large_distance", "remaining"]
    per_bucket_limit = max(1, int(round(float(max_events) / float(max(1, len(order))))))
    for name in order:
        rows = sorted(buckets.get(name, []), key=lambda item: event_priority(item, threshold), reverse=True)
        for row in rows[:per_bucket_limit]:
            add_unique(selected, row)
            if len(selected) >= max_events:
                return selected
    leftovers = sorted(events, key=lambda item: event_priority(item, threshold), reverse=True)
    for row in leftovers:
        add_unique(selected, row)
        if len(selected) >= max_events:
            break
    return selected


def bucket_events(events: List[Dict[str, Any]], threshold: float) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket events by review interest."""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "known_same_gt": [],
        "known_different_gt": [],
        "high_similarity": [],
        "near_threshold": [],
        "large_gap": [],
        "large_distance": [],
        "remaining": [],
    }
    for row in events:
        label = str(row.get("reid_gt_pair_label") or row.get("same_gt_diagnostic"))
        sim = safe_float(row.get("reid_similarity"), None)
        gap = abs(safe_float(row.get("temporal_gap"), 0.0) or 0.0)
        dist = safe_float(row.get("spatial_distance"), 0.0) or 0.0
        if label in ("same_gt", "true_match"):
            buckets["known_same_gt"].append(row)
        elif label in ("different_gt", "false_match"):
            buckets["known_different_gt"].append(row)
        elif sim is not None and sim >= threshold + 0.08:
            buckets["high_similarity"].append(row)
        elif sim is not None and abs(sim - threshold) <= 0.04:
            buckets["near_threshold"].append(row)
        elif gap >= 120.0:
            buckets["large_gap"].append(row)
        elif dist >= 8.0:
            buckets["large_distance"].append(row)
        else:
            buckets["remaining"].append(row)
    return buckets


def event_priority(event: Dict[str, Any], threshold: float) -> float:
    """Score visual-review priority."""
    sim = safe_float(event.get("reid_similarity"), threshold) or threshold
    gap = abs(safe_float(event.get("temporal_gap"), 0.0) or 0.0)
    dist = safe_float(event.get("spatial_distance"), 0.0) or 0.0
    known_bad = 1.0 if str(event.get("reid_gt_pair_label")) == "different_gt" else 0.0
    near = 1.0 - min(1.0, abs(float(sim) - float(threshold)) / 0.20)
    return float(sim) + 0.10 * near + 0.01 * min(gap, 300.0) + 0.05 * min(dist, 20.0) + 2.0 * known_bad


def add_unique(selected: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    """Append row if merge_event_id not already selected."""
    event_id = str(row.get("merge_event_id", ""))
    if event_id in set([str(item.get("merge_event_id", "")) for item in selected]):
        return
    selected.append(row)

