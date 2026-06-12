"""Lightweight downstream estimates without running the global MTMC pipeline."""

from typing import Any, Dict, List, Sequence

from deep_oc_sort_3d.local_tracker_benchmark.local_track_metrics import compute_track_metrics, group_tracks


def compute_downstream_probe(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate tracklet/candidate pressure and bbox motion quality."""
    metrics = compute_track_metrics(rows)
    groups = group_tracks(rows)
    good = 0
    suspicious = 0
    invalid = 0
    for values in groups.values():
        ordered = sorted(values, key=lambda row: int(float(row.get("frame_id", 0))))
        jumps = []
        for left, right in zip(ordered[:-1], ordered[1:]):
            left_center = _center(left)
            right_center = _center(right)
            width = max(1.0, float(left.get("bbox_x2", 0.0)) - float(left.get("bbox_x1", 0.0)))
            height = max(1.0, float(left.get("bbox_y2", 0.0)) - float(left.get("bbox_y1", 0.0)))
            normalized = (((right_center[0] - left_center[0]) / width) ** 2 + ((right_center[1] - left_center[1]) / height) ** 2) ** 0.5
            jumps.append(normalized)
        maximum = max(jumps) if jumps else 0.0
        if maximum > 5.0:
            invalid += 1
        elif maximum > 2.0:
            suspicious += 1
        else:
            good += 1
    tracklets = len(groups)
    return {
        "tracklet_count_probe": tracklets,
        "short_tracklet_ratio": metrics.get("short_track_ratio_le3"),
        "candidate_count_probe": _candidate_pressure(groups),
        "motion_good": good,
        "motion_suspicious": suspicious,
        "motion_invalid": invalid,
    }


def _candidate_pressure(groups: Dict[Any, List[Dict[str, Any]]]) -> int:
    counts = {}
    for key in groups.keys():
        scene_class = (key[0], key[1], key[3])
        counts[scene_class] = counts.get(scene_class, 0) + 1
    return sum([count * (count - 1) // 2 for count in counts.values()])


def _center(row: Dict[str, Any]) -> Any:
    return (
        (float(row.get("bbox_x1", 0.0)) + float(row.get("bbox_x2", 0.0))) / 2.0,
        (float(row.get("bbox_y1", 0.0)) + float(row.get("bbox_y2", 0.0))) / 2.0,
    )
