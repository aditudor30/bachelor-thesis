"""Reusable metrics for fragmentation audits."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    add_count,
    rate,
    safe_float,
    safe_int,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds


def length_distribution(lengths: List[int], thresholds: Optional[FragmentationThresholds] = None) -> Dict[str, Any]:
    """Compute distribution stats for track/candidate lengths."""
    cfg = thresholds if thresholds is not None else FragmentationThresholds()
    values = sorted([int(item) for item in lengths if int(item) >= 0])
    total = len(values)
    singleton = len([item for item in values if item <= cfg.singleton_length])
    short = len([item for item in values if item <= cfg.short_track_length])
    very_short = len([item for item in values if item <= cfg.very_short_track_length])
    long_count = len([item for item in values if item >= cfg.long_track_length])
    return {
        "count": total,
        "mean": mean(values),
        "median": percentile(values, 50.0),
        "p05": percentile(values, 5.0),
        "p25": percentile(values, 25.0),
        "p75": percentile(values, 75.0),
        "p95": percentile(values, 95.0),
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
        "singleton_count": singleton,
        "singleton_ratio": rate(singleton, total),
        "short_count": short,
        "short_ratio": rate(short, total),
        "very_short_count": very_short,
        "very_short_ratio": rate(very_short, total),
        "long_count": long_count,
        "long_ratio": rate(long_count, total),
    }


def mean(values: Iterable[Any]) -> Optional[float]:
    """Mean for numeric values."""
    nums = [float(item) for item in values if item is not None]
    if not nums:
        return None
    return sum(nums) / float(len(nums))


def percentile(values: List[Any], pct: float) -> Optional[float]:
    """Simple percentile without numpy dependency."""
    if not values:
        return None
    nums = sorted([float(item) for item in values])
    if len(nums) == 1:
        return nums[0]
    pos = (float(pct) / 100.0) * float(len(nums) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(nums) - 1)
    frac = pos - float(lo)
    return nums[lo] * (1.0 - frac) + nums[hi] * frac


def row_scope(row: Dict[str, Any]) -> Dict[str, str]:
    """Extract common subset/scene/camera/class fields from a row."""
    return {
        "subset": str(row.get("subset") or row.get("split") or "unknown"),
        "scene": str(row.get("scene_name") or row.get("scene") or "unknown"),
        "camera": str(row.get("camera_id") or "unknown"),
        "class": str(row.get("class_name") or row.get("class_id") or "unknown"),
    }


def scope_key(row: Dict[str, Any], field: str) -> str:
    """Return one common grouping key."""
    return row_scope(row).get(field, "unknown")


def add_scope_counts(output: Dict[str, Dict[str, int]], row: Dict[str, Any], amount: int = 1) -> None:
    """Increment common scope counters."""
    scope = row_scope(row)
    for name, key in scope.items():
        add_count(output.setdefault("per_%s" % name, {}), key, amount)


def track_key(row: Dict[str, Any], id_field: str) -> Tuple[str, str, str, str, int]:
    """Build a stable per-camera track key."""
    scope = row_scope(row)
    return (
        scope["subset"],
        scope["scene"],
        scope["camera"],
        scope["class"],
        safe_int(row.get(id_field), -1),
    )


def update_track_accumulator(
    tracks: Dict[Tuple[str, str, str, str, int], Dict[str, Any]],
    row: Dict[str, Any],
    id_field: str,
) -> None:
    """Accumulate frame rows into per-track length records."""
    key = track_key(row, id_field)
    item = tracks.get(key)
    frame_id = safe_int(row.get("frame_id"), -1)
    if item is None:
        scope = row_scope(row)
        item = {
            "subset": scope["subset"],
            "scene": scope["scene"],
            "camera": scope["camera"],
            "class": scope["class"],
            "track_id": key[-1],
            "length": 0,
            "start_frame": frame_id,
            "end_frame": frame_id,
            "gt_ids": {},
        }
        tracks[key] = item
    item["length"] += 1
    if frame_id >= 0:
        item["start_frame"] = frame_id if item["start_frame"] < 0 else min(item["start_frame"], frame_id)
        item["end_frame"] = max(item["end_frame"], frame_id)
    gt_id = row.get("matched_gt_object_id")
    if gt_id not in (None, ""):
        add_count(item["gt_ids"], gt_id)


def summarize_track_accumulator(
    tracks: Dict[Tuple[str, str, str, str, int], Dict[str, Any]],
    thresholds: Optional[FragmentationThresholds] = None,
) -> Dict[str, Any]:
    """Summarize accumulated per-track lengths."""
    lengths = [safe_int(item.get("length")) for item in tracks.values()]
    summary = length_distribution(lengths, thresholds)
    summary["num_tracks"] = summary.pop("count")
    summary["per_subset"] = {}
    summary["per_scene"] = {}
    summary["per_camera"] = {}
    summary["per_class"] = {}
    summary["gt_fragmentation_approx"] = 0
    gt_to_tracks = {}
    for item in tracks.values():
        row = {"subset": item["subset"], "scene_name": item["scene"], "camera_id": item["camera"], "class_name": item["class"]}
        add_scope_counts(summary, row)
        for gt_id in item.get("gt_ids", {}).keys():
            gt_key = "%s/%s/%s/%s" % (item["subset"], item["scene"], item["class"], gt_id)
            gt_to_tracks[gt_key] = gt_to_tracks.get(gt_key, 0) + 1
    summary["gt_fragmentation_approx"] = sum(max(0, int(count) - 1) for count in gt_to_tracks.values())
    return summary


def summarize_records_by_id(
    rows: Iterable[Dict[str, Any]],
    id_field: str,
    thresholds: Optional[FragmentationThresholds] = None,
) -> Dict[str, Any]:
    """Summarize frame rows grouped by one id field."""
    tracks = {}
    total = 0
    for row in rows:
        total += 1
        update_track_accumulator(tracks, row, id_field)
    summary = summarize_track_accumulator(tracks, thresholds)
    summary["total_records"] = total
    return summary


def metric_delta(left: Any, right: Any) -> Optional[float]:
    """Return right-left for numeric values."""
    left_value = safe_float(left)
    right_value = safe_float(right)
    if left_value is None or right_value is None:
        return None
    return right_value - left_value

