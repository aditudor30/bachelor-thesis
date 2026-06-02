"""Simple local tracking diagnostics using matched GT object ids."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord


def evaluate_local_tracks(records: List[LocalTrackRecord]) -> Dict[str, Any]:
    """Evaluate local tracks with lightweight GT-based diagnostics."""
    by_track = _group_by(records, "local_track_id")
    lengths = [len(items) for items in by_track.values()]
    per_class_num_tracks = {}
    per_class_lengths = {}
    purities = []
    for track_id, items in by_track.items():
        class_name = items[0].class_name if items else "unknown"
        per_class_num_tracks[class_name] = per_class_num_tracks.get(class_name, 0) + 1
        if class_name not in per_class_lengths:
            per_class_lengths[class_name] = []
        per_class_lengths[class_name].append(len(items))
        purity = _track_purity(items)
        if purity is not None:
            purities.append(purity)

    gt_records = [record for record in records if record.matched_gt_object_id is not None]
    metrics = {
        "num_records": len(records),
        "num_tracks": len(by_track),
        "mean_track_length": _mean(lengths),
        "median_track_length": _median(lengths),
        "per_class_num_tracks": per_class_num_tracks,
        "per_class_mean_track_length": _per_class_mean(per_class_lengths),
        "num_gt_matched_records": len(gt_records),
        "id_switches_approx": _id_switches_approx(gt_records),
        "fragmentations_approx": _fragmentations_approx(gt_records),
        "purity_mean": _mean(purities),
        "has_gt": len(gt_records) > 0,
    }
    if not gt_records:
        metrics["gt_note"] = "No matched GT object ids available; GT-based metrics are diagnostic zeros/None."
    return metrics


def save_track_eval_json(metrics: Dict[str, Any], path: Path) -> None:
    """Save tracking evaluation metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_track_eval_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Save tracking evaluation metrics as a compact CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, value in metrics.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _group_by(records: List[LocalTrackRecord], field: str) -> Dict[int, List[LocalTrackRecord]]:
    grouped = {}
    for record in records:
        key = int(getattr(record, field))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(record)
    return grouped


def _track_purity(records: List[LocalTrackRecord]) -> Any:
    counts = {}
    total = 0
    for record in records:
        if record.matched_gt_object_id is None:
            continue
        object_id = int(record.matched_gt_object_id)
        counts[object_id] = counts.get(object_id, 0) + 1
        total += 1
    if total == 0:
        return None
    return float(max(counts.values())) / float(total)


def _id_switches_approx(records: List[LocalTrackRecord]) -> int:
    by_gt = {}
    for record in records:
        object_id = int(record.matched_gt_object_id)
        if object_id not in by_gt:
            by_gt[object_id] = []
        by_gt[object_id].append(record)
    switches = 0
    for items in by_gt.values():
        items = sorted(items, key=lambda record: (record.frame_id, record.local_track_id))
        previous = None
        for record in items:
            if previous is not None and int(record.local_track_id) != int(previous):
                switches += 1
            previous = int(record.local_track_id)
    return switches


def _fragmentations_approx(records: List[LocalTrackRecord]) -> int:
    by_gt_tracks = {}
    for record in records:
        object_id = int(record.matched_gt_object_id)
        if object_id not in by_gt_tracks:
            by_gt_tracks[object_id] = set()
        by_gt_tracks[object_id].add(int(record.local_track_id))
    total = 0
    for track_ids in by_gt_tracks.values():
        total += max(len(track_ids) - 1, 0)
    return total


def _mean(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _per_class_mean(values_by_class: Dict[str, List[int]]) -> Dict[str, Any]:
    output = {}
    for class_name, values in values_by_class.items():
        output[class_name] = _mean(values)
    return output
