"""Diagnostic evaluation for final frame-level global exports."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.final_export.global_frame_types import GlobalFrameRecord


def evaluate_global_frame_records(records: List[GlobalFrameRecord]) -> Dict[str, Any]:
    """Evaluate frame-level global ids using matched GT only for diagnostics."""
    assigned = [record for record in records if record.global_track_id is not None]
    unassigned = [record for record in records if record.global_track_id is None]
    gt_records = [record for record in assigned if record.matched_gt_object_id is not None]
    purity_values = _purity_values(gt_records)
    false_merges = _false_merge_count(gt_records)
    fragmentation = _fragmentation(gt_records)
    return {
        "num_records": len(records),
        "assigned_records": len(assigned),
        "unassigned_records": len(unassigned),
        "unique_global_tracks": len(set([record.global_track_id for record in assigned])),
        "records_with_gt": len(gt_records),
        "global_id_purity_mean": _mean(purity_values),
        "global_id_purity_median": _median(purity_values),
        "false_merge_count": false_merges,
        "fragmentation_approx": fragmentation,
        "per_class_records": _count_by(records, "class_name"),
        "per_class_global_tracks": _per_class_global_tracks(assigned),
        "per_class_purity": _per_class_purity(gt_records),
        "gt_object_coverage": len(set([record.matched_gt_object_id for record in gt_records])),
        "gt_note": "" if gt_records else "GT diagnostic unavailable for these records.",
    }


def save_final_eval_json(metrics: Dict[str, Any], path: Path) -> None:
    """Save final eval metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_final_eval_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Save final eval metrics as compact CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _purity_values(records: List[GlobalFrameRecord]) -> List[float]:
    groups = _records_by_global_id(records)
    values = []
    for grouped in groups.values():
        counts = {}
        for record in grouped:
            key = str(record.matched_gt_object_id)
            counts[key] = counts.get(key, 0) + 1
        if counts:
            values.append(float(max(counts.values())) / float(sum(counts.values())))
    return values


def _false_merge_count(records: List[GlobalFrameRecord]) -> int:
    count = 0
    for grouped in _records_by_global_id(records).values():
        gt_ids = set([record.matched_gt_object_id for record in grouped if record.matched_gt_object_id is not None])
        if len(gt_ids) > 1:
            count += 1
    return count


def _fragmentation(records: List[GlobalFrameRecord]) -> int:
    gt_to_global = {}
    for record in records:
        if record.matched_gt_object_id is None or record.global_track_id is None:
            continue
        gt_to_global.setdefault(str(record.matched_gt_object_id), set()).add(int(record.global_track_id))
    return sum([max(0, len(values) - 1) for values in gt_to_global.values()])


def _records_by_global_id(records: List[GlobalFrameRecord]) -> Dict[int, List[GlobalFrameRecord]]:
    groups = {}
    for record in records:
        if record.global_track_id is None:
            continue
        groups.setdefault(int(record.global_track_id), []).append(record)
    return groups


def _count_by(records: List[GlobalFrameRecord], field: str) -> Dict[str, int]:
    counts = {}
    for record in records:
        key = str(getattr(record, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _per_class_global_tracks(records: List[GlobalFrameRecord]) -> Dict[str, int]:
    values = {}
    for record in records:
        if record.global_track_id is None:
            continue
        values.setdefault(record.class_name, set()).add(int(record.global_track_id))
    return {key: len(item) for key, item in values.items()}


def _per_class_purity(records: List[GlobalFrameRecord]) -> Dict[str, Optional[float]]:
    values = {}
    for global_id, grouped in _records_by_global_id(records).items():
        _unused_global_id = global_id
        class_name = grouped[0].class_name
        counts = {}
        for record in grouped:
            if record.matched_gt_object_id is None:
                continue
            key = str(record.matched_gt_object_id)
            counts[key] = counts.get(key, 0) + 1
        if counts:
            values.setdefault(class_name, []).append(float(max(counts.values())) / float(sum(counts.values())))
    return {key: _mean(item) for key, item in values.items()}


def _mean(values: List[Any]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[Any]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))
