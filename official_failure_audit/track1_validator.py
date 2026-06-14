"""Validation and structural summaries for Track1-like predictions."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_failure_audit.failure_io import write_csv, write_json
from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row


def audit_prediction_rows(
    rows: Sequence[AuditTrack1Row], output_root: Path, source_summary: Dict[str, Any],
) -> Dict[str, Any]:
    valid_scenes = {20, 21, 22}
    valid_classes = set(range(7))
    duplicates = 0
    seen = set()
    non_positive = 0
    non_finite = 0
    per_scene: Dict[str, int] = defaultdict(int)
    per_class: Dict[str, int] = defaultdict(int)
    per_frame: Dict[str, int] = defaultdict(int)
    track_lengths: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for row in rows:
        if row.key() in seen:
            duplicates += 1
        seen.add(row.key())
        if row.width <= 0.0 or row.length <= 0.0 or row.height <= 0.0:
            non_positive += 1
        if not all(math.isfinite(value) for value in [row.x, row.y, row.z, row.width, row.length, row.height, row.yaw]):
            non_finite += 1
        per_scene[str(row.scene_id)] += 1
        per_class[str(row.class_id)] += 1
        per_frame["%d:%d" % (row.scene_id, row.frame_id)] += 1
        track_lengths[(row.scene_id, row.class_id, row.object_id)] += 1
    report = {
        "status": "ok" if rows and not non_finite else "error",
        "rows": len(rows), "tracks": len(track_lengths), "scene_ids": sorted(set(row.scene_id for row in rows)),
        "class_ids": sorted(set(row.class_id for row in rows)),
        "invalid_scene_rows": sum(1 for row in rows if row.scene_id not in valid_scenes),
        "invalid_class_rows": sum(1 for row in rows if row.class_id not in valid_classes),
        "non_positive_dimensions": non_positive, "nan_or_inf_values": non_finite,
        "duplicate_keys": duplicates, "per_scene_rows": dict(sorted(per_scene.items())),
        "per_class_rows": dict(sorted(per_class.items())),
        "coordinate_frame_distribution": _count_strings([row.coordinate_frame for row in rows]),
        "rows_per_track": _stats(list(track_lengths.values())),
        "source_summary": source_summary,
    }
    root = output_root / "pred_audit"
    write_json(root / "pred_source_summary.json", source_summary)
    write_json(root / "pred_validation_summary.json", report)
    write_csv(root / "pred_range_summary.csv", _range_rows(rows), ["field", "min", "max", "mean", "median"])
    write_csv(root / "pred_class_distribution.csv", [
        {"class_id": key, "rows": value, "tracks": len(set((row.scene_id, row.object_id) for row in rows if str(row.class_id) == key))}
        for key, value in sorted(per_class.items(), key=lambda item: int(item[0]))
    ])
    write_csv(root / "pred_frame_summary.csv", [
        {"scene_frame": key, "rows": value} for key, value in sorted(per_frame.items())
    ])
    return report


def _range_rows(rows: Sequence[AuditTrack1Row]) -> List[Dict[str, Any]]:
    output = []
    for field in ["x", "y", "z", "width", "length", "height", "yaw", "frame_id"]:
        values = np.asarray([float(getattr(row, field)) for row in rows], dtype=float)
        output.append({
            "field": field, "min": float(np.min(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
        })
    return output


def _stats(values: Sequence[int]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "min": int(np.min(array)) if array.size else None, "max": int(np.max(array)) if array.size else None,
        "mean": float(np.mean(array)) if array.size else None, "median": float(np.median(array)) if array.size else None,
    }


def _count_strings(values: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))
