"""Fragmentation and duration metrics for local tracker benchmark outputs."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def compute_track_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute requested length, confidence and duration statistics."""
    groups = group_tracks(rows)
    lengths = [len(values) for values in groups.values()]
    durations = []
    confidences = []
    for values in groups.values():
        frames = [int(float(row.get("frame_id", 0))) for row in values]
        durations.append(max(frames) - min(frames) + 1 if frames else 0)
        confidences.extend([float(row.get("confidence", 0.0)) for row in values])
    return {
        "num_tracks": len(groups),
        "num_records": len(rows),
        "mean_track_length": _mean(lengths),
        "median_track_length": _percentile(lengths, 50),
        "p25_track_length": _percentile(lengths, 25),
        "p75_track_length": _percentile(lengths, 75),
        "p90_track_length": _percentile(lengths, 90),
        "num_length_1_tracks": _count_at_most(lengths, 1),
        "num_length_le_3_tracks": _count_at_most(lengths, 3),
        "num_length_le_5_tracks": _count_at_most(lengths, 5),
        "short_track_ratio_len1": _ratio_at_most(lengths, 1),
        "short_track_ratio_le3": _ratio_at_most(lengths, 3),
        "short_track_ratio_le5": _ratio_at_most(lengths, 5),
        "mean_confidence": _mean(confidences),
        "median_confidence": _percentile(confidences, 50),
        "track_duration_mean": _mean(durations),
        "track_duration_median": _percentile(durations, 50),
    }


def compute_person_vs_nonperson(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return separate Person and non-Person metrics."""
    person = [row for row in rows if int(float(row.get("class_id", -1))) == 0]
    non_person = [row for row in rows if int(float(row.get("class_id", -1))) != 0]
    output = []
    for label, values in (("Person", person), ("NonPerson", non_person)):
        metric = compute_track_metrics(values)
        metric["group"] = label
        output.append(metric)
    return output


def group_tracks(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]]:
    """Group by subset/scene/camera/class/track to avoid id namespace collisions."""
    groups = {}
    for row in rows:
        key = (
            str(row.get("subset", "")), str(row.get("scene_name", "")), str(row.get("camera_id", "")),
            str(row.get("class_id", "")), str(row.get("track_id", row.get("local_track_id", ""))),
        )
        groups.setdefault(key, []).append(dict(row))
    return groups


def metric_rows_by_fields(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """Aggregate metrics for arbitrary scene/camera/class group fields."""
    groups = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        metric = compute_track_metrics(values)
        for field, value in zip(fields, key):
            metric[field] = value
        output.append(metric)
    return output


def _mean(values: Sequence[float]) -> Any:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: Sequence[float], percentile: float) -> Any:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _count_at_most(values: Sequence[int], threshold: int) -> int:
    return len([value for value in values if int(value) <= int(threshold)])


def _ratio_at_most(values: Sequence[int], threshold: int) -> Any:
    return None if not values else float(_count_at_most(values, threshold)) / float(len(values))
