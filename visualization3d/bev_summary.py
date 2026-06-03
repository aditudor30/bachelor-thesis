"""Coordinate summaries for robust BEV visualization."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from deep_oc_sort_3d.visualization3d.bev_track_selection import BEVTrack


def compute_bev_coordinate_summary(
    tracks: List[BEVTrack],
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> Dict[str, Any]:
    """Compute coordinate and track length statistics for BEV tracks."""
    x_values, y_values = _flatten_xy(tracks)
    lengths = np.asarray([track.length for track in tracks], dtype=float)
    per_class_counts = _per_class_counts(tracks)
    if x_values.size == 0 or y_values.size == 0:
        return {
            "num_tracks": len(tracks),
            "num_points": 0,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "x_p_low": None,
            "x_p_high": None,
            "y_p_low": None,
            "y_p_high": None,
            "outlier_points_count": 0,
            "outlier_points_ratio": 0.0,
            "track_length_mean": None,
            "track_length_median": None,
            "track_length_max": None,
            "per_class_counts": per_class_counts,
            "coordinate_units_note": "coordinate-space units; not map-aligned",
        }

    x_low = float(np.percentile(x_values, float(lower_percentile)))
    x_high = float(np.percentile(x_values, float(upper_percentile)))
    y_low = float(np.percentile(y_values, float(lower_percentile)))
    y_high = float(np.percentile(y_values, float(upper_percentile)))
    outside = np.logical_or.reduce(
        [
            x_values < x_low,
            x_values > x_high,
            y_values < y_low,
            y_values > y_high,
        ]
    )
    outlier_count = int(np.sum(outside))
    num_points = int(x_values.size)
    return {
        "num_tracks": len(tracks),
        "num_points": num_points,
        "x_min": float(np.min(x_values)),
        "x_max": float(np.max(x_values)),
        "y_min": float(np.min(y_values)),
        "y_max": float(np.max(y_values)),
        "x_p_low": x_low,
        "x_p_high": x_high,
        "y_p_low": y_low,
        "y_p_high": y_high,
        "outlier_points_count": outlier_count,
        "outlier_points_ratio": float(outlier_count) / float(max(num_points, 1)),
        "track_length_mean": float(np.mean(lengths)) if lengths.size else None,
        "track_length_median": float(np.median(lengths)) if lengths.size else None,
        "track_length_max": int(np.max(lengths)) if lengths.size else None,
        "per_class_counts": per_class_counts,
        "coordinate_units_note": "coordinate-space units; not map-aligned",
    }


def write_bev_summary_json(summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write BEV summary as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_bev_summary_csv(summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a flat BEV summary CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def print_bev_summary(summary: Dict[str, Any]) -> None:
    """Print key BEV summary fields."""
    for key in [
        "num_tracks",
        "num_points",
        "x_min",
        "x_max",
        "x_p_low",
        "x_p_high",
        "y_min",
        "y_max",
        "y_p_low",
        "y_p_high",
        "outlier_points_count",
        "outlier_points_ratio",
        "track_length_mean",
        "track_length_median",
        "track_length_max",
        "coordinate_units_note",
    ]:
        print("%s: %s" % (key, str(summary.get(key))))
    print("per_class_counts: %s" % json.dumps(summary.get("per_class_counts", {}), sort_keys=True))


def _flatten_xy(tracks: List[BEVTrack]) -> Any:
    x_values = []
    y_values = []
    for track in tracks:
        x_values.extend(track.x)
        y_values.extend(track.y)
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    valid = np.logical_and(np.isfinite(x_arr), np.isfinite(y_arr))
    return x_arr[valid], y_arr[valid]


def _per_class_counts(tracks: List[BEVTrack]) -> Dict[str, int]:
    counts = {}
    for track in tracks:
        if track.class_name is not None:
            key = str(track.class_name)
        elif track.class_id is not None:
            key = str(track.class_id)
        else:
            key = "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts

