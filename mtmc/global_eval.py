"""Diagnostic evaluation for global MTMC association."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.mtmc.global_types import GlobalTrack


def evaluate_global_tracks(global_tracks: List[GlobalTrack]) -> Dict[str, Any]:
    """Evaluate global tracks using diagnostic GT ids when available."""
    multi_camera = [track for track in global_tracks if track.num_cameras > 1]
    singleton = [track for track in global_tracks if track.num_cameras <= 1]
    gt_tracks = [track for track in global_tracks if track.num_gt_ids > 0]
    purity_values = [float(track.gt_purity) for track in gt_tracks if track.gt_purity is not None]
    false_merges = [track for track in gt_tracks if track.num_gt_ids > 1]
    fragmentation, per_class_fragmentation = _fragmentation(global_tracks)
    per_class_purity = _per_class_purity(gt_tracks)
    return {
        "num_global_tracks": len(global_tracks),
        "num_multi_camera_tracks": len(multi_camera),
        "num_singleton_tracks": len(singleton),
        "mean_candidates_per_global_track": _mean([track.num_candidates for track in global_tracks]),
        "mean_cameras_per_global_track": _mean([track.num_cameras for track in global_tracks]),
        "per_class_global_tracks": _count_by(global_tracks, "class_name"),
        "per_class_multi_camera_tracks": _count_by(multi_camera, "class_name"),
        "gt_available_tracks": len(gt_tracks),
        "global_purity_mean": _mean(purity_values),
        "global_purity_median": _median(purity_values),
        "false_merge_count": len(false_merges),
        "false_merge_rate": _ratio(len(false_merges), len(gt_tracks)),
        "fragmentation_approx": fragmentation,
        "gt_object_coverage": len(_gt_to_tracks(global_tracks)),
        "per_class_purity": per_class_purity,
        "per_class_fragmentation": per_class_fragmentation,
        "gt_note": "" if gt_tracks else "GT diagnostic unavailable for these global tracks.",
    }


def save_global_eval_json(metrics: Dict[str, Any], path: Path) -> None:
    """Save evaluation metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_global_eval_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Save evaluation metrics as compact CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _fragmentation(global_tracks: List[GlobalTrack]) -> Any:
    gt_to_tracks = _gt_to_tracks(global_tracks)
    fragmentation = 0
    per_class = {}
    id_to_class = {}
    for track in global_tracks:
        for gt_id in track.gt_id_counts.keys():
            id_to_class[gt_id] = track.class_name
    for gt_id, track_ids in gt_to_tracks.items():
        value = max(0, len(track_ids) - 1)
        fragmentation += value
        class_name = id_to_class.get(gt_id, "unknown")
        per_class[class_name] = per_class.get(class_name, 0) + value
    return fragmentation, per_class


def _gt_to_tracks(global_tracks: List[GlobalTrack]) -> Dict[str, List[int]]:
    output = {}
    for track in global_tracks:
        for gt_id in track.gt_id_counts.keys():
            output.setdefault(str(gt_id), []).append(int(track.global_track_id))
    return output


def _per_class_purity(global_tracks: List[GlobalTrack]) -> Dict[str, Optional[float]]:
    values = {}
    for track in global_tracks:
        if track.gt_purity is None:
            continue
        values.setdefault(track.class_name, []).append(float(track.gt_purity))
    return {key: _mean(items) for key, items in values.items()}


def _count_by(global_tracks: List[GlobalTrack], field: str) -> Dict[str, int]:
    counts = {}
    for track in global_tracks:
        key = str(getattr(track, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _mean(values: List[Any]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[Any]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)
