"""Evaluation diagnostics for local tracklets."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


def evaluate_tracklets(tracklets: List[LocalTracklet]) -> Dict[str, Any]:
    """Evaluate local tracklet quality and GT-purity diagnostics."""
    lengths = [int(tracklet.length) for tracklet in tracklets]
    confidences = [float(tracklet.mean_confidence) for tracklet in tracklets]
    purity_values = [float(tracklet.gt_purity) for tracklet in tracklets if tracklet.gt_purity is not None]
    metrics = {
        "num_tracklets": len(tracklets),
        "valid_for_mtmc": len([item for item in tracklets if item.is_valid_for_mtmc]),
        "invalid_count": len([item for item in tracklets if not item.is_valid_for_mtmc]),
        "mean_length": _mean(lengths),
        "median_length": _median(lengths),
        "p25_length": _percentile(lengths, 25),
        "p75_length": _percentile(lengths, 75),
        "mean_confidence": _mean(confidences),
        "per_class_tracklets": _count_by(tracklets, "class_name"),
        "per_class_valid_tracklets": _count_valid_by(tracklets, "class_name"),
        "per_class_mean_length": _mean_length_by(tracklets, "class_name"),
        "per_camera_tracklets": _count_by(tracklets, "camera_id"),
        "per_scene_tracklets": _count_by(tracklets, "scene_name"),
        "gt_available_count": len(purity_values),
        "purity_mean": _mean(purity_values),
        "purity_median": _median(purity_values),
        "mixed_gt_tracklets": len([item for item in tracklets if item.num_gt_ids > 1]),
        "no_3d_tracklets": len([item for item in tracklets if not item.trajectory_3d]),
        "short_tracklets": len([item for item in tracklets if item.quality_flag == "short"]),
        "low_confidence_tracklets": len([item for item in tracklets if item.quality_flag == "low_confidence"]),
        "quality_flags": _count_by(tracklets, "quality_flag"),
    }
    if not purity_values:
        metrics["gt_note"] = "No GT ids available; purity metrics are None for this input."
    return metrics


def save_tracklet_eval_json(metrics: Dict[str, Any], path: Path) -> None:
    """Save tracklet evaluation metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_tracklet_eval_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Save tracklet evaluation metrics as compact CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _mean(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: List[Any]) -> Any:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _percentile(values: List[Any], percentile: int) -> Any:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _count_by(tracklets: List[LocalTracklet], field: str) -> Dict[str, int]:
    counts = {}
    for tracklet in tracklets:
        key = str(getattr(tracklet, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_valid_by(tracklets: List[LocalTracklet], field: str) -> Dict[str, int]:
    counts = {}
    for tracklet in tracklets:
        if not tracklet.is_valid_for_mtmc:
            continue
        key = str(getattr(tracklet, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _mean_length_by(tracklets: List[LocalTracklet], field: str) -> Dict[str, Any]:
    values = {}
    for tracklet in tracklets:
        key = str(getattr(tracklet, field))
        if key not in values:
            values[key] = []
        values[key].append(int(tracklet.length))
    return {key: _mean(items) for key, items in values.items()}
