"""Estimate per-class displacement priors from train/validation ground truth."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import (
    CLASS_NAMES,
    subset_entries,
    velocity_priors_root,
)
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.data.ground_truth import load_ground_truth_json


def estimate_velocity_priors(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Estimate world displacement per frame by class on train/validation only."""
    section = config.get("velocity_priors", {})
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    samples = {name: [] for name in CLASS_NAMES.values()}
    diagnostics = []
    scenes = [item for item in subset_entries(config, include_test=False) if item[1] in ("train", "val")]
    for subset, split, scene_name in progress_iter(scenes, progress, "estimate velocity priors"):
        path = dataset_root / split / scene_name / "ground_truth.json"
        if not path.exists():
            diagnostics.append({"subset": subset, "scene_name": scene_name, "status": "missing_gt", "samples": 0})
            continue
        objects = load_ground_truth_json(path)
        grouped = {}
        for obj in objects:
            grouped.setdefault((obj.object_type, obj.object_id), []).append((obj.frame_id, obj.location_3d))
        scene_samples = 0
        for (class_name, _object_id), values in grouped.items():
            if class_name not in samples:
                continue
            ordered = sorted(values, key=lambda item: item[0])
            for index in range(1, len(ordered)):
                frame_a, point_a = ordered[index - 1]
                frame_b, point_b = ordered[index]
                gap = int(frame_b) - int(frame_a)
                if gap <= 0:
                    continue
                speed = float(np.linalg.norm(np.asarray(point_b) - np.asarray(point_a))) / float(gap)
                if np.isfinite(speed):
                    samples[class_name].append(speed)
                    scene_samples += 1
        diagnostics.append({"subset": subset, "scene_name": scene_name, "status": "ok", "samples": scene_samples})

    percentile = float(section.get("percentile_for_vmax", 95))
    min_samples = int(section.get("min_samples_per_class", 100))
    margin_scale = float(section.get("margin_scale", 1.0))
    fallbacks = section.get("fallback_velocity_priors", {})
    rows = []
    priors = {}
    for class_id, class_name in CLASS_NAMES.items():
        values = samples.get(class_name, [])
        fallback = fallbacks.get(class_name, {}) if isinstance(fallbacks, dict) else {}
        robust = len(values) >= min_samples
        p50 = _percentile(values, 50)
        p75 = _percentile(values, 75)
        p90 = _percentile(values, 90)
        p95 = _percentile(values, 95)
        p99 = _percentile(values, 99)
        estimated_vmax = _percentile(values, percentile)
        fallback_vmax = float(fallback.get("v_max", 3.0))
        recommended_vmax = (
            max(float(estimated_vmax), fallback_vmax)
            if robust and estimated_vmax is not None
            else fallback_vmax
        )
        estimated_margin = None if p75 is None else max(float(p75) * margin_scale, 0.25)
        fallback_margin = float(fallback.get("margin", 1.5))
        recommended_margin = (
            max(float(estimated_margin), fallback_margin)
            if robust and estimated_margin is not None
            else fallback_margin
        )
        absolute_cap = float(fallback.get("absolute_cap", max(recommended_vmax * 4.0, 12.0)))
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "num_samples": len(values),
            "speed_p50": p50,
            "speed_p75": p75,
            "speed_p90": p90,
            "speed_p95": p95,
            "speed_p99": p99,
            "recommended_v_max": recommended_vmax,
            "recommended_margin": recommended_margin,
            "absolute_cap": absolute_cap,
            "source": "gt_train_val_with_config_floor" if robust else "config_fallback",
            "confidence": "high" if len(values) >= 1000 else ("medium" if robust else "low"),
        }
        rows.append(row)
        priors[class_name] = dict(row)
    root = velocity_priors_root(config)
    write_csv(root / "class_velocity_priors.csv", rows)
    write_json(root / "class_velocity_priors.json", {"classes": priors, "units": "world_distance_per_frame"})
    write_csv(root / "velocity_prior_diagnostics.csv", diagnostics)
    return {"classes": priors, "rows": rows, "diagnostics": diagnostics}


def load_or_estimate_velocity_priors(
    config: Dict[str, Any],
    progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Load existing priors unless a fresh estimate is requested."""
    path = velocity_priors_root(config) / "class_velocity_priors.json"
    if path.exists() and not overwrite:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"classes": {}}
    return estimate_velocity_priors(config, progress=progress)


def _percentile(values: List[float], value: float) -> Any:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))
