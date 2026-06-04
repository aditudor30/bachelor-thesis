"""Depth outlier detection and stabilization for pseudo-3D tracks."""

from typing import Any, Dict, Tuple

import numpy as np

from deep_oc_sort_3d.pseudo3d.temporal_smoothing import median_filter_sequence


def detect_depth_outliers(
    depths: np.ndarray,
    method: str = "median_mad",
    threshold: float = 3.5,
) -> Dict[str, Any]:
    """Detect depth outliers using a robust median/MAD score."""
    values = np.asarray(depths, dtype=float).reshape(-1)
    valid = values[np.isfinite(values)]
    if not valid.size:
        return {"method": method, "threshold": threshold, "outlier_indices": [], "num_outliers": 0, "median_depth": None}
    median = float(np.median(valid))
    if method != "median_mad":
        return {"method": method, "threshold": threshold, "outlier_indices": [], "num_outliers": 0, "median_depth": median}
    deviations = np.abs(valid - median)
    mad = float(np.median(deviations))
    outliers = []
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        score = 0.0 if mad == 0.0 else 0.6745 * abs(float(value) - median) / mad
        if score > float(threshold):
            outliers.append(index)
    return {
        "method": method,
        "threshold": threshold,
        "outlier_indices": outliers,
        "num_outliers": len(outliers),
        "median_depth": median,
        "mad": mad,
    }


def stabilize_depth_sequence(
    depths: np.ndarray,
    method: str = "median_filter",
    window: int = 5,
    max_relative_change: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Smooth depth and clamp implausible relative changes to track median."""
    values = np.asarray(depths, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy(), {"num_clamped": 0, "changed_indices": []}
    smoothed = median_filter_sequence(values, window) if method == "median_filter" else values.copy()
    valid = smoothed[np.isfinite(smoothed)]
    if not valid.size:
        return smoothed, {"num_clamped": 0, "changed_indices": [], "median_depth": None}
    median = float(np.median(valid))
    lower = median * max(0.0, 1.0 - float(max_relative_change))
    upper = median * (1.0 + float(max_relative_change))
    changed = []
    for index, value in enumerate(smoothed):
        if not np.isfinite(value) or median <= 0.0:
            continue
        clamped = min(max(float(value), lower), upper)
        if abs(clamped - float(value)) > 1e-9:
            smoothed[index] = clamped
            changed.append(index)
    outlier_report = detect_depth_outliers(values)
    return smoothed, {
        "method": method,
        "window": window,
        "max_relative_change": max_relative_change,
        "median_depth": median,
        "lower_depth": lower,
        "upper_depth": upper,
        "num_clamped": len(changed),
        "changed_indices": changed,
        "raw_depth_outliers": outlier_report,
    }
