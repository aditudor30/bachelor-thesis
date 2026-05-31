"""Depth quality diagnostics and unit heuristics."""

from typing import Any, Dict, Optional

import numpy as np


def summarize_depth_array(depth: np.ndarray, sample_name: str = "") -> Dict[str, Any]:
    """Summarize one loaded depth frame without assuming its units."""
    arr = np.asarray(depth)
    total = int(arr.size)
    flat = arr.astype(float).reshape(-1)
    finite = np.isfinite(flat)
    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)
    zero_mask = flat == 0.0
    valid_mask = finite & (flat > 0.0)
    valid = flat[valid_mask]

    stats = {
        "sample_name": sample_name,
        "shape": tuple(int(dim) for dim in arr.shape),
        "dtype": str(arr.dtype),
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "percentiles": {
            1: None,
            5: None,
            25: None,
            50: None,
            75: None,
            95: None,
            99: None,
        },
        "num_nan": int(np.sum(nan_mask)),
        "num_inf": int(np.sum(inf_mask)),
        "num_zero": int(np.sum(zero_mask)),
        "valid_count": int(valid.size),
        "valid_ratio": 0.0 if total == 0 else float(valid.size) / float(total),
    }

    if valid.size == 0:
        return stats

    stats["min"] = float(np.min(valid))
    stats["max"] = float(np.max(valid))
    stats["mean"] = float(np.mean(valid))
    stats["median"] = float(np.median(valid))
    for percentile in (1, 5, 25, 50, 75, 95, 99):
        stats["percentiles"][percentile] = float(np.percentile(valid, percentile))
    return stats


def guess_depth_unit(depth_stats: Dict[str, Any]) -> str:
    """Guess whether depth values are meters, millimeters, or unknown."""
    median = depth_stats.get("median")
    if median is None:
        return "unknown"
    median_value = float(median)
    if median_value > 100.0:
        return "millimeters_likely"
    if 0.1 <= median_value <= 100.0:
        return "meters_likely"
    return "unknown"


def convert_depth_units(depth: np.ndarray, unit: str = "auto") -> np.ndarray:
    """Convert depth array to meters when units are likely millimeters."""
    arr = np.asarray(depth, dtype=float)
    selected_unit = unit
    if unit == "auto":
        selected_unit = guess_depth_unit(summarize_depth_array(arr))
    if selected_unit in ("millimeters", "millimeter", "mm", "millimeters_likely"):
        return arr / 1000.0
    return arr.copy()


def compare_backprojection_to_gt(
    backprojected_center: Optional[np.ndarray],
    gt_center: Optional[np.ndarray],
) -> Optional[float]:
    """Return Euclidean distance between backprojected and GT centers."""
    if backprojected_center is None or gt_center is None:
        return None
    backprojected = np.asarray(backprojected_center, dtype=float).reshape(3)
    gt = np.asarray(gt_center, dtype=float).reshape(3)
    return float(np.linalg.norm(backprojected - gt))

