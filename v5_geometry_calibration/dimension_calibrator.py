"""Robust class-level dimension scale calibration."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np


def fit_dimension_scales(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Fit clipped median GT/pred dimension ratios per official class."""
    rules = config.get("dimension_calibration", {})
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in rows:
        try:
            pred = np.asarray([row["pred_width"], row["pred_length"], row["pred_height"]], dtype=float)
            gt = np.asarray([row["gt_width"], row["gt_length"], row["gt_height"]], dtype=float)
            if np.all(pred > 0.0) and np.all(gt > 0.0):
                grouped[str(int(float(row["official_class_id"])))].append(gt / pred)
        except (KeyError, TypeError, ValueError):
            continue
    minimum = int(rules.get("min_samples_per_class", 50))
    lower = float(rules.get("scale_min", 0.85))
    upper = float(rules.get("scale_max", 1.15))
    output: Dict[str, Any] = {}
    for class_id, values in sorted(grouped.items(), key=lambda item: int(item[0])):
        array = np.asarray(values, dtype=float)
        raw = np.median(array, axis=0)
        output[class_id] = {
            "samples": len(values), "raw_scale": raw.tolist(),
            "scale": np.clip(raw, lower, upper).tolist(),
            "eligible_by_sample_count": len(values) >= minimum, "selected": False,
        }
    return output
