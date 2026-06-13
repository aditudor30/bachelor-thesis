"""Diagnostic distance/depth scale calibration."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np


def fit_depth_scales(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate class-level GT-distance/pred-distance ratios for diagnostics."""
    rules = config.get("depth_scale_calibration", {})
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        try:
            predicted = float(row["pred_distance"])
            target = float(row["gt_distance"])
            if predicted > 1e-6 and target > 0.0:
                grouped[str(int(float(row["official_class_id"])))].append(target / predicted)
        except (KeyError, TypeError, ValueError):
            continue
    minimum = int(rules.get("min_samples_per_bin", 100))
    lower = float(rules.get("scale_min", 0.90))
    upper = float(rules.get("scale_max", 1.10))
    output: Dict[str, Any] = {}
    for class_id, values in sorted(grouped.items(), key=lambda item: int(item[0])):
        raw = float(np.median(values))
        output[class_id] = {
            "samples": len(values), "raw_scale": raw, "scale": float(np.clip(raw, lower, upper)),
            "eligible_by_sample_count": len(values) >= minimum, "selected": False,
            "application_status": "not_applied_due_to_missing_camera_mapping",
        }
    return output
