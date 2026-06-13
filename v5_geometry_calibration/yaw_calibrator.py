"""Circular class-level yaw bias calibration."""

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import angle_delta


def fit_yaw_biases(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Fit bounded circular median GT-pred yaw bias for allowed classes."""
    rules = config.get("yaw_calibration", {})
    allowed = set(int(value) for value in rules.get("apply_to_classes", [1, 2, 3, 6]))
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        try:
            class_id = int(float(row["official_class_id"]))
            if class_id in allowed:
                grouped[str(class_id)].append(angle_delta(float(row["pred_yaw"]), float(row["gt_yaw"])))
        except (KeyError, TypeError, ValueError):
            continue
    minimum = int(rules.get("min_samples_per_class", 50))
    cap = float(rules.get("max_yaw_bias_rad", 0.35))
    output: Dict[str, Any] = {}
    for class_id, values in sorted(grouped.items(), key=lambda item: int(item[0])):
        raw = circular_median(values)
        output[class_id] = {
            "samples": len(values), "raw_bias_rad": raw,
            "bias_rad": max(-cap, min(cap, raw)),
            "eligible_by_sample_count": len(values) >= minimum, "selected": False,
        }
    return output


def circular_median(values: Sequence[float]) -> float:
    """Return the observed angle minimizing circular L1 distance."""
    candidates = [float((value + math.pi) % (2.0 * math.pi) - math.pi) for value in values]
    if not candidates:
        return 0.0
    return min(candidates, key=lambda candidate: sum(abs(angle_delta(candidate, value)) for value in candidates))
