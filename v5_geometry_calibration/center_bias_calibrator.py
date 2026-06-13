"""Robust conservative class-level center bias calibration."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np


def fit_center_biases(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Fit bounded median GT-pred xyz biases per official class."""
    rules = config.get("center_bias_calibration", {})
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in rows:
        try:
            pred = np.asarray([row["pred_x"], row["pred_y"], row["pred_z"]], dtype=float)
            gt = np.asarray([row["gt_x"], row["gt_y"], row["gt_z"]], dtype=float)
            grouped[str(int(float(row["official_class_id"])))].append(gt - pred)
        except (KeyError, TypeError, ValueError):
            continue
    minimum = int(rules.get("min_samples_per_class", 100))
    output: Dict[str, Any] = {}
    for class_id, values in sorted(grouped.items(), key=lambda item: int(item[0])):
        array = np.asarray(values, dtype=float)
        raw = np.median(array, axis=0)
        cap = float(rules.get("max_bias_m_person", 0.50)) if class_id == "0" else float(rules.get("max_bias_m_default", 1.00))
        spread_cap = float(rules.get("max_residual_mad_m_person", 2.00)) if class_id == "0" else float(rules.get("max_residual_mad_m_default", 3.00))
        norm = float(np.linalg.norm(raw))
        residual_mad = float(np.median(np.linalg.norm(array - raw, axis=1)))
        bounded = raw if norm <= cap or norm <= 0.0 else raw * (cap / norm)
        output[class_id] = {
            "samples": len(values), "raw_bias": raw.tolist(), "raw_bias_norm_m": norm,
            "bias": bounded.tolist(), "bias_norm_m": float(np.linalg.norm(bounded)),
            "residual_mad_m": residual_mad, "max_residual_mad_m": spread_cap,
            "eligible_by_sample_count": len(values) >= minimum,
            "eligible_by_variability": residual_mad <= spread_cap, "selected": False,
        }
    return output
