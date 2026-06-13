"""Apply selected class-level calibration to immutable official Track1 rows."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import clone_row, normalize_angle


def apply_corrections_to_track1(
    rows: Sequence[OfficialTrack1Row], corrections: Dict[str, Any],
) -> Tuple[List[OfficialTrack1Row], Dict[str, Any]]:
    """Apply only selected dimension, center and yaw corrections by official class."""
    output: List[OfficialTrack1Row] = []
    counts = {"dimension": 0, "center": 0, "depth": 0, "yaw": 0}
    per_class: Dict[str, Dict[str, int]] = {}
    for row in rows:
        class_id = str(int(row.class_id))
        changes: Dict[str, float] = {}
        dimension_item = corrections.get("dimension", {}).get(class_id, {})
        center_item = corrections.get("center", {}).get(class_id, {})
        yaw_item = corrections.get("yaw", {}).get(class_id, {})
        if dimension_item.get("selected"):
            scale = np.asarray(dimension_item.get("scale", [1.0, 1.0, 1.0]), dtype=float)
            values = np.asarray([row.width, row.length, row.height], dtype=float) * scale
            if np.all(np.isfinite(values)) and np.all(values > 0.0):
                changes.update(width=float(values[0]), length=float(values[1]), height=float(values[2]))
                counts["dimension"] += 1
                _increment(per_class, class_id, "dimension")
        if center_item.get("selected"):
            bias = np.asarray(center_item.get("bias", [0.0, 0.0, 0.0]), dtype=float)
            values = np.asarray([row.x, row.y, row.z], dtype=float) + bias
            if np.all(np.isfinite(values)):
                changes.update(x=float(values[0]), y=float(values[1]), z=float(values[2]))
                counts["center"] += 1
                _increment(per_class, class_id, "center")
        if yaw_item.get("selected"):
            changes["yaw"] = normalize_angle(float(row.yaw) + float(yaw_item.get("bias_rad", 0.0)))
            counts["yaw"] += 1
            _increment(per_class, class_id, "yaw")
        output.append(clone_row(row, **changes) if changes else row)
    summary = {
        "input_rows": len(rows), "output_rows": len(output), "applied_rows_by_component": counts,
        "applied_rows_by_class_component": per_class,
        "depth_application_status": "not_applied_due_to_missing_camera_mapping",
    }
    return output, summary


def _increment(values: Dict[str, Dict[str, int]], class_id: str, component: str) -> None:
    class_counts = values.setdefault(class_id, {})
    class_counts[component] = class_counts.get(component, 0) + 1
