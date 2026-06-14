"""Prediction-only convention transforms for the Step 23A sweep."""

import math
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_failure_audit.box3d_utils import normalize_yaw
from deep_oc_sort_3d.official_failure_audit.failure_audit_config import internal_to_official
from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row


INDIVIDUAL_HYPOTHESES = {
    "axis": [
        "original", "swap_y_z", "swap_x_y", "swap_x_z", "flip_x", "flip_y", "flip_z",
        "flip_x_y", "flip_x_z", "flip_y_z", "scale_xyz_0_01", "scale_xyz_0_1",
        "scale_xyz_10", "scale_xyz_100",
    ],
    "center": [
        "center_original", "z_plus_height_half", "z_minus_height_half", "z_plus_height",
        "z_minus_height", "y_plus_height_half", "y_minus_height_half",
    ],
    "dimension": ["w_l_h_original", "l_w_h_swap", "w_h_l", "h_l_w", "l_h_w", "h_w_l"],
    "yaw": [
        "yaw_original", "yaw_neg", "yaw_plus_pi_over_2", "yaw_minus_pi_over_2", "yaw_plus_pi",
        "yaw_degrees_to_radians", "yaw_radians_to_degrees", "yaw_zero",
    ],
    "frame": ["frame_original", "frame_plus_1", "frame_minus_1"],
    "class": ["official_mapping", "internal_mapping_no_remap", "swap_official_2_6"],
}


def individual_definitions(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = config.get("hypothesis_sweep", {})
    enabled = {
        "axis": bool(rules.get("run_axis_hypotheses", True)),
        "center": bool(rules.get("run_center_hypotheses", True)),
        "dimension": bool(rules.get("run_dimension_order_hypotheses", True)),
        "yaw": bool(rules.get("run_yaw_hypotheses", True)),
        "frame": bool(rules.get("run_frame_hypotheses", True)),
        "class": bool(rules.get("run_class_hypotheses", True)),
    }
    output = []
    for category, names in INDIVIDUAL_HYPOTHESES.items():
        if not enabled[category]:
            continue
        for name in names:
            output.append({"name": "%s:%s" % (category, name), "category": category, "operations": {category: name}})
    return output


def transform_rows(
    rows: Sequence[AuditTrack1Row], operations: Dict[str, str], config: Dict[str, Any],
) -> List[AuditTrack1Row]:
    return [transform_row(row, operations, config) for row in rows]


def transform_row(
    row: AuditTrack1Row, operations: Dict[str, str], config: Dict[str, Any],
) -> AuditTrack1Row:
    class_id = _class_id(row, operations.get("class", "official_mapping"), config)
    frame_id = _frame_id(row.frame_id, operations.get("frame", "frame_original"))
    x, y, z = _axis(row.x, row.y, row.z, operations.get("axis", "original"))
    width, length, height = _dimensions(
        row.width, row.length, row.height, operations.get("dimension", "w_l_h_original"),
    )
    scale = _axis_scale(operations.get("axis", "original"))
    if scale != 1.0:
        width, length, height = width * scale, length * scale, height * scale
    x, y, z = _center(x, y, z, height, operations.get("center", "center_original"))
    yaw = _yaw(row.yaw, operations.get("yaw", "yaw_original"))
    return row.clone(
        class_id=class_id, frame_id=frame_id, x=x, y=y, z=z,
        width=width, length=length, height=height, yaw=yaw,
    )


def _axis(x: float, y: float, z: float, name: str) -> Tuple[float, float, float]:
    if name == "swap_y_z":
        return x, z, y
    if name == "swap_x_y":
        return y, x, z
    if name == "swap_x_z":
        return z, y, x
    if name == "flip_x":
        return -x, y, z
    if name == "flip_y":
        return x, -y, z
    if name == "flip_z":
        return x, y, -z
    if name == "flip_x_y":
        return -x, -y, z
    if name == "flip_x_z":
        return -x, y, -z
    if name == "flip_y_z":
        return x, -y, -z
    scale = _axis_scale(name)
    return x * scale, y * scale, z * scale


def _axis_scale(name: str) -> float:
    return {
        "scale_xyz_0_01": 0.01, "scale_xyz_0_1": 0.1,
        "scale_xyz_10": 10.0, "scale_xyz_100": 100.0,
    }.get(name, 1.0)


def _center(x: float, y: float, z: float, height: float, name: str) -> Tuple[float, float, float]:
    if name == "z_plus_height_half":
        z += height / 2.0
    elif name == "z_minus_height_half":
        z -= height / 2.0
    elif name == "z_plus_height":
        z += height
    elif name == "z_minus_height":
        z -= height
    elif name == "y_plus_height_half":
        y += height / 2.0
    elif name == "y_minus_height_half":
        y -= height / 2.0
    return x, y, z


def _dimensions(width: float, length: float, height: float, name: str) -> Tuple[float, float, float]:
    values = {
        "w_l_h_original": (width, length, height), "l_w_h_swap": (length, width, height),
        "w_h_l": (width, height, length), "h_l_w": (height, length, width),
        "l_h_w": (length, height, width), "h_w_l": (height, width, length),
    }
    return values.get(name, (width, length, height))


def _yaw(value: float, name: str) -> float:
    if name == "yaw_neg":
        value = -value
    elif name == "yaw_plus_pi_over_2":
        value += math.pi / 2.0
    elif name == "yaw_minus_pi_over_2":
        value -= math.pi / 2.0
    elif name == "yaw_plus_pi":
        value += math.pi
    elif name == "yaw_degrees_to_radians":
        value = math.radians(value)
    elif name == "yaw_radians_to_degrees":
        value = math.degrees(value)
    elif name == "yaw_zero":
        value = 0.0
    return normalize_yaw(value)


def _frame_id(value: int, name: str) -> int:
    if name == "frame_plus_1":
        return value + 1
    if name == "frame_minus_1":
        return value - 1
    return value


def _class_id(row: AuditTrack1Row, name: str, config: Dict[str, Any]) -> int:
    raw = row.raw_class_id if row.raw_class_id is not None else row.class_id
    mapped = internal_to_official(config).get(raw, raw) if row.source_class_space == "internal" else row.class_id
    if name == "internal_mapping_no_remap":
        return raw
    if name == "swap_official_2_6":
        if mapped == 2:
            return 6
        if mapped == 6:
            return 2
    return mapped
