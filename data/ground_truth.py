"""Ground-truth JSON parsing for SmartSpaces MTMC scenes."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class GroundTruthObject:
    """One annotated 3D object for a frame."""

    frame_id: int
    object_type: str
    object_id: int
    location_3d: np.ndarray
    bbox3d_scale: np.ndarray
    bbox3d_rotation: np.ndarray
    visible_bboxes_2d: Dict[str, Tuple[float, float, float, float]]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_vector3(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=float)
        except (TypeError, ValueError):
            pass
    return np.zeros(3, dtype=float)


def _to_visible_bboxes(value: Any) -> Dict[str, Tuple[float, float, float, float]]:
    visible = {}
    if not isinstance(value, dict):
        return visible
    for camera_id, bbox in value.items():
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            visible[str(camera_id)] = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            continue
    return visible


def parse_ground_truth_json_dict(data: Dict[str, Any]) -> List[GroundTruthObject]:
    """Parse ground-truth JSON data into a flat list of objects.

    Missing object fields are filled with conservative defaults so lightweight
    inspection can continue on partially malformed annotation files.
    """
    objects = []
    for frame_key in sorted(data.keys(), key=lambda item: _to_int(item, 0)):
        frame_objects = data.get(frame_key)
        if not isinstance(frame_objects, list):
            continue
        frame_id = _to_int(frame_key, 0)
        for raw_object in frame_objects:
            if not isinstance(raw_object, dict):
                continue
            obj = GroundTruthObject(
                frame_id=frame_id,
                object_type=str(_get_first(raw_object, ["object_type", "object type"], "")),
                object_id=_to_int(_get_first(raw_object, ["object_id", "object id"], None), -1),
                location_3d=_to_vector3(_get_first(raw_object, ["3d_location", "3d location"], None)),
                bbox3d_scale=_to_vector3(
                    _get_first(raw_object, ["3d_bounding_box_scale", "3d bounding box scale"], None)
                ),
                bbox3d_rotation=_to_vector3(
                    _get_first(raw_object, ["3d_bounding_box_rotation", "3d bounding box rotation"], None)
                ),
                visible_bboxes_2d=_to_visible_bboxes(
                    _get_first(raw_object, ["2d_bounding_box_visible", "2d bounding box visible"], {})
                ),
            )
            objects.append(obj)
    return objects


def _get_first(data: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def load_ground_truth_json(path: Path) -> List[GroundTruthObject]:
    """Load and parse a ground-truth JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return []
    return parse_ground_truth_json_dict(data)
