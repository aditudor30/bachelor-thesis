"""Calibration JSON parsing for SmartSpaces MTMC scenes."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class CameraCalibration:
    """Calibration fields for a single camera/sensor."""

    camera_id: str
    coordinates: Optional[Dict[str, Any]]
    scale_factor: Optional[float]
    translation_to_global: Optional[Dict[str, Any]]
    fps: Optional[float]
    direction: Optional[float]
    direction3d: Optional[Tuple[float, float, float]]
    frame_width: Optional[int]
    frame_height: Optional[int]
    intrinsic_matrix: Optional[np.ndarray]
    extrinsic_matrix: Optional[np.ndarray]
    camera_matrix: Optional[np.ndarray]
    homography: Optional[np.ndarray]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_tuple3(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _to_ndarray(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _camera_id_from_sensor(sensor: Dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "sensor id", "sensor_id", "sensorId", "name", "camera_id", "cameraId"):
        value = sensor.get(key)
        if value is not None:
            return str(value)
    return "sensor_%03d" % fallback_index


def parse_calibration_json_dict(data: Dict[str, Any]) -> Dict[str, CameraCalibration]:
    """Parse calibration JSON data into per-camera dataclasses.

    The parser accepts the expected top-level ``calibrationType`` and ``sensors``
    structure, while tolerating missing or partial fields on each sensor.
    """
    sensors = data.get("sensors", [])
    if isinstance(sensors, dict):
        sensor_items = []
        for key, value in sensors.items():
            if isinstance(value, dict):
                sensor = dict(value)
                if "id" not in sensor:
                    sensor["id"] = key
                sensor_items.append(sensor)
    elif isinstance(sensors, list):
        sensor_items = [sensor for sensor in sensors if isinstance(sensor, dict)]
    else:
        sensor_items = []

    calibrations = {}
    for index, sensor in enumerate(sensor_items):
        camera_id = _camera_id_from_sensor(sensor, index)
        attributes = sensor.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}

        calibration = CameraCalibration(
            camera_id=camera_id,
            coordinates=sensor.get("coordinates") if isinstance(sensor.get("coordinates"), dict) else None,
            scale_factor=_to_float(sensor.get("scaleFactor")),
            translation_to_global=(
                sensor.get("translationToGlobalCoordinates")
                if isinstance(sensor.get("translationToGlobalCoordinates"), dict)
                else None
            ),
            fps=_to_float(attributes.get("fps")),
            direction=_to_float(attributes.get("direction")),
            direction3d=_to_float_tuple3(attributes.get("direction3d")),
            frame_width=_to_int(attributes.get("frameWidth")),
            frame_height=_to_int(attributes.get("frameHeight")),
            intrinsic_matrix=_to_ndarray(sensor.get("intrinsicMatrix")),
            extrinsic_matrix=_to_ndarray(sensor.get("extrinsicMatrix")),
            camera_matrix=_to_ndarray(sensor.get("cameraMatrix")),
            homography=_to_ndarray(sensor.get("homography")),
        )
        calibrations[camera_id] = calibration

    return calibrations


def load_calibration_json(path: Path) -> Dict[str, CameraCalibration]:
    """Load and parse a calibration JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return parse_calibration_json_dict(data)
