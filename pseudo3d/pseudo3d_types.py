"""Dataclasses for the future pseudo-3D estimator API."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Pseudo3DInput:
    """Input contract for one future pseudo-3D estimate."""

    scene_name: str
    camera_id: str
    frame_id: int
    class_id: int
    class_name: str
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    image_width: int
    image_height: int
    calibration: Dict[str, Any]
    track_id: Optional[int] = None


@dataclass
class Pseudo3DOutput:
    """Output contract for one future pseudo-3D estimate."""

    center_3d: Optional[np.ndarray]
    dimensions_3d: Optional[np.ndarray]
    yaw: Optional[float]
    depth: Optional[float]
    confidence_3d: float
    center_3d_source: str
    dimensions_3d_source: str
    yaw_source: str
    depth_source: str
    is_gt_derived: bool
    is_estimated_for_test: bool
    pseudo3d_method: str
    pseudo3d_version: str
    failure_reason: Optional[str] = None


@dataclass
class Pseudo3DPriors:
    """Class-wise dimension prior used by the future estimator."""

    class_id: int
    class_name: str
    width: float
    length: float
    height: float
    confidence_level: str
    source: str


def pseudo3d_input_to_dict(item: Pseudo3DInput) -> Dict[str, Any]:
    """Serialize Pseudo3DInput to a plain dictionary."""
    return {
        "scene_name": item.scene_name,
        "camera_id": item.camera_id,
        "frame_id": item.frame_id,
        "class_id": item.class_id,
        "class_name": item.class_name,
        "bbox_xyxy": list(item.bbox_xyxy),
        "confidence": item.confidence,
        "image_width": item.image_width,
        "image_height": item.image_height,
        "calibration": item.calibration,
        "track_id": item.track_id,
    }


def pseudo3d_output_to_dict(item: Pseudo3DOutput) -> Dict[str, Any]:
    """Serialize Pseudo3DOutput to a plain dictionary."""
    return {
        "center_3d": _array_to_list(item.center_3d),
        "dimensions_3d": _array_to_list(item.dimensions_3d),
        "yaw": item.yaw,
        "depth": item.depth,
        "confidence_3d": item.confidence_3d,
        "center_3d_source": item.center_3d_source,
        "dimensions_3d_source": item.dimensions_3d_source,
        "yaw_source": item.yaw_source,
        "depth_source": item.depth_source,
        "is_gt_derived": item.is_gt_derived,
        "is_estimated_for_test": item.is_estimated_for_test,
        "pseudo3d_method": item.pseudo3d_method,
        "pseudo3d_version": item.pseudo3d_version,
        "failure_reason": item.failure_reason,
    }


def pseudo3d_priors_to_dict(item: Pseudo3DPriors) -> Dict[str, Any]:
    """Serialize Pseudo3DPriors to a plain dictionary."""
    return {
        "class_id": item.class_id,
        "class_name": item.class_name,
        "width": item.width,
        "length": item.length,
        "height": item.height,
        "confidence_level": item.confidence_level,
        "source": item.source,
    }


def _array_to_list(value: Optional[np.ndarray]) -> Optional[Any]:
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1).tolist()

