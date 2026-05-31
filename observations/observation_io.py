"""JSONL I/O for Observation3D records."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.observations.observation_types import Observation3D


def observation_to_dict(obs: Observation3D) -> Dict[str, Any]:
    """Convert an Observation3D to a JSON-serializable dictionary."""
    return {
        "scene_id": obs.scene_id,
        "scene_name": obs.scene_name,
        "split": obs.split,
        "camera_id": obs.camera_id,
        "frame_id": obs.frame_id,
        "detection_id": obs.detection_id,
        "class_id": obs.class_id,
        "class_name": obs.class_name,
        "confidence": obs.confidence,
        "bbox_xyxy": list(obs.bbox_xyxy),
        "bbox_xywh": list(obs.bbox_xywh),
        "center_3d": _array_to_list(obs.center_3d),
        "dimensions_3d": _array_to_list(obs.dimensions_3d),
        "yaw": obs.yaw,
        "object_id": obs.object_id,
        "matched_gt": obs.matched_gt,
        "matched_iou": obs.matched_iou,
        "depth_value": obs.depth_value,
        "depth_sampling_method": obs.depth_sampling_method,
        "source": obs.source,
    }


def observation_from_dict(data: Dict[str, Any]) -> Observation3D:
    """Create an Observation3D from a dictionary."""
    return Observation3D(
        scene_id=int(data["scene_id"]),
        scene_name=str(data["scene_name"]),
        split=str(data["split"]),
        camera_id=str(data["camera_id"]),
        frame_id=int(data["frame_id"]),
        detection_id=int(data["detection_id"]),
        class_id=int(data["class_id"]),
        class_name=str(data["class_name"]),
        confidence=float(data["confidence"]),
        bbox_xyxy=tuple(float(value) for value in data["bbox_xyxy"]),
        bbox_xywh=tuple(float(value) for value in data["bbox_xywh"]),
        center_3d=_list_to_array(data.get("center_3d")),
        dimensions_3d=_list_to_array(data.get("dimensions_3d")),
        yaw=None if data.get("yaw") is None else float(data["yaw"]),
        object_id=None if data.get("object_id") is None else int(data["object_id"]),
        matched_gt=bool(data["matched_gt"]),
        matched_iou=None if data.get("matched_iou") is None else float(data["matched_iou"]),
        depth_value=None if data.get("depth_value") is None else float(data["depth_value"]),
        depth_sampling_method=data.get("depth_sampling_method"),
        source=str(data["source"]),
    )


def write_observations_jsonl(observations: List[Observation3D], path: Path) -> None:
    """Write observations to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(observation_to_dict(obs), sort_keys=True) for obs in observations]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_observations_jsonl(path: Path) -> List[Observation3D]:
    """Read observations from JSONL."""
    observations = []
    if not path.exists():
        return observations
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        observations.append(observation_from_dict(json.loads(line)))
    return observations


def _array_to_list(value: Optional[np.ndarray]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _list_to_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=float)

