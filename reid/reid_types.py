"""Dataclasses and serialization helpers for ReID embeddings."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReIDCropSample:
    """Metadata for one RGB crop used to compute a ReID embedding."""

    subset: str
    split: str
    scene_name: str
    camera_id: str
    frame_id: int
    local_track_id: int
    global_track_id: Optional[int]
    candidate_id: Optional[str]
    class_id: int
    class_name: str
    bbox_xyxy: Tuple[float, float, float, float]
    crop_path: Optional[str]
    confidence: float
    source: str


@dataclass
class ReIDEmbeddingRecord:
    """A generic ReID embedding record."""

    embedding_id: str
    subset: str
    split: str
    scene_name: str
    camera_id: str
    frame_id: Optional[int]
    local_track_id: Optional[int]
    global_track_id: Optional[int]
    candidate_id: Optional[str]
    class_id: int
    class_name: str
    embedding: np.ndarray
    embedding_dim: int
    backend: str
    num_crops: int
    crop_frame_ids: List[int]
    mean_confidence: Optional[float]
    notes: str


@dataclass
class TrackletReIDEmbedding:
    """Aggregated ReID embedding for one local tracklet."""

    subset: str
    split: str
    scene_name: str
    camera_id: str
    local_track_id: int
    class_id: int
    class_name: str
    embedding: np.ndarray
    embedding_dim: int
    backend: str
    num_crops: int
    frame_ids: List[int]
    mean_confidence: Optional[float]
    candidate_id: Optional[str]
    global_track_id: Optional[int]


@dataclass
class CandidateReIDEmbedding:
    """Aggregated ReID embedding for one MTMC candidate."""

    subset: str
    split: str
    scene_name: str
    camera_id: str
    local_track_id: int
    candidate_id: str
    class_id: int
    class_name: str
    embedding: np.ndarray
    embedding_dim: int
    backend: str
    num_crops: int
    frame_ids: List[int]
    mean_confidence: Optional[float]
    global_track_id: Optional[int]


def normalize_embedding_l2(embedding) -> np.ndarray:
    """Return an L2-normalized embedding, preserving zero vectors."""
    arr = np.asarray(embedding, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr.copy()
    return arr / norm


def reid_embedding_to_dict(record: Any, include_embedding: bool = True) -> Dict[str, Any]:
    """Convert a ReID embedding dataclass to a JSON-friendly dictionary."""
    data = {
        "embedding_id": _field(record, "embedding_id", _default_embedding_id(record)),
        "subset": _field(record, "subset", ""),
        "split": _field(record, "split", ""),
        "scene_name": _field(record, "scene_name", ""),
        "camera_id": _field(record, "camera_id", ""),
        "frame_id": _field(record, "frame_id", None),
        "local_track_id": _field(record, "local_track_id", None),
        "global_track_id": _field(record, "global_track_id", None),
        "candidate_id": _field(record, "candidate_id", None),
        "class_id": int(_field(record, "class_id", -1)),
        "class_name": _field(record, "class_name", ""),
        "embedding_dim": int(_field(record, "embedding_dim", 0)),
        "backend": _field(record, "backend", ""),
        "num_crops": int(_field(record, "num_crops", 0)),
        "crop_frame_ids": list(_field(record, "crop_frame_ids", _field(record, "frame_ids", []))),
        "mean_confidence": _field(record, "mean_confidence", None),
        "notes": _field(record, "notes", ""),
    }
    if include_embedding:
        data["embedding"] = [float(item) for item in np.asarray(_field(record, "embedding", []), dtype=float).reshape(-1)]
    return data


def reid_embedding_from_dict(data: Dict[str, Any]) -> ReIDEmbeddingRecord:
    """Create a generic ReIDEmbeddingRecord from a dictionary."""
    embedding = np.asarray(data.get("embedding", []), dtype=float).reshape(-1)
    return ReIDEmbeddingRecord(
        embedding_id=str(data.get("embedding_id", "")),
        subset=str(data.get("subset", "")),
        split=str(data.get("split", "")),
        scene_name=str(data.get("scene_name", "")),
        camera_id=str(data.get("camera_id", "")),
        frame_id=_optional_int(data.get("frame_id")),
        local_track_id=_optional_int(data.get("local_track_id")),
        global_track_id=_optional_int(data.get("global_track_id")),
        candidate_id=_optional_str(data.get("candidate_id")),
        class_id=int(data.get("class_id", -1)),
        class_name=str(data.get("class_name", "")),
        embedding=embedding,
        embedding_dim=int(data.get("embedding_dim", int(embedding.size))),
        backend=str(data.get("backend", "")),
        num_crops=int(data.get("num_crops", 0)),
        crop_frame_ids=[int(item) for item in data.get("crop_frame_ids", [])],
        mean_confidence=_optional_float(data.get("mean_confidence")),
        notes=str(data.get("notes", "")),
    )


def _default_embedding_id(record: Any) -> str:
    candidate_id = _field(record, "candidate_id", None)
    if candidate_id not in (None, ""):
        return str(candidate_id)
    return "%s_%s_%s_%s" % (
        _field(record, "subset", ""),
        _field(record, "scene_name", ""),
        _field(record, "camera_id", ""),
        str(_field(record, "local_track_id", "")),
    )


def _field(record: Any, name: str, default: Any) -> Any:
    return getattr(record, name, default)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)

