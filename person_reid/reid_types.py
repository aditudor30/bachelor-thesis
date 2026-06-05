"""Dataclasses for Person ReID crop and embedding records."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PersonCropRecord:
    """Metadata for one extracted Person crop."""

    crop_id: str
    subset: str
    split: str
    scene_name: str
    camera_id: str
    frame_id: int
    local_track_id: Optional[int]
    global_track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    crop_path: str
    matched_gt_object_id: Optional[int]
    source_csv: str


@dataclass
class PersonEmbeddingRecord:
    """Embedding record for a crop or aggregated fragment."""

    embedding_id: str
    level: str
    subset: str
    split: str
    scene_name: str
    camera_id: str
    frame_id: Optional[int]
    local_track_id: Optional[int]
    global_track_id: Optional[int]
    class_id: int
    class_name: str
    embedding: np.ndarray
    embedding_dim: int
    backend: str
    num_crops: int
    crop_ids: List[str]
    frame_ids: List[int]
    mean_confidence: Optional[float]
    matched_gt_object_id: Optional[int]
    notes: str

