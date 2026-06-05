"""Read/write helpers for Person ReID embedding records."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.person_reid.reid_types import PersonEmbeddingRecord


EMBEDDING_METADATA_FIELDS = [
    "embedding_id",
    "level",
    "subset",
    "split",
    "scene_name",
    "camera_id",
    "frame_id",
    "local_track_id",
    "global_track_id",
    "class_id",
    "class_name",
    "embedding_dim",
    "backend",
    "num_crops",
    "crop_ids",
    "frame_ids",
    "mean_confidence",
    "matched_gt_object_id",
    "notes",
]


def embedding_record_to_dict(record: PersonEmbeddingRecord, include_embedding: bool = True) -> Dict[str, Any]:
    """Serialize embedding record."""
    data = {
        "embedding_id": record.embedding_id,
        "level": record.level,
        "subset": record.subset,
        "split": record.split,
        "scene_name": record.scene_name,
        "camera_id": record.camera_id,
        "frame_id": record.frame_id,
        "local_track_id": record.local_track_id,
        "global_track_id": record.global_track_id,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "embedding_dim": record.embedding_dim,
        "backend": record.backend,
        "num_crops": record.num_crops,
        "crop_ids": list(record.crop_ids),
        "frame_ids": list(record.frame_ids),
        "mean_confidence": record.mean_confidence,
        "matched_gt_object_id": record.matched_gt_object_id,
        "notes": record.notes,
    }
    if include_embedding:
        data["embedding"] = [float(item) for item in np.asarray(record.embedding, dtype=float).reshape(-1)]
    return data


def embedding_record_from_dict(data: Dict[str, Any]) -> PersonEmbeddingRecord:
    """Deserialize embedding record."""
    embedding = np.asarray(data.get("embedding", []), dtype=float).reshape(-1)
    return PersonEmbeddingRecord(
        embedding_id=str(data.get("embedding_id", "")),
        level=str(data.get("level", "")),
        subset=str(data.get("subset", "")),
        split=str(data.get("split", "")),
        scene_name=str(data.get("scene_name", "")),
        camera_id=str(data.get("camera_id", "")),
        frame_id=_optional_int(data.get("frame_id")),
        local_track_id=_optional_int(data.get("local_track_id")),
        global_track_id=_optional_int(data.get("global_track_id")),
        class_id=int(float(data.get("class_id", -1))),
        class_name=str(data.get("class_name", "")),
        embedding=embedding,
        embedding_dim=int(data.get("embedding_dim", int(embedding.size))),
        backend=str(data.get("backend", "")),
        num_crops=int(data.get("num_crops", 0)),
        crop_ids=[str(item) for item in _list_value(data.get("crop_ids"))],
        frame_ids=[int(float(item)) for item in _list_value(data.get("frame_ids"))],
        mean_confidence=_optional_float(data.get("mean_confidence")),
        matched_gt_object_id=_optional_int(data.get("matched_gt_object_id")),
        notes=str(data.get("notes", "")),
    )


def write_embeddings_jsonl(records: Iterable[PersonEmbeddingRecord], path: Path, include_embedding: bool = True) -> int:
    """Write embedding records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(embedding_record_to_dict(record, include_embedding=include_embedding), sort_keys=True) + "\n")
            count += 1
    return count


def iter_embeddings_jsonl(path: Path) -> Iterable[PersonEmbeddingRecord]:
    """Iterate embedding records from JSONL."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield embedding_record_from_dict(json.loads(line))


def read_embeddings_jsonl(path: Path) -> List[PersonEmbeddingRecord]:
    """Read all embedding records from JSONL."""
    return [record for record in iter_embeddings_jsonl(path)]


def write_embeddings_npy(records: List[PersonEmbeddingRecord], npy_path: Path, metadata_path: Path) -> None:
    """Write embeddings as npy matrix plus metadata CSV."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        matrix = np.vstack([np.asarray(record.embedding, dtype=float).reshape(1, -1) for record in records])
    else:
        matrix = np.zeros((0, 0), dtype=float)
    np.save(str(npy_path), matrix)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EMBEDDING_METADATA_FIELDS)
        writer.writeheader()
        for record in records:
            row = embedding_record_to_dict(record, include_embedding=False)
            row["crop_ids"] = json.dumps(row.get("crop_ids", []))
            row["frame_ids"] = json.dumps(row.get("frame_ids", []))
            writer.writerow({field: row.get(field, "") for field in EMBEDDING_METADATA_FIELDS})


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


def _list_value(value: Any) -> List[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []

