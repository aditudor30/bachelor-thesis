"""I/O helpers for ReID embedding records."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from deep_oc_sort_3d.reid.reid_types import ReIDEmbeddingRecord, reid_embedding_from_dict, reid_embedding_to_dict


def write_reid_embeddings_jsonl(
    records: List[Any],
    path: Union[str, Path],
    include_embedding_in_jsonl: bool = True,
) -> None:
    """Write ReID embedding records to JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        lines.append(json.dumps(reid_embedding_to_dict(record, include_embedding=include_embedding_in_jsonl), sort_keys=True))
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_reid_embeddings_jsonl(path: Union[str, Path]) -> List[ReIDEmbeddingRecord]:
    """Read ReID embeddings from JSONL."""
    records = []
    input_path = Path(path)
    if not input_path.exists():
        return records
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(reid_embedding_from_dict(json.loads(line)))
    return records


def write_reid_embeddings_npy(
    records: List[Any],
    npy_path: Union[str, Path],
    metadata_path: Union[str, Path],
) -> None:
    """Write embeddings as an NPY matrix plus metadata CSV."""
    npy_output = Path(npy_path)
    metadata_output = Path(metadata_path)
    npy_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    if records:
        matrix = np.vstack([np.asarray(getattr(record, "embedding"), dtype=float).reshape(1, -1) for record in records])
    else:
        matrix = np.zeros((0, 0), dtype=float)
    np.save(str(npy_output), matrix)
    fields = [
        "embedding_id",
        "subset",
        "split",
        "scene_name",
        "camera_id",
        "frame_id",
        "local_track_id",
        "global_track_id",
        "candidate_id",
        "class_id",
        "class_name",
        "embedding_dim",
        "backend",
        "num_crops",
        "crop_frame_ids",
        "mean_confidence",
        "notes",
    ]
    with metadata_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = reid_embedding_to_dict(record, include_embedding=False)
            row["crop_frame_ids"] = json.dumps(row.get("crop_frame_ids", []))
            writer.writerow({field: row.get(field, "") for field in fields})


def read_reid_embeddings_npy(
    npy_path: Union[str, Path],
    metadata_path: Union[str, Path],
) -> List[ReIDEmbeddingRecord]:
    """Read an NPY embedding matrix plus metadata CSV."""
    matrix = np.load(str(npy_path))
    metadata = []
    with Path(metadata_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata.append(dict(row))
    records = []
    for index, row in enumerate(metadata):
        embedding = matrix[index].reshape(-1) if index < matrix.shape[0] else np.asarray([], dtype=float)
        data = dict(row)
        data["embedding"] = [float(item) for item in embedding]
        data["crop_frame_ids"] = _json_list(row.get("crop_frame_ids"))
        records.append(reid_embedding_from_dict(data))
    return records


def write_reid_summary_csv(rows: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Write summary rows as CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set([key for row in rows for key in row.keys()])) if rows else ["name", "value"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reid_summary_json(summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a summary dictionary as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_list(value: Any) -> List[int]:
    if value in (None, ""):
        return []
    try:
        data = json.loads(str(value))
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [int(item) for item in data]

