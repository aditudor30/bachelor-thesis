"""Build fine-tuned fragment embeddings from crop embeddings."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.person_reid.reid_embedding_io import embedding_record_to_dict, write_embeddings_jsonl
from deep_oc_sort_3d.person_reid.reid_types import PersonEmbeddingRecord
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import (
    count_by,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
)


FRAGMENT_INDEX_FIELDS = [
    "fragment_embedding_id",
    "subset",
    "split",
    "scene_name",
    "scene_id",
    "class_id",
    "class_name",
    "camera_id_optional",
    "local_track_id_optional",
    "tracklet_id_optional",
    "candidate_id_optional",
    "global_track_id",
    "num_crop_embeddings",
    "num_valid_crop_embeddings",
    "frame_start",
    "frame_end",
    "cameras",
    "mean_confidence",
    "matched_gt_object_id",
    "embedding_index",
    "valid_embedding",
    "invalid_reason",
]


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized vector, preserving zero vectors."""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def mean_pool_l2(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    """Mean-pool vectors and L2-normalize the result."""
    valid = [np.asarray(vector, dtype=np.float32).reshape(1, -1) for vector in vectors if vector is not None]
    if not valid:
        return None
    matrix = np.vstack(valid)
    return l2_normalize_vector(np.mean(matrix, axis=0))


def fragment_key_from_crop(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Return stable fragment key from crop metadata."""
    return (
        str(row.get("subset", "")),
        str(row.get("scene_name", "")),
        str(row.get("class_id", "")),
        str(row.get("global_track_id", "")),
    )


def aggregate_crop_embeddings_to_fragments(
    crop_embeddings: np.ndarray,
    crop_rows: List[Dict[str, Any]],
    backend: str = "torchreid_osnet_finetuned",
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[PersonEmbeddingRecord]]:
    """Aggregate crop embeddings into final-export global-fragment embeddings."""
    groups: Dict[Tuple[str, str, str, str], List[int]] = {}
    for index, row in enumerate(crop_rows):
        if str(row.get("valid_embedding", "1")) not in ("1", "true", "True"):
            continue
        key = fragment_key_from_crop(row)
        if key[3] in ("", "None", "nan"):
            continue
        groups.setdefault(key, []).append(index)
    matrix_rows: List[np.ndarray] = []
    index_rows: List[Dict[str, Any]] = []
    jsonl_records: List[PersonEmbeddingRecord] = []
    for key in sorted(groups.keys(), key=lambda item: str(item)):
        indices = groups[key]
        vector = mean_pool_l2([crop_embeddings[index] for index in indices])
        if vector is None:
            continue
        embedding_index = len(matrix_rows)
        rows = [crop_rows[index] for index in indices]
        meta = fragment_index_row(key, rows, embedding_index, valid=True, invalid_reason="")
        matrix_rows.append(vector)
        index_rows.append(meta)
        jsonl_records.append(fragment_jsonl_record(meta, vector, rows, backend))
    matrix = np.vstack([row.reshape(1, -1) for row in matrix_rows]).astype(np.float32) if matrix_rows else np.zeros((0, 0), dtype=np.float32)
    return matrix, index_rows, jsonl_records


def write_fragment_embedding_outputs(
    matrix: np.ndarray,
    index_rows: List[Dict[str, Any]],
    jsonl_records: List[PersonEmbeddingRecord],
    output_root: Path,
) -> Dict[str, Any]:
    """Write Step 18C fragment embedding artifacts."""
    embeddings_dir = Path(output_root) / "embeddings"
    fragment_dir = embeddings_dir / "fragment_embeddings"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    npy_path = embeddings_dir / "finetuned_fragment_embeddings.npy"
    index_path = embeddings_dir / "finetuned_fragment_embeddings_index.csv"
    np.save(str(npy_path), matrix.astype(np.float32))
    write_csv_rows(index_rows, index_path, FRAGMENT_INDEX_FIELDS)
    jsonl_path = fragment_dir / "person_global_fragment_embeddings.jsonl"
    write_embeddings_jsonl(jsonl_records, jsonl_path, include_embedding=True)
    metadata_path = embeddings_dir / "fragment_embeddings_metadata.csv"
    write_csv_rows(index_rows, metadata_path, FRAGMENT_INDEX_FIELDS)
    summary = summarize_fragment_embeddings(index_rows, matrix)
    summary.update(
        {
            "npy_path": str(npy_path),
            "index_path": str(index_path),
            "jsonl_path": str(jsonl_path),
            "metadata_path": str(metadata_path),
        }
    )
    write_json(summary, embeddings_dir / "fragment_embedding_summary.json")
    return summary


def summarize_fragment_embeddings(index_rows: List[Dict[str, Any]], matrix: np.ndarray) -> Dict[str, Any]:
    """Summarize aggregated fragment embeddings."""
    valid = [row for row in index_rows if str(row.get("valid_embedding", "")) in ("1", "true", "True")]
    return {
        "num_fragments": len(index_rows),
        "num_valid_fragments": len(valid),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] > 0 else None,
        "per_subset": count_by(valid, "subset"),
        "per_scene": count_by(valid, "scene_name"),
        "per_class": count_by(valid, "class_name"),
        "mean_num_crop_embeddings": _mean([row.get("num_crop_embeddings") for row in valid]),
    }


def fragment_index_row(key: Tuple[str, str, str, str], rows: List[Dict[str, Any]], embedding_index: int, valid: bool, invalid_reason: str) -> Dict[str, Any]:
    """Create one fragment index row."""
    first = rows[0] if rows else {}
    frames = sorted([safe_int(row.get("frame_id"), None) for row in rows if safe_int(row.get("frame_id"), None) is not None])
    cameras = sorted(set([str(row.get("camera_id", "")) for row in rows if str(row.get("camera_id", ""))]))
    confidences = [safe_float(row.get("confidence"), None) for row in rows]
    confidences = [value for value in confidences if value is not None]
    gt_counts: Dict[str, int] = {}
    for row in rows:
        gt = row.get("matched_gt_object_id")
        if gt in (None, ""):
            continue
        gt_key = str(gt)
        gt_counts[gt_key] = gt_counts.get(gt_key, 0) + 1
    matched_gt = _majority_key(gt_counts)
    return {
        "fragment_embedding_id": "global_fragment__%s__%s__%s" % (key[0], key[1], key[3]),
        "subset": key[0],
        "split": str(first.get("split", "")),
        "scene_name": key[1],
        "scene_id": str(first.get("scene_id", "")),
        "class_id": key[2],
        "class_name": str(first.get("class_name", "Person")),
        "camera_id_optional": "",
        "local_track_id_optional": "",
        "tracklet_id_optional": str(first.get("tracklet_id", "")),
        "candidate_id_optional": str(first.get("candidate_id", "")),
        "global_track_id": key[3],
        "num_crop_embeddings": len(rows),
        "num_valid_crop_embeddings": len(rows),
        "frame_start": frames[0] if frames else "",
        "frame_end": frames[-1] if frames else "",
        "cameras": ";".join(cameras),
        "mean_confidence": float(sum(confidences)) / float(len(confidences)) if confidences else "",
        "matched_gt_object_id": "" if matched_gt is None else matched_gt,
        "embedding_index": embedding_index,
        "valid_embedding": "1" if valid else "0",
        "invalid_reason": invalid_reason,
    }


def fragment_jsonl_record(meta: Dict[str, Any], embedding: np.ndarray, crop_rows: List[Dict[str, Any]], backend: str) -> PersonEmbeddingRecord:
    """Create a compatibility JSONL embedding record."""
    frame_ids = sorted([safe_int(row.get("frame_id"), None) for row in crop_rows if safe_int(row.get("frame_id"), None) is not None])
    crop_ids = [str(row.get("crop_embedding_id", row.get("crop_id", ""))) for row in crop_rows]
    matched_gt = safe_int(meta.get("matched_gt_object_id"), None)
    return PersonEmbeddingRecord(
        embedding_id=str(meta.get("fragment_embedding_id", "")),
        level="global_fragment",
        subset=str(meta.get("subset", "")),
        split=str(meta.get("split", "")),
        scene_name=str(meta.get("scene_name", "")),
        camera_id="",
        frame_id=None,
        local_track_id=None,
        global_track_id=safe_int(meta.get("global_track_id"), None),
        class_id=safe_int(meta.get("class_id"), 0) or 0,
        class_name=str(meta.get("class_name", "Person")),
        embedding=l2_normalize_vector(embedding),
        embedding_dim=int(np.asarray(embedding).size),
        backend=backend,
        num_crops=len(crop_rows),
        crop_ids=crop_ids,
        frame_ids=frame_ids,
        mean_confidence=safe_float(meta.get("mean_confidence"), None),
        matched_gt_object_id=matched_gt,
        notes="finetuned_osnet_mean_pool",
    )


def write_crop_embedding_jsonl(crop_rows: List[Dict[str, Any]], embeddings: np.ndarray, output_path: Path) -> None:
    """Write crop embeddings as JSONL for debugging and portability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(crop_rows):
            payload = dict(row)
            payload["embedding"] = [float(value) for value in embeddings[index].reshape(-1)]
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _majority_key(counts: Dict[str, int]) -> Optional[str]:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _mean(values: List[Any]) -> Optional[float]:
    numeric = [safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))
