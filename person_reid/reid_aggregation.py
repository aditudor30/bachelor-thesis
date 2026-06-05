"""Compute and aggregate Person ReID embeddings."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.person_reid.crop_extraction import crop_record_from_row
from deep_oc_sort_3d.person_reid.reid_backends import build_person_reid_backend
from deep_oc_sort_3d.person_reid.reid_embedding_io import (
    embedding_record_to_dict,
    iter_embeddings_jsonl,
    write_embeddings_jsonl,
    write_embeddings_npy,
)
from deep_oc_sort_3d.person_reid.reid_types import PersonCropRecord, PersonEmbeddingRecord
from deep_oc_sort_3d.person_reid.reid_utils import count_by, l2_normalize, progress_iter, read_csv_rows, write_json


def compute_person_crop_embeddings_from_config(config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Compute ReID embeddings for extracted Person crops."""
    root = Path(str(config.get("reid_person", {}).get("output_root", "output/reid_person/baseline_v2_pseudo3d_fullcam")))
    crop_metadata = root / "crops" / "person_crops.csv"
    output_jsonl = root / "embeddings_crop" / "person_crop_embeddings.jsonl"
    summary_path = root / "summaries" / "crop_embedding_summary.json"
    if output_jsonl.exists() and not overwrite:
        summary = {"status": "skipped_existing", "embedding_jsonl": str(output_jsonl)}
        write_json(summary, summary_path)
        return summary
    backend_result = build_person_reid_backend(config.get("backend", {}))
    if not backend_result.available or backend_result.backend is None:
        summary = {
            "status": "backend_unavailable",
            "backend": backend_result.backend_name,
            "message": backend_result.message,
            "weights_loaded": backend_result.weights_loaded,
            "embedding_jsonl": str(output_jsonl),
        }
        write_json(summary, summary_path)
        return summary
    rows, _fields = read_csv_rows(crop_metadata)
    batch_size = int(config.get("backend", {}).get("batch_size", 128))
    records_for_npy: List[PersonEmbeddingRecord] = []
    save_npy = bool(config.get("embeddings", {}).get("also_save_npy", True))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    missing_crop_count = 0
    invalid_crop_count = 0
    embeddings_generated = 0
    norm_values: List[float] = []
    meta_rows: List[Dict[str, Any]] = []
    batch_crops: List[np.ndarray] = []
    batch_records: List[PersonCropRecord] = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in progress_iter(rows, show_progress, "compute Person ReID crop embeddings", "crop"):
            crop_record = crop_record_from_row(row)
            crop = _read_crop(crop_record.crop_path)
            if crop is None:
                missing_crop_count += 1
                continue
            if crop.size == 0:
                invalid_crop_count += 1
                continue
            batch_crops.append(crop)
            batch_records.append(crop_record)
            if len(batch_crops) >= batch_size:
                batch_output = _embed_batch(batch_crops, batch_records, backend_result.backend, backend_result.backend_name)
                embeddings_generated += _write_embedding_batch(handle, batch_output, records_for_npy if save_npy else None, norm_values, meta_rows)
                batch_crops = []
                batch_records = []
        if batch_crops:
            batch_output = _embed_batch(batch_crops, batch_records, backend_result.backend, backend_result.backend_name)
            embeddings_generated += _write_embedding_batch(handle, batch_output, records_for_npy if save_npy else None, norm_values, meta_rows)
    if save_npy:
        write_embeddings_npy(records_for_npy, root / "embeddings_crop" / "person_crop_embeddings.npy", root / "embeddings_crop" / "person_crop_embeddings.metadata.csv")
    summary = {
        "status": "ok",
        "backend": backend_result.backend_name,
        "weights_loaded": backend_result.weights_loaded,
        "embedding_dim": backend_result.embedding_dim,
        "crop_rows": len(rows),
        "embeddings_generated": embeddings_generated,
        "missing_crop_count": missing_crop_count,
        "invalid_crop_count": invalid_crop_count,
        "embedding_norm_mean": float(np.mean(norm_values)) if norm_values else None,
        "embedding_norm_std": float(np.std(norm_values)) if norm_values else None,
        "per_subset": count_by(meta_rows, "subset"),
        "per_scene": count_by(meta_rows, "scene_name"),
        "per_camera": count_by(meta_rows, "camera_id"),
        "embedding_jsonl": str(output_jsonl),
    }
    write_json(summary, summary_path)
    return summary


def aggregate_person_embeddings_from_config(config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Aggregate crop embeddings into local-track and global-fragment embeddings."""
    root = Path(str(config.get("reid_person", {}).get("output_root", "output/reid_person/baseline_v2_pseudo3d_fullcam")))
    crop_jsonl = root / "embeddings_crop" / "person_crop_embeddings.jsonl"
    if not crop_jsonl.exists():
        summary = {"status": "missing_crop_embeddings", "crop_jsonl": str(crop_jsonl)}
        write_json(summary, root / "summaries" / "aggregation_summary.json")
        return summary
    records = list(iter_embeddings_jsonl(crop_jsonl))
    aggregate_method = str(config.get("embeddings", {}).get("aggregate_method", "mean"))
    local_records = _aggregate_records(records, "local_track", _local_key, aggregate_method)
    global_records = _aggregate_records(records, "global_fragment", _global_key, aggregate_method)
    local_jsonl = root / "embeddings_track" / "person_local_track_embeddings.jsonl"
    global_jsonl = root / "embeddings_global_fragment" / "person_global_fragment_embeddings.jsonl"
    write_embeddings_jsonl(local_records, local_jsonl, include_embedding=True)
    write_embeddings_jsonl(global_records, global_jsonl, include_embedding=True)
    if bool(config.get("embeddings", {}).get("also_save_npy", True)):
        write_embeddings_npy(local_records, root / "embeddings_track" / "person_local_track_embeddings.npy", root / "embeddings_track" / "person_local_track_embeddings.metadata.csv")
        write_embeddings_npy(global_records, root / "embeddings_global_fragment" / "person_global_fragment_embeddings.npy", root / "embeddings_global_fragment" / "person_global_fragment_embeddings.metadata.csv")
    summary = {
        "status": "ok",
        "crop_embeddings": len(records),
        "local_track_embeddings": len(local_records),
        "global_fragment_embeddings": len(global_records),
        "aggregate_method": aggregate_method,
        "local_track_jsonl": str(local_jsonl),
        "global_fragment_jsonl": str(global_jsonl),
    }
    write_json(summary, root / "summaries" / "aggregation_summary.json")
    return summary


def aggregate_embeddings(vectors: List[np.ndarray], method: str = "mean") -> Optional[np.ndarray]:
    """Aggregate vectors and L2-normalize."""
    if not vectors:
        return None
    matrix = np.vstack([np.asarray(vector, dtype=float).reshape(1, -1) for vector in vectors])
    if method == "median":
        vector = np.median(matrix, axis=0)
    else:
        vector = np.mean(matrix, axis=0)
    return l2_normalize(vector)


def _embed_batch(crops: List[np.ndarray], crop_records: List[PersonCropRecord], backend: Any, backend_name: str) -> List[PersonEmbeddingRecord]:
    embeddings = backend.extract_batch(crops)
    records = []
    for index, crop_record in enumerate(crop_records):
        embedding = embeddings[index].reshape(-1)
        records.append(
            PersonEmbeddingRecord(
                embedding_id=crop_record.crop_id,
                level="crop",
                subset=crop_record.subset,
                split=crop_record.split,
                scene_name=crop_record.scene_name,
                camera_id=crop_record.camera_id,
                frame_id=crop_record.frame_id,
                local_track_id=crop_record.local_track_id,
                global_track_id=crop_record.global_track_id,
                class_id=crop_record.class_id,
                class_name=crop_record.class_name,
                embedding=l2_normalize(embedding),
                embedding_dim=int(embedding.size),
                backend=backend_name,
                num_crops=1,
                crop_ids=[crop_record.crop_id],
                frame_ids=[crop_record.frame_id],
                mean_confidence=crop_record.confidence,
                matched_gt_object_id=crop_record.matched_gt_object_id,
                notes="",
            )
        )
    return records


def _write_embedding_batch(
    handle: Any,
    records: List[PersonEmbeddingRecord],
    records_for_npy: Optional[List[PersonEmbeddingRecord]],
    norm_values: List[float],
    meta_rows: List[Dict[str, Any]],
) -> int:
    count = 0
    for record in records:
        handle.write(json.dumps(embedding_record_to_dict(record, include_embedding=True), sort_keys=True) + "\n")
        if records_for_npy is not None:
            records_for_npy.append(record)
        norm_values.append(float(np.linalg.norm(record.embedding)))
        meta_rows.append(_record_meta(record))
        count += 1
    return count


def _aggregate_records(
    records: List[PersonEmbeddingRecord],
    level: str,
    key_fn: Any,
    method: str,
) -> List[PersonEmbeddingRecord]:
    groups: Dict[Tuple[Any, ...], List[PersonEmbeddingRecord]] = {}
    for record in records:
        groups.setdefault(key_fn(record), []).append(record)
    output = []
    for key, group in progress_iter(sorted(groups.items(), key=lambda item: str(item[0])), False, "aggregate ReID", "track"):
        first = group[0]
        embedding = aggregate_embeddings([record.embedding for record in group], method=method)
        if embedding is None:
            continue
        gt_ids = [record.matched_gt_object_id for record in group if record.matched_gt_object_id is not None]
        output.append(
            PersonEmbeddingRecord(
                embedding_id="%s__%s" % (level, "__".join([str(item) for item in key])),
                level=level,
                subset=first.subset,
                split=first.split,
                scene_name=first.scene_name,
                camera_id=first.camera_id if level == "local_track" else "",
                frame_id=None,
                local_track_id=first.local_track_id if level == "local_track" else None,
                global_track_id=first.global_track_id,
                class_id=first.class_id,
                class_name=first.class_name,
                embedding=embedding,
                embedding_dim=int(embedding.size),
                backend=first.backend,
                num_crops=len(group),
                crop_ids=[record.embedding_id for record in group],
                frame_ids=sorted([frame for record in group for frame in record.frame_ids]),
                mean_confidence=float(np.mean([record.mean_confidence or 0.0 for record in group])),
                matched_gt_object_id=_majority_gt(gt_ids),
                notes="aggregated_%s" % method,
            )
        )
    return output


def _local_key(record: PersonEmbeddingRecord) -> Tuple[Any, ...]:
    return (record.subset, record.scene_name, record.camera_id, record.local_track_id, record.global_track_id)


def _global_key(record: PersonEmbeddingRecord) -> Tuple[Any, ...]:
    return (record.subset, record.scene_name, record.global_track_id)


def _majority_gt(values: List[Optional[int]]) -> Optional[int]:
    if not values:
        return None
    counts: Dict[int, int] = {}
    for value in values:
        if value is None:
            continue
        counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]


def _read_crop(path: str) -> Optional[np.ndarray]:
    if path in (None, ""):
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _record_meta(record: PersonEmbeddingRecord) -> Dict[str, Any]:
    return {"subset": record.subset, "scene_name": record.scene_name, "camera_id": record.camera_id}
