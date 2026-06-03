"""Summary helpers for ReID embedding outputs."""

from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.reid.reid_io import write_reid_summary_csv, write_reid_summary_json


def summarize_reid_embeddings(records: List[Any]) -> Dict[str, Any]:
    """Summarize ReID embedding records."""
    dims = [int(getattr(record, "embedding_dim", 0)) for record in records]
    crops = [int(getattr(record, "num_crops", 0)) for record in records]
    embeddings_present = [record for record in records if getattr(record, "embedding", None) is not None]
    return {
        "total_records": len(records),
        "embedding_dim": max(dims) if dims else None,
        "backend": _first_non_empty([getattr(record, "backend", "") for record in records]),
        "num_with_embedding": len(embeddings_present),
        "num_without_embedding": max(0, len(records) - len(embeddings_present)),
        "per_subset": _count_by(records, "subset"),
        "per_scene": _count_by(records, "scene_name"),
        "per_camera": _count_by(records, "camera_id"),
        "per_class": _count_by(records, "class_name"),
        "mean_num_crops": float(np.mean(np.asarray(crops, dtype=float))) if crops else None,
        "missing_crop_count": len([value for value in crops if value <= 0]),
        "invalid_bbox_count": 0,
    }


def print_reid_summary(summary: Dict[str, Any]) -> None:
    """Print a ReID summary."""
    for key in [
        "total_records",
        "embedding_dim",
        "backend",
        "num_with_embedding",
        "num_without_embedding",
        "mean_num_crops",
        "missing_crop_count",
        "invalid_bbox_count",
    ]:
        print("%s: %s" % (key, str(summary.get(key))))
    for key in ["per_subset", "per_scene", "per_camera", "per_class"]:
        print("%s: %s" % (key, str(summary.get(key))))


def write_summary_json(summary: Dict[str, Any], path: Any) -> None:
    """Write summary JSON."""
    write_reid_summary_json(summary, path)


def write_summary_csv(summary: Dict[str, Any], path: Any) -> None:
    """Write flattened summary CSV."""
    rows = []
    for key, value in summary.items():
        rows.append({"name": key, "value": str(value)})
    write_reid_summary_csv(rows, path)


def _count_by(records: List[Any], field: str) -> Dict[str, int]:
    counts = {}
    for record in records:
        key = str(getattr(record, field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _first_non_empty(values: List[Any]) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None

