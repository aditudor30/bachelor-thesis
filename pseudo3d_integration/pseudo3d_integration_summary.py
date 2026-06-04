"""Summaries for baseline_v2 pseudo-3D observation integration."""

from typing import Any, Dict, List


def summarize_integrated_observations(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize integrated Observation3D dictionaries."""
    total = len(rows)
    pseudo_used = sum(1 for row in rows if row.get("pseudo3d_used"))
    fallback_original = sum(1 for row in rows if row.get("fallback_original_used"))
    class_prior_dims = sum(1 for row in rows if row.get("class_prior_dimensions_used"))
    no_3d = sum(1 for row in rows if not row.get("has_3d"))
    metadata_complete = _metadata_completeness(rows)
    return {
        "output_observations": total,
        "pseudo3d_matched": sum(1 for row in rows if row.get("pseudo3d_matched")),
        "pseudo3d_missing": sum(1 for row in rows if not row.get("pseudo3d_matched")),
        "pseudo3d_used": pseudo_used,
        "pseudo3d_used_rate": float(pseudo_used) / float(total) if total else None,
        "fallback_original_used": fallback_original,
        "class_prior_dimensions_used": class_prior_dims,
        "no_3d_records": no_3d,
        "source_metadata_completeness": metadata_complete,
        "per_class": _count(rows, "class_id"),
        "per_source": _count(rows, "center_3d_source"),
    }


def _metadata_completeness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    fields = [
        "center_3d_source",
        "dimensions_3d_source",
        "yaw_source",
        "depth_source",
        "pseudo3d_method",
        "pseudo3d_version",
    ]
    summary = {"total": total}
    for field in fields:
        count = sum(1 for row in rows if row.get(field) not in (None, "", "unknown"))
        summary["%s_complete" % field] = count
        summary["%s_complete_rate" % field] = float(count) / float(total) if total else None
    estimated = sum(1 for row in rows if row.get("is_estimated_for_test"))
    summary["is_estimated_for_test_set"] = estimated
    summary["is_estimated_for_test_rate"] = float(estimated) / float(total) if total else None
    return summary


def _count(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts

