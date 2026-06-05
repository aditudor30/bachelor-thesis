"""Observation-level fragmentation and provenance audit."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    add_count,
    find_data_files,
    iter_table_rows,
    progress_iter,
    rate,
    read_json,
    safe_bool,
    write_json,
)
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import add_scope_counts


def audit_observations(pipeline_root: Path, output_path: Path, run_name: str, show_progress: bool = True) -> Dict[str, Any]:
    """Audit Observation3D files under one pipeline root."""
    observations_root = pipeline_root / "observations3d"
    files = find_data_files(observations_root)
    summary_file = pipeline_root / "summaries" / "pseudo3d_observation_summary.json"
    existing_summary = read_json(summary_file)
    output = {
        "run_name": run_name,
        "pipeline_root": str(pipeline_root),
        "observations_root": str(observations_root),
        "files": len(files),
        "missing_root": not observations_root.exists(),
        "total_observations": 0,
        "has_3d": 0,
        "pseudo3d_used": 0,
        "fallback_original_used": 0,
        "no_3d_records": 0,
        "center_3d_source_distribution": {},
        "dimensions_3d_source_distribution": {},
        "yaw_source_distribution": {},
        "depth_source_distribution": {},
        "source_distribution": {},
        "per_subset": {},
        "per_scene": {},
        "per_camera": {},
        "per_class": {},
        "existing_summary": existing_summary,
    }
    for path in progress_iter(files, show_progress, "%s observations" % run_name):
        for row in iter_table_rows(path):
            output["total_observations"] += 1
            add_scope_counts(output, row)
            if safe_bool(row.get("has_3d")) or _has_center(row):
                output["has_3d"] += 1
            else:
                output["no_3d_records"] += 1
            if safe_bool(row.get("pseudo3d_used")):
                output["pseudo3d_used"] += 1
            if safe_bool(row.get("fallback_original_used")):
                output["fallback_original_used"] += 1
            add_count(output["center_3d_source_distribution"], row.get("center_3d_source", "unknown"))
            add_count(output["dimensions_3d_source_distribution"], row.get("dimensions_3d_source", "unknown"))
            add_count(output["yaw_source_distribution"], row.get("yaw_source", "unknown"))
            add_count(output["depth_source_distribution"], row.get("depth_source", "unknown"))
            add_count(output["source_distribution"], row.get("source", "unknown"))
    total = output["total_observations"]
    if total == 0 and existing_summary:
        total = int(existing_summary.get("output_observations", 0) or 0)
        output["total_observations"] = total
        output["pseudo3d_used"] = int(existing_summary.get("pseudo3d_used", 0) or 0)
        output["fallback_original_used"] = int(existing_summary.get("fallback_original_used", 0) or 0)
        output["no_3d_records"] = int(existing_summary.get("no_3d_records", 0) or 0)
    output["has_3d_rate"] = rate(output["has_3d"], total)
    output["pseudo3d_used_rate"] = existing_summary.get("pseudo3d_used_rate", rate(output["pseudo3d_used"], total))
    output["fallback_original_used_rate"] = existing_summary.get(
        "fallback_original_used_rate",
        rate(output["fallback_original_used"], total),
    )
    output["no_3d_rate"] = rate(output["no_3d_records"], total)
    write_json(output, output_path)
    return output


def _has_center(row: Dict[str, Any]) -> bool:
    if row.get("center_3d") not in (None, ""):
        return True
    return row.get("center_x") not in (None, "") and row.get("center_y") not in (None, "") and row.get("center_z") not in (None, "")

