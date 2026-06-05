"""MTMC candidate fragmentation audit."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    add_count,
    find_data_files,
    iter_table_rows,
    progress_iter,
    rate,
    safe_bool,
    safe_float,
    safe_int,
    write_csv,
    write_json,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import add_scope_counts, length_distribution


def audit_candidates(
    candidates_root: Path,
    output_path: Path,
    diagnostics_root: Path,
    run_name: str,
    thresholds: FragmentationThresholds,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit MTMCTrackletCandidate files."""
    files = find_data_files(candidates_root)
    lengths = []
    rejected_rows = []
    output = {
        "run_name": run_name,
        "root": str(candidates_root),
        "files": len(files),
        "missing_root": not candidates_root.exists(),
        "total_candidates": 0,
        "kept_candidates": 0,
        "rejected_candidates": 0,
        "has_3d": 0,
        "reject_reasons": {},
        "quality_flags": {},
        "per_subset": {},
        "per_scene": {},
        "per_camera": {},
        "per_class": {},
        "mean_gt_purity": None,
    }
    purity_values = []
    for path in progress_iter(files, show_progress, "%s candidates" % run_name):
        for row in iter_table_rows(path):
            output["total_candidates"] += 1
            length = safe_int(row.get("length"))
            lengths.append(length)
            if safe_bool(row.get("is_candidate")):
                output["kept_candidates"] += 1
            else:
                output["rejected_candidates"] += 1
                if len(rejected_rows) < 5000:
                    rejected_rows.append(_candidate_diag_row(row))
            if safe_bool(row.get("has_3d")):
                output["has_3d"] += 1
            add_count(output["reject_reasons"], row.get("reject_reason", "ok"))
            add_count(output["quality_flags"], row.get("quality_flag", "unknown"))
            add_scope_counts(output, row)
            purity = safe_float(row.get("gt_purity"))
            if purity is not None:
                purity_values.append(purity)
    output.update(length_distribution(lengths, thresholds))
    output["kept_ratio"] = rate(output["kept_candidates"], output["total_candidates"])
    output["rejected_ratio"] = rate(output["rejected_candidates"], output["total_candidates"])
    output["has_3d_rate"] = rate(output["has_3d"], output["total_candidates"])
    if purity_values:
        output["mean_gt_purity"] = sum(purity_values) / float(len(purity_values))
    write_json(output, output_path)
    write_csv(rejected_rows, diagnostics_root / "%s_rejected_candidates.csv" % run_name)
    return output


def _candidate_diag_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subset": row.get("subset") or row.get("split"),
        "scene_name": row.get("scene_name"),
        "camera_id": row.get("camera_id"),
        "class_name": row.get("class_name"),
        "candidate_id": row.get("candidate_id"),
        "local_track_id": row.get("local_track_id"),
        "length": row.get("length"),
        "reject_reason": row.get("reject_reason"),
        "quality_flag": row.get("quality_flag"),
    }

