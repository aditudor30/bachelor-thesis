"""Motion-filtering fragmentation audit."""

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
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import add_scope_counts, length_distribution, percentile


def audit_motion_filtering(
    motion_root: Path,
    output_path: Path,
    diagnostics_root: Path,
    run_name: str,
    thresholds: FragmentationThresholds,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit motion-clean candidate files and motion metrics."""
    files = find_data_files(motion_root)
    lengths = []
    max_steps = []
    jump_ratios = []
    worst = []
    output = {
        "run_name": run_name,
        "root": str(motion_root),
        "files": len(files),
        "missing_root": not motion_root.exists(),
        "total_candidates": 0,
        "motion_good": 0,
        "motion_suspicious": 0,
        "motion_invalid": 0,
        "motion_unknown": 0,
        "motion_clean": 0,
        "motion_invalid_ratio": None,
        "motion_quality_flags": {},
        "reject_reasons": {},
        "per_subset": {},
        "per_scene": {},
        "per_camera": {},
        "per_class": {},
    }
    for path in progress_iter(files, show_progress, "%s motion filtering" % run_name):
        for row in iter_table_rows(path):
            if not _looks_like_motion_row(row):
                continue
            output["total_candidates"] += 1
            length = safe_int(row.get("length"))
            lengths.append(length)
            flag = str(row.get("motion_quality_flag") or "motion_unknown")
            add_count(output["motion_quality_flags"], flag)
            if flag in output:
                output[flag] += 1
            if safe_bool(row.get("is_motion_clean")):
                output["motion_clean"] += 1
            reason = row.get("motion_reject_reason") or row.get("reject_reason") or "ok"
            add_count(output["reject_reasons"], reason)
            add_scope_counts(output, row)
            max_step = safe_float(row.get("max_step_distance_3d"))
            if max_step is not None:
                max_steps.append(max_step)
                if len(worst) < 1000:
                    worst.append(_motion_diag_row(row, max_step))
            jump_ratio = safe_float(row.get("jump_ratio"))
            if jump_ratio is not None:
                jump_ratios.append(jump_ratio)
    output.update(length_distribution(lengths, thresholds))
    output["motion_invalid_ratio"] = rate(output["motion_invalid"], output["total_candidates"])
    output["motion_clean_ratio"] = rate(output["motion_clean"], output["total_candidates"])
    output["max_step_distance_3d_mean"] = _mean(max_steps)
    output["max_step_distance_3d_p95"] = percentile(max_steps, 95.0)
    output["max_step_distance_3d_p99"] = percentile(max_steps, 99.0)
    output["max_step_distance_3d_max"] = max(max_steps) if max_steps else None
    output["jump_ratio_mean"] = _mean(jump_ratios)
    output["jump_ratio_p95"] = percentile(jump_ratios, 95.0)
    worst = sorted(worst, key=lambda item: float(item.get("max_step_distance_3d") or 0.0), reverse=True)
    write_json(output, output_path)
    write_csv(worst[:500], diagnostics_root / "%s_worst_motion_outliers.csv" % run_name)
    return output


def _looks_like_motion_row(row: Dict[str, Any]) -> bool:
    return "motion_quality_flag" in row or "max_step_distance_3d" in row or "is_motion_clean" in row


def _motion_diag_row(row: Dict[str, Any], max_step: float) -> Dict[str, Any]:
    return {
        "subset": row.get("subset") or row.get("split"),
        "scene_name": row.get("scene_name"),
        "camera_id": row.get("camera_id"),
        "class_name": row.get("class_name"),
        "candidate_id": row.get("candidate_id"),
        "local_track_id": row.get("local_track_id"),
        "length": row.get("length"),
        "max_step_distance_3d": max_step,
        "motion_quality_flag": row.get("motion_quality_flag"),
        "motion_reject_reason": row.get("motion_reject_reason"),
        "jump_ratio": row.get("jump_ratio"),
    }


def _mean(values: List[float]) -> Any:
    if not values:
        return None
    return sum(values) / float(len(values))

