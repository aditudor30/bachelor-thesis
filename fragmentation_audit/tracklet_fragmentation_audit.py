"""Local tracklet fragmentation audit."""

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


def audit_tracklets(
    tracklets_root: Path,
    output_path: Path,
    diagnostics_root: Path,
    run_name: str,
    thresholds: FragmentationThresholds,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit compact LocalTracklet files."""
    files = find_data_files(tracklets_root)
    lengths = []
    rows_out = []
    output = _empty_summary(run_name, tracklets_root, len(files), not tracklets_root.exists())
    for path in progress_iter(files, show_progress, "%s tracklets" % run_name):
        for row in iter_table_rows(path):
            length = safe_int(row.get("length"))
            lengths.append(length)
            output["total_tracklets"] += 1
            if safe_bool(row.get("is_valid_for_mtmc")):
                output["valid_tracklets"] += 1
            add_count(output["quality_flags"], row.get("quality_flag", "unknown"))
            add_scope_counts(output, row)
            if length <= thresholds.short_track_length:
                rows_out.append(_compact_tracklet_row(row, length))
    output.update(length_distribution(lengths, thresholds))
    output["valid_ratio"] = rate(output["valid_tracklets"], output["total_tracklets"])
    output["mean_gt_purity"] = _mean_gt_purity(files, show_progress)
    write_json(output, output_path)
    write_csv(rows_out[:5000], diagnostics_root / "%s_short_tracklets.csv" % run_name)
    return output


def _empty_summary(run_name: str, root: Path, files: int, missing_root: bool) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "root": str(root),
        "files": files,
        "missing_root": missing_root,
        "total_tracklets": 0,
        "valid_tracklets": 0,
        "quality_flags": {},
        "per_subset": {},
        "per_scene": {},
        "per_camera": {},
        "per_class": {},
    }


def _compact_tracklet_row(row: Dict[str, Any], length: int) -> Dict[str, Any]:
    return {
        "subset": row.get("subset") or row.get("split"),
        "scene_name": row.get("scene_name"),
        "camera_id": row.get("camera_id"),
        "class_name": row.get("class_name"),
        "local_track_id": row.get("local_track_id"),
        "length": length,
        "quality_flag": row.get("quality_flag"),
        "is_valid_for_mtmc": row.get("is_valid_for_mtmc"),
        "gt_purity": row.get("gt_purity"),
    }


def _mean_gt_purity(files: List[Path], show_progress: bool) -> Any:
    values = []
    for path in progress_iter(files, show_progress, "tracklet purity scan"):
        for row in iter_table_rows(path):
            value = safe_float(row.get("gt_purity"))
            if value is not None:
                values.append(value)
    if not values:
        return None
    return sum(values) / float(len(values))

