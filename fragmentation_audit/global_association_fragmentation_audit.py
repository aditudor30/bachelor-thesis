"""Global association fragmentation audit."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    add_count,
    find_data_files,
    iter_table_rows_progress,
    iter_table_rows,
    progress_iter,
    rate,
    read_json,
    safe_bool,
    safe_float,
    safe_int,
    safe_json,
    write_csv,
    write_json,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import add_scope_counts, length_distribution, percentile


def audit_global_association(
    global_root: Path,
    output_path: Path,
    diagnostics_root: Path,
    run_name: str,
    thresholds: FragmentationThresholds,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit global tracks and association edges."""
    track_files = _files_named(global_root, ["global_tracks"])
    edge_files = _files_named(global_root, ["association_edges", "overlap_edges", "transition_edges"])
    output = {
        "run_name": run_name,
        "root": str(global_root),
        "missing_root": not global_root.exists(),
        "global_track_files": len(track_files),
        "edge_files": len(edge_files),
        "global_tracks": 0,
        "multi_camera_tracks": 0,
        "singleton_tracks": 0,
        "single_candidate_tracks": 0,
        "single_camera_tracks": 0,
        "accepted_edges": 0,
        "rejected_edges": 0,
        "overlap_edges_accepted": 0,
        "transition_edges_accepted": 0,
        "accepted_edge_relations": {},
        "edge_reject_reasons": {},
        "per_subset": {},
        "per_scene": {},
        "per_camera": {},
        "per_class": {},
        "global_purity_mean": None,
        "false_merge_rate": None,
        "fragmentation_approx": 0,
    }
    lengths = []
    component_sizes = []
    purity_values = []
    worst_tracks = []
    for path in progress_iter(track_files, show_progress, "%s global tracks" % run_name):
        for row in iter_table_rows(path):
            output["global_tracks"] += 1
            num_candidates = safe_int(row.get("num_candidates"), 0)
            num_cameras = safe_int(row.get("num_cameras"), 0)
            component_sizes.append(num_candidates)
            lengths.append(safe_int(row.get("duration"), safe_int(row.get("length"), num_candidates)))
            if num_candidates <= 1:
                output["single_candidate_tracks"] += 1
            if num_cameras <= 1:
                output["single_camera_tracks"] += 1
            if num_cameras > 1:
                output["multi_camera_tracks"] += 1
            else:
                output["singleton_tracks"] += 1
            add_scope_counts(output, row)
            for camera_id in safe_json(row.get("camera_ids_json"), []):
                add_count(output["per_camera"], camera_id)
            purity = safe_float(row.get("gt_purity"))
            if purity is not None:
                purity_values.append(purity)
            if len(worst_tracks) < 5000:
                worst_tracks.append(_global_track_diag_row(row, num_candidates, num_cameras))
    for path in progress_iter(edge_files, show_progress, "%s global edges" % run_name):
        for row in iter_table_rows_progress(path, show_progress, "%s edge rows" % run_name):
            if safe_bool(row.get("accepted")):
                output["accepted_edges"] += 1
                relation = str(row.get("temporal_relation") or "unknown")
                add_count(output["accepted_edge_relations"], relation)
                if relation == "overlap":
                    output["overlap_edges_accepted"] += 1
                else:
                    output["transition_edges_accepted"] += 1
            else:
                output["rejected_edges"] += 1
                add_count(output["edge_reject_reasons"], row.get("reject_reason", "unknown"))
    output.update(length_distribution(lengths, thresholds))
    output["component_size_mean"] = _mean(component_sizes)
    output["component_size_p95"] = percentile(component_sizes, 95.0)
    output["multi_camera_ratio"] = rate(output["multi_camera_tracks"], output["global_tracks"])
    output["singleton_ratio"] = rate(output["singleton_tracks"], output["global_tracks"])
    output["single_candidate_ratio"] = rate(output["single_candidate_tracks"], output["global_tracks"])
    output["global_purity_mean"] = _mean(purity_values)
    _merge_scene_summaries(global_root, output)
    worst_tracks = sorted(worst_tracks, key=lambda row: (safe_int(row.get("num_candidates")), safe_int(row.get("duration"))))
    write_json(output, output_path)
    write_csv(worst_tracks[:1000], diagnostics_root / "%s_singleton_global_tracks.csv" % run_name)
    return output


def _files_named(root: Path, names: List[str]) -> List[Path]:
    files = find_data_files(root)
    selected = []
    for path in files:
        lower = path.name.lower()
        if any(name in lower for name in names):
            selected.append(path)
    return selected


def _merge_scene_summaries(root: Path, output: Dict[str, Any]) -> None:
    purity_values = []
    false_merge_values = []
    for path in sorted(root.rglob("summary.json")):
        if "summaries" in set(path.parts):
            continue
        data = read_json(path)
        metrics = data.get("diagnostic_gt_metrics", {})
        if isinstance(metrics, dict):
            if metrics.get("global_purity_mean") is not None:
                purity_values.append(float(metrics.get("global_purity_mean")))
            if metrics.get("false_merge_rate") is not None:
                false_merge_values.append(float(metrics.get("false_merge_rate")))
            output["fragmentation_approx"] += safe_int(metrics.get("fragmentation_approx"))
    if purity_values:
        output["global_purity_mean"] = _mean(purity_values)
    if false_merge_values:
        output["false_merge_rate"] = _mean(false_merge_values)


def _global_track_diag_row(row: Dict[str, Any], num_candidates: int, num_cameras: int) -> Dict[str, Any]:
    return {
        "subset": row.get("subset") or row.get("split"),
        "scene_name": row.get("scene_name"),
        "class_name": row.get("class_name"),
        "global_track_id": row.get("global_track_id"),
        "num_candidates": num_candidates,
        "num_cameras": num_cameras,
        "duration": row.get("duration"),
        "start_frame": row.get("start_frame"),
        "end_frame": row.get("end_frame"),
        "gt_purity": row.get("gt_purity"),
    }


def _mean(values: List[Any]) -> Any:
    nums = [float(item) for item in values if item is not None]
    if not nums:
        return None
    return sum(nums) / float(len(nums))
