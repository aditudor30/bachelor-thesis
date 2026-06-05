"""Metric collection for global tuning runs."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.global_tuning.tuning_io import count_csv_rows, mean, read_json, safe_float, write_json


def collect_run_metrics(
    run_name: str,
    run_root: Path,
    global_root: Optional[Path] = None,
    final_export_root: Optional[Path] = None,
    track1_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect comparable metrics for one tuning run."""
    if global_root is None:
        global_root = run_root / "global_association"
    if final_export_root is None:
        final_export_root = run_root / "final_export"
    if track1_root is None:
        track1_root = run_root / "track1_submission"
    metrics = {"run_name": run_name}
    metrics.update(collect_global_metrics(global_root))
    metrics.update(collect_final_export_metrics(final_export_root))
    metrics.update(collect_track1_metrics(track1_root))
    return metrics


def collect_global_metrics(global_root: Path) -> Dict[str, Any]:
    """Collect association metrics from per-scene global summaries."""
    summaries = [item for item in global_root.rglob("summary.json") if "summaries" not in item.parts]
    evals = [item for item in global_root.rglob("eval.json") if "summaries" not in item.parts]
    summary_pairs = []
    for path in summaries:
        data = read_json(path)
        if data is not None:
            summary_pairs.append((path, data))
    summary_data = [item[1] for item in summary_pairs]
    eval_data = [read_json(path) for path in evals]
    eval_data = [item for item in eval_data if item is not None]
    global_tracks = sum([int(item.get("global_tracks", 0)) for item in summary_data])
    singleton_tracks = sum([int(item.get("singleton_tracks", 0)) for item in summary_data])
    multi_camera_tracks = sum([int(item.get("multi_camera_tracks", 0)) for item in summary_data])
    accepted_edges = sum([int(item.get("accepted_edges", 0)) for item in summary_data])
    rejected_edges = sum([int(item.get("rejected_edges", 0)) for item in summary_data])
    transition_edges_accepted = _count_transition_edges(global_root, accepted_only=True)
    transition_edges_total = _count_transition_edges(global_root, accepted_only=False)
    per_class_tracks = {}
    per_scene_tracks = {}
    per_class_fragmentation = {}
    per_class_purity_values = {}
    for item in summary_data:
        _merge_counts(per_class_tracks, item.get("per_class_tracks", {}))
    for path, item in summary_pairs:
        scene_name = path.parent.name
        per_scene_tracks[scene_name] = per_scene_tracks.get(scene_name, 0) + int(item.get("global_tracks", 0))
    fragmentation_values = [safe_float(item.get("fragmentation_approx"), None) for item in eval_data]
    fragmentation_values = [value for value in fragmentation_values if value is not None]
    purity_values = [safe_float(item.get("global_purity_mean"), None) for item in eval_data]
    false_merge_values = [safe_float(item.get("false_merge_rate"), None) for item in eval_data]
    for item in eval_data:
        _merge_counts(per_class_fragmentation, item.get("per_class_fragmentation", {}))
        per_class_purity = item.get("per_class_purity", {})
        if isinstance(per_class_purity, dict):
            for class_name, value in per_class_purity.items():
                parsed = safe_float(value, None)
                if parsed is not None:
                    per_class_purity_values.setdefault(str(class_name), []).append(parsed)
    return {
        "global_root": str(global_root),
        "global_summary_files": len(summary_data),
        "num_global_tracks": global_tracks,
        "singleton_tracks": singleton_tracks,
        "multi_camera_tracks": multi_camera_tracks,
        "singleton_ratio": _ratio(singleton_tracks, global_tracks),
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "transition_edges_total": transition_edges_total,
        "transition_edges_accepted": transition_edges_accepted,
        "global_purity_mean": mean(purity_values),
        "false_merge_rate": mean(false_merge_values),
        "fragmentation_approx": sum(fragmentation_values) if fragmentation_values else None,
        "average_component_size": _ratio(sum([int(item.get("total_candidates", 0)) for item in summary_data]), global_tracks),
        "per_class_tracks": per_class_tracks,
        "per_class_fragmentation": per_class_fragmentation,
        "per_class_purity": {key: mean(values) for key, values in per_class_purity_values.items()},
        "per_scene_tracks": per_scene_tracks,
    }


def collect_final_export_metrics(final_export_root: Path) -> Dict[str, Any]:
    """Collect final export metrics."""
    propagation = read_json(final_export_root / "summaries" / "propagation_summary.json") or {}
    export_summary = read_json(final_export_root / "summaries" / "export_summary.json") or {}
    validation = read_json(final_export_root / "validation" / "global_validation_summary.json") or {}
    eval_summary = read_json(final_export_root / "eval" / "global_eval.json") or {}
    generic_rows = _count_generic_rows(final_export_root / "generic_tracking_export")
    return {
        "final_export_root": str(final_export_root),
        "propagation_assignment_ratio": propagation.get("assignment_ratio"),
        "propagation_assigned_records": propagation.get("assigned_records"),
        "propagation_unassigned_records": propagation.get("unassigned_records"),
        "generic_rows": export_summary.get("rows_written", generic_rows),
        "generic_rows_counted": generic_rows,
        "final_validation_errors": validation.get("num_errors"),
        "final_validation_warnings": validation.get("num_warnings"),
        "final_eval_purity_mean": eval_summary.get("global_id_purity_mean"),
        "final_eval_fragmentation_approx": eval_summary.get("fragmentation_approx"),
    }


def collect_track1_metrics(track1_root: Path) -> Dict[str, Any]:
    """Collect Track 1 export and validation metrics."""
    summary = read_json(track1_root / "track1_export_summary.json") or {}
    validation = _find_track1_validation(track1_root)
    output_path = Path(str(summary.get("output_path", track1_root / "track1.txt")))
    if not output_path.is_absolute():
        output_path = Path(output_path)
    rows_written = summary.get("rows_written")
    if rows_written is None:
        rows_written = _count_text_rows(output_path)
    row_stats = track1_rows_per_object_stats(output_path)
    return {
        "track1_root": str(track1_root),
        "track1_rows": rows_written,
        "track1_output_path": str(output_path),
        "track1_validation_status": validation.get("status"),
        "track1_validation_errors": validation.get("num_errors"),
        "track1_duplicate_errors": _count_errors_matching(validation, "duplicate"),
        "track1_rows_per_object_mean": row_stats.get("mean"),
        "track1_rows_per_object_median": row_stats.get("median"),
        "track1_rows_per_object_p95": row_stats.get("p95"),
        "track1_unique_object_ids": row_stats.get("num_objects"),
        "track1_short_object_ids": row_stats.get("short_objects"),
    }


def write_run_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Write one run metrics file."""
    write_json(metrics, path)


def compute_metric_deltas(run: Dict[str, Any], baseline: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Compute useful deltas between a run and a baseline."""
    output = {}
    keys = [
        "fragmentation_approx",
        "track1_rows",
        "multi_camera_tracks",
        "num_global_tracks",
        "singleton_ratio",
        "global_purity_mean",
        "false_merge_rate",
        "accepted_edges",
        "transition_edges_accepted",
    ]
    for key in keys:
        run_value = safe_float(run.get(key), None)
        base_value = safe_float(baseline.get(key), None)
        if run_value is None or base_value is None:
            output["%s_%s_delta" % (prefix, key)] = None
        else:
            output["%s_%s_delta" % (prefix, key)] = run_value - base_value
    base_frag = safe_float(baseline.get("fragmentation_approx"), None)
    run_frag = safe_float(run.get("fragmentation_approx"), None)
    output["%s_fragmentation_reduction" % prefix] = None
    if base_frag is not None and base_frag != 0 and run_frag is not None:
        output["%s_fragmentation_reduction" % prefix] = (base_frag - run_frag) / base_frag
    return output


def track1_rows_per_object_stats(path: Path) -> Dict[str, Any]:
    """Compute simple row-count distribution per official object id."""
    if not path.exists():
        return {"num_objects": 0, "mean": None, "median": None, "p95": None, "short_objects": 0}
    counts = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        for row in reader:
            row = [item for item in row if item != ""]
            if len(row) < 4:
                continue
            key = (row[0], row[1], row[2])
            counts[key] = counts.get(key, 0) + 1
    values = sorted(counts.values())
    return {
        "num_objects": len(values),
        "mean": _mean_int(values),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "short_objects": len([value for value in values if value <= 2]),
    }


def _count_transition_edges(global_root: Path, accepted_only: bool) -> int:
    total = 0
    for path in sorted(global_root.rglob("transition_edges.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if accepted_only and str(row.get("accepted", "")).lower() not in ("true", "1", "yes"):
                    continue
                total += 1
    return total


def _count_generic_rows(root: Path) -> int:
    return sum([count_csv_rows(path) for path in root.rglob("*.csv")])


def _count_text_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _find_track1_validation(root: Path) -> Dict[str, Any]:
    for name in ["track1_validation_report.json", "validation_report.json"]:
        data = read_json(root / name)
        if data is not None:
            return data
    data = read_json(root / "validation" / "track1_validation_report.json")
    return data if data is not None else {}


def _count_errors_matching(validation: Dict[str, Any], text: str) -> int:
    errors = validation.get("errors", [])
    if not isinstance(errors, list):
        return 0
    return len([item for item in errors if text in str(item)])


def _merge_counts(left: Dict[str, int], right: Dict[str, Any]) -> None:
    if not isinstance(right, dict):
        return
    for key, value in right.items():
        left[str(key)] = left.get(str(key), 0) + int(value)


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    den = safe_float(denominator, None)
    num = safe_float(numerator, None)
    if den is None or num is None or den == 0:
        return None
    return float(num) / float(den)


def _mean_int(values: List[int]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _percentile(values: List[int], percentile: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    index = int(round((float(percentile) / 100.0) * float(len(values) - 1)))
    index = max(0, min(len(values) - 1, index))
    return float(values[index])
