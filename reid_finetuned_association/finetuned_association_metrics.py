"""Metrics and deltas for fine-tuned Person ReID association sweeps."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_association.person_association_metrics import collect_person_association_metrics, compute_association_deltas
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import (
    output_root_from_config,
    read_json,
    safe_float,
    write_csv_rows,
    write_json,
)


SWEEP_SUMMARY_FIELDS = [
    "run_name",
    "run_status",
    "track1_valid",
    "track1_errors",
    "track1_rows",
    "global_tracks",
    "person_global_tracks",
    "multi_camera_tracks",
    "person_multi_camera_tracks",
    "singleton_tracks",
    "person_singleton_tracks",
    "accepted_edges",
    "reid_edges_accepted",
    "transition_edges_accepted",
    "global_purity_mean",
    "person_purity_mean",
    "false_merge_rate",
    "person_false_merge_rate",
    "fragmentation_approx",
    "person_fragmentation",
    "person_fragmentation_delta",
    "track1_rows_delta",
    "person_rows_delta",
    "non_person_rows_delta",
    "duplicate_keys",
    "nan_inf_count",
    "non_positive_dimensions",
]


def compare_sweep_to_v2(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Collect all sweep metrics and compare them with V2 current."""
    _unused_progress = progress
    output_root = output_root_from_config(config)
    paths = config.get("paths", {})
    baseline = collect_person_association_metrics(
        "v2_current",
        Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam"))),
        Path(str(paths.get("v2_track1_root", "output/track1_submission/baseline_v2_pseudo3d_fullcam"))),
        None,
    )
    rows: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    for run_root in sorted((output_root / "sweep_runs").iterdir()) if (output_root / "sweep_runs").exists() else []:
        if not run_root.is_dir():
            continue
        metrics = load_or_collect_run_metrics(run_root)
        if not metrics:
            continue
        status = read_json(run_root / "summaries" / "run_status.json") or {}
        metrics["run_status"] = status.get("status", metrics.get("run_status", "unknown"))
        metrics["run_name"] = run_root.name
        delta = compute_association_deltas(metrics, baseline, "vs_v2")
        metrics.update(delta)
        row = sweep_summary_row(metrics)
        rows.append(row)
        delta_row = dict(metrics)
        deltas.append(delta_row)
    write_csv_rows(rows, output_root / "comparison" / "sweep_summary.csv", SWEEP_SUMMARY_FIELDS)
    write_json({"baseline": baseline, "runs": rows}, output_root / "comparison" / "sweep_summary.json")
    delta_fields = _union_fields(deltas)
    write_csv_rows(deltas, output_root / "comparison" / "metric_deltas_vs_v2_current.csv", delta_fields)
    write_json({"baseline": baseline, "runs": deltas}, output_root / "comparison" / "metric_deltas_vs_v2_current.json")
    return {"baseline": baseline, "runs": deltas, "sweep_summary_rows": rows}


def load_or_collect_run_metrics(run_root: Path) -> Dict[str, Any]:
    """Load metrics produced by the existing ReID runner."""
    metrics = read_json(run_root / "summaries" / "run_metrics.json")
    if metrics is not None:
        return metrics
    final_root = run_root / "final_export"
    track1_root = run_root / "track1_submission"
    if not final_root.exists():
        return {}
    return collect_person_association_metrics(run_root.name, final_root, track1_root, run_root / "diagnostics" / "reid_merge_summary.json")


def sweep_summary_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize run metrics into requested summary columns."""
    track1_errors = metrics.get("track1_validation_errors")
    merge_summary = read_json(Path(str(metrics.get("final_export_root", ""))).parent / "diagnostics" / "reid_merge_summary.json") or {}
    track1_valid = track1_errors in (0, 0.0, "0") and str(metrics.get("track1_validation_status", "ok")) in ("ok", "None", "")
    return {
        "run_name": metrics.get("run_name"),
        "run_status": metrics.get("run_status", "ok"),
        "track1_valid": "1" if track1_valid else "0",
        "track1_errors": track1_errors,
        "track1_rows": metrics.get("track1_rows"),
        "global_tracks": metrics.get("global_unique_tracks"),
        "person_global_tracks": metrics.get("person_unique_tracks"),
        "multi_camera_tracks": metrics.get("multi_camera_tracks"),
        "person_multi_camera_tracks": metrics.get("person_multi_camera_tracks", ""),
        "singleton_tracks": metrics.get("singleton_tracks", ""),
        "person_singleton_tracks": metrics.get("person_singleton_tracks"),
        "accepted_edges": merge_summary.get("selected_edges_before_conflict_filter", metrics.get("selected_edges_before_conflict_filter")),
        "reid_edges_accepted": merge_summary.get("selected_edges_with_reid", metrics.get("selected_edges_with_reid")),
        "transition_edges_accepted": metrics.get("transition_edges_accepted", ""),
        "global_purity_mean": metrics.get("global_purity_mean"),
        "person_purity_mean": metrics.get("person_purity"),
        "false_merge_rate": metrics.get("false_merge_rate"),
        "person_false_merge_rate": metrics.get("person_false_merge_rate"),
        "fragmentation_approx": metrics.get("fragmentation_approx"),
        "person_fragmentation": metrics.get("person_fragmentation_approx"),
        "person_fragmentation_delta": metrics.get("vs_v2_person_fragmentation_approx_delta"),
        "track1_rows_delta": metrics.get("vs_v2_track1_rows_delta"),
        "person_rows_delta": metrics.get("vs_v2_person_rows_delta"),
        "non_person_rows_delta": metrics.get("vs_v2_non_person_rows_delta"),
        "duplicate_keys": _validation_metric(metrics, "duplicate_key_count"),
        "nan_inf_count": _validation_metric(metrics, "nan_or_inf_values"),
        "non_positive_dimensions": _validation_metric(metrics, "non_positive_dimensions"),
    }


def metric_delta(run: Dict[str, Any], baseline: Dict[str, Any], key: str) -> Optional[float]:
    """Return run-baseline delta for one numeric key."""
    left = safe_float(run.get(key), None)
    right = safe_float(baseline.get(key), None)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _validation_metric(metrics: Dict[str, Any], key: str) -> Any:
    track1_root = Path(str(metrics.get("track1_root", "")))
    data = read_json(track1_root / "track1_validation_report.json") or read_json(track1_root / "validation_report.json") or {}
    return data.get(key, "")


def _union_fields(rows: List[Dict[str, Any]]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    return fields
