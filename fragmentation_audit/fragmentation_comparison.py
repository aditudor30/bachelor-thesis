"""Comparison helpers for V1 vs V2 fragmentation audits."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    read_json,
    safe_float,
    write_csv,
    write_json,
)
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import metric_delta


STAGE_METRICS = {
    "observations": [
        "total_observations",
        "pseudo3d_used_rate",
        "fallback_original_used_rate",
        "has_3d_rate",
        "no_3d_rate",
    ],
    "local_tracking": [
        "total_records",
        "num_tracks",
        "mean",
        "median",
        "p95",
        "singleton_ratio",
        "short_ratio",
        "gt_fragmentation_approx",
    ],
    "tracklets": [
        "total_tracklets",
        "valid_tracklets",
        "valid_ratio",
        "mean",
        "median",
        "singleton_ratio",
        "short_ratio",
    ],
    "candidates": [
        "total_candidates",
        "kept_candidates",
        "kept_ratio",
        "rejected_ratio",
        "mean",
        "median",
        "short_ratio",
        "has_3d_rate",
    ],
    "motion_filtering": [
        "total_candidates",
        "motion_clean_ratio",
        "motion_invalid_ratio",
        "max_step_distance_3d_p95",
        "max_step_distance_3d_p99",
        "jump_ratio_mean",
        "short_ratio",
    ],
    "global_association": [
        "global_tracks",
        "multi_camera_tracks",
        "singleton_tracks",
        "singleton_ratio",
        "single_candidate_ratio",
        "accepted_edges",
        "overlap_edges_accepted",
        "transition_edges_accepted",
        "global_purity_mean",
        "false_merge_rate",
        "fragmentation_approx",
    ],
    "final_export": [
        "generic_rows",
        "frame_records",
        "track1_rows",
        "track1_validation_errors",
        "unique_global_tracks",
        "rows_per_track_singleton_ratio",
        "rows_per_track_short_ratio",
    ],
}


def build_stage_comparison(stage: str, v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact comparison for one stage."""
    metrics = STAGE_METRICS.get(stage, sorted(set(list(v1.keys()) + list(v2.keys()))))
    rows = []
    deltas = {}
    for metric in metrics:
        left = _nested(v1, metric)
        right = _nested(v2, metric)
        delta = metric_delta(left, right)
        deltas[metric] = delta
        rows.append(
            {
                "stage": stage,
                "metric": metric,
                "baseline_v1": left,
                "baseline_v2_fullcam": right,
                "delta": delta,
            }
        )
    return {"stage": stage, "rows": rows, "deltas": deltas}


def build_full_comparison(stage_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build full V1/V2 fragmentation comparison."""
    stages = {}
    rows = []
    for stage, result in stage_results.items():
        comp = build_stage_comparison(stage, result.get("baseline_v1", {}), result.get("baseline_v2", {}))
        stages[stage] = comp
        rows.extend(comp["rows"])
    output = {"stages": stages, "rows": rows}
    output["high_level"] = _high_level_summary(stages)
    return output


def write_comparison_outputs(comparison: Dict[str, Any], output_root: Path, stage_results: Dict[str, Dict[str, Any]]) -> None:
    """Write comparison JSON and CSV files."""
    comparisons_root = output_root / "comparisons"
    write_json(comparison, comparisons_root / "v1_vs_v2_fragmentation_summary.json")
    write_csv(comparison.get("rows", []), comparisons_root / "v1_vs_v2_fragmentation_summary.csv")
    write_csv(comparison.get("rows", []), comparisons_root / "per_stage_delta.csv")
    write_csv(_scope_rows(stage_results, "per_subset"), comparisons_root / "per_subset_fragmentation.csv")
    write_csv(_scope_rows(stage_results, "per_scene"), comparisons_root / "per_scene_fragmentation.csv")
    write_csv(_scope_rows(stage_results, "per_camera"), comparisons_root / "per_camera_fragmentation.csv")
    write_csv(_scope_rows(stage_results, "per_class"), comparisons_root / "per_class_fragmentation.csv")


def load_stage_result(path: Path) -> Dict[str, Any]:
    """Load one stage summary if available."""
    return read_json(path)


def _scope_rows(stage_results: Dict[str, Dict[str, Any]], scope_key: str) -> List[Dict[str, Any]]:
    rows = []
    for stage, result in stage_results.items():
        v1_counts = _ensure_dict(result.get("baseline_v1", {}).get(scope_key, {}))
        v2_counts = _ensure_dict(result.get("baseline_v2", {}).get(scope_key, {}))
        keys = sorted(set(list(v1_counts.keys()) + list(v2_counts.keys())))
        for key in keys:
            left = safe_float(v1_counts.get(key), 0.0)
            right = safe_float(v2_counts.get(key), 0.0)
            rows.append(
                {
                    "stage": stage,
                    "scope": scope_key.replace("per_", ""),
                    "key": key,
                    "baseline_v1": left,
                    "baseline_v2_fullcam": right,
                    "delta": metric_delta(left, right),
                }
            )
    if rows:
        return rows
    return [{"stage": "not_available", "scope": scope_key.replace("per_", ""), "key": "not_available"}]


def _high_level_summary(stages: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def d(stage: str, metric: str) -> Any:
        return stages.get(stage, {}).get("deltas", {}).get(metric)

    return {
        "local_track_delta": d("local_tracking", "num_tracks"),
        "local_short_ratio_delta": d("local_tracking", "short_ratio"),
        "tracklet_delta": d("tracklets", "total_tracklets"),
        "candidate_delta": d("candidates", "total_candidates"),
        "motion_invalid_ratio_delta": d("motion_filtering", "motion_invalid_ratio"),
        "global_tracks_delta": d("global_association", "global_tracks"),
        "global_singleton_ratio_delta": d("global_association", "singleton_ratio"),
        "global_fragmentation_delta": d("global_association", "fragmentation_approx"),
        "generic_rows_delta": d("final_export", "generic_rows"),
        "track1_rows_delta": d("final_export", "track1_rows"),
    }


def _nested(data: Dict[str, Any], key: str) -> Any:
    if not isinstance(data, dict):
        return None
    if "." not in key:
        return data.get(key)
    current = data
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
