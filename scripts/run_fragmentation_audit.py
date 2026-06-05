"""Run the full baseline_v2_pseudo3d_fullcam fragmentation audit."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.fragmentation_audit.candidate_fragmentation_audit import audit_candidates
from deep_oc_sort_3d.fragmentation_audit.final_export_fragmentation_audit import audit_final_export
from deep_oc_sort_3d.fragmentation_audit.fragmentation_comparison import (
    build_full_comparison,
    build_stage_comparison,
    write_comparison_outputs,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    load_yaml,
    safe_float,
    write_csv,
    write_json,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_report import write_fragmentation_report
from deep_oc_sort_3d.fragmentation_audit.fragmentation_root_cause import analyze_root_cause
from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.global_association_fragmentation_audit import audit_global_association
from deep_oc_sort_3d.fragmentation_audit.local_tracking_fragmentation_audit import audit_local_tracking
from deep_oc_sort_3d.fragmentation_audit.motion_filtering_fragmentation_audit import audit_motion_filtering
from deep_oc_sort_3d.fragmentation_audit.observation_fragmentation_audit import audit_observations
from deep_oc_sort_3d.fragmentation_audit.tracklet_fragmentation_audit import audit_tracklets


STAGES = [
    "observations",
    "local_tracking",
    "tracklets",
    "candidates",
    "motion_filtering",
    "global_association",
    "final_export",
]


def run_fragmentation_audit(config_path: Path, stages: List[str], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Run selected fragmentation audit stages."""
    config = load_yaml(config_path)
    output_root = Path(config.get("fragmentation_audit", {}).get("output_root", "output/baseline_v2_pseudo3d_fullcam_fragmentation_audit"))
    thresholds = thresholds_from_config(config)
    stage_summaries_root = output_root / "stage_summaries"
    diagnostics_root = output_root / "diagnostics"
    stage_summaries_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    selected = list(STAGES) if not stages or "all" in stages else stages
    stage_results = {}
    for stage in selected:
        stage_results[stage] = run_one_stage(config, stage, thresholds, output_root, show_progress, overwrite)
    comparison = build_full_comparison(stage_results)
    write_comparison_outputs(comparison, output_root, stage_results)
    root_cause = analyze_root_cause(comparison)
    write_named_diagnostics(stage_results, comparison, root_cause, diagnostics_root)
    write_fragmentation_report(comparison, root_cause, stage_results, output_root)
    final_summary = {
        "config": str(config_path),
        "output_root": str(output_root),
        "stages": list(stage_results.keys()),
        "root_cause": root_cause,
        "high_level": comparison.get("high_level", {}),
    }
    write_json(final_summary, output_root / "report" / "BASELINE_V2_FULLCAM_FRAGMENTATION_AUDIT_SUMMARY_COMPACT.json")
    print("verdict: %s" % root_cause.get("verdict"))
    print("Wrote %s" % output_root)
    return final_summary


def run_one_stage(
    config: Dict[str, Any],
    stage: str,
    thresholds: FragmentationThresholds,
    output_root: Path,
    show_progress: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    """Run one stage for V1 and V2, then write a paired summary."""
    if stage not in STAGES:
        raise ValueError("Unknown fragmentation stage: %s" % stage)
    stage_path = output_root / "stage_summaries" / ("%s_fragmentation_summary.json" % stage)
    if stage_path.exists() and not overwrite:
        return load_yaml(stage_path) if stage_path.suffix in (".yaml", ".yml") else _read_json(stage_path)
    paths = config.get("paths", {})
    diagnostics_root = output_root / "diagnostics"
    v1 = _run_stage_for_version(stage, "baseline_v1", paths, thresholds, output_root, diagnostics_root, show_progress)
    v2 = _run_stage_for_version(stage, "baseline_v2_fullcam", paths, thresholds, output_root, diagnostics_root, show_progress)
    comparison = build_stage_comparison(stage, v1, v2)
    result = {"stage": stage, "baseline_v1": v1, "baseline_v2": v2, "comparison": comparison}
    write_json(result, stage_path)
    return result


def thresholds_from_config(config: Dict[str, Any]) -> FragmentationThresholds:
    """Build thresholds from config."""
    data = config.get("thresholds", {})
    return FragmentationThresholds(
        singleton_length=int(data.get("singleton_length", 1)),
        short_track_length=int(data.get("short_track_length", 3)),
        very_short_track_length=int(data.get("very_short_track_length", 5)),
        long_track_length=int(data.get("long_track_length", 30)),
        high_fragmentation_ratio=float(data.get("high_fragmentation_ratio", 0.5)),
        high_singleton_ratio=float(data.get("high_singleton_ratio", 0.5)),
        high_rows_per_track_p95=float(data.get("high_rows_per_track_p95", 100.0)),
        motion_invalid_warning_ratio=float(data.get("motion_invalid_warning_ratio", 0.10)),
    )


def write_named_diagnostics(
    stage_results: Dict[str, Dict[str, Any]],
    comparison: Dict[str, Any],
    root_cause: Dict[str, Any],
    diagnostics_root: Path,
) -> None:
    """Write requested top-level diagnostic CSV aliases."""
    write_csv(_worst_scope_rows(stage_results, "per_scene"), diagnostics_root / "worst_scenes_by_fragmentation.csv")
    write_csv(_worst_scope_rows(stage_results, "per_camera"), diagnostics_root / "worst_cameras_by_fragmentation.csv")
    write_csv(_worst_scope_rows(stage_results, "per_class"), diagnostics_root / "worst_classes_by_fragmentation.csv")
    write_csv(_singleton_summary_rows(stage_results), diagnostics_root / "singleton_analysis.csv")
    write_csv(_short_summary_rows(stage_results), diagnostics_root / "short_track_analysis.csv")
    write_csv(_final_rows_summary_rows(stage_results), diagnostics_root / "rows_per_track_distribution.csv")
    write_csv(_global_id_summary_rows(stage_results), diagnostics_root / "global_id_fragmentation_analysis.csv")
    write_csv(_transition_edge_rows(stage_results), diagnostics_root / "transition_edge_analysis.csv")
    write_csv(_local_to_global_rows(stage_results), diagnostics_root / "local_to_global_fragmentation_map.csv")
    write_csv(_worst_fragmented_rows(stage_results), diagnostics_root / "worst_fragmented_tracks.csv")
    write_json(root_cause, diagnostics_root / "root_cause_analysis.json")
    write_json(comparison.get("high_level", {}), diagnostics_root / "high_level_fragmentation_deltas.json")


def _run_stage_for_version(
    stage: str,
    version: str,
    paths: Dict[str, Any],
    thresholds: FragmentationThresholds,
    output_root: Path,
    diagnostics_root: Path,
    show_progress: bool,
) -> Dict[str, Any]:
    prefix = "v1" if version == "baseline_v1" else "v2"
    stage_output = output_root / "stage_summaries" / ("%s_%s_raw.json" % (version, stage))
    if stage == "observations":
        return audit_observations(Path(paths.get("%s_pipeline_root" % prefix, "")), stage_output, version, show_progress)
    if stage == "local_tracking":
        return audit_local_tracking(Path(paths.get("%s_local_tracks_root" % prefix, "")), stage_output, diagnostics_root, version, thresholds, show_progress)
    if stage == "tracklets":
        return audit_tracklets(Path(paths.get("%s_tracklets_root" % prefix, "")), stage_output, diagnostics_root, version, thresholds, show_progress)
    if stage == "candidates":
        return audit_candidates(Path(paths.get("%s_candidates_root" % prefix, "")), stage_output, diagnostics_root, version, thresholds, show_progress)
    if stage == "motion_filtering":
        return audit_motion_filtering(Path(paths.get("%s_motion_clean_root" % prefix, "")), stage_output, diagnostics_root, version, thresholds, show_progress)
    if stage == "global_association":
        return audit_global_association(Path(paths.get("%s_global_root" % prefix, "")), stage_output, diagnostics_root, version, thresholds, show_progress)
    if stage == "final_export":
        return audit_final_export(
            Path(paths.get("%s_final_export_root" % prefix, "")),
            Path(paths.get("%s_track1_root" % prefix, "")),
            stage_output,
            diagnostics_root,
            version,
            thresholds,
            show_progress,
        )
    raise ValueError("Unknown stage: %s" % stage)


def _worst_scope_rows(stage_results: Dict[str, Dict[str, Any]], scope_name: str) -> List[Dict[str, Any]]:
    rows = []
    for stage, result in stage_results.items():
        v1 = result.get("baseline_v1", {}).get(scope_name, {})
        v2 = result.get("baseline_v2", {}).get(scope_name, {})
        keys = sorted(set(list(_dict(v1).keys()) + list(_dict(v2).keys())))
        for key in keys:
            v1_count = safe_float(_dict(v1).get(key), 0.0)
            v2_count = safe_float(_dict(v2).get(key), 0.0)
            rows.append({"stage": stage, "scope": scope_name, "key": key, "v1": v1_count, "v2": v2_count, "delta": _delta(v1_count, v2_count)})
    return sorted(rows, key=lambda row: float(row.get("delta") or 0.0), reverse=True)[:500]


def _singleton_summary_rows(stage_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _metric_rows(stage_results, ["singleton_ratio", "singleton_count", "singleton_tracks", "single_candidate_ratio"])


def _short_summary_rows(stage_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _metric_rows(stage_results, ["short_ratio", "short_count", "very_short_ratio", "very_short_count"])


def _final_rows_summary_rows(stage_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    final_stage = stage_results.get("final_export", {})
    for version_key in ["baseline_v1", "baseline_v2"]:
        data = final_stage.get(version_key, {})
        rows.append(
            {
                "version": version_key,
                "generic_rows": data.get("generic_rows"),
                "track1_rows": data.get("track1_rows"),
                "unique_global_tracks": data.get("unique_global_tracks"),
                "rows_per_track_mean": data.get("rows_per_track", {}).get("mean"),
                "rows_per_track_p95": data.get("rows_per_track", {}).get("p95"),
                "rows_per_track_short_ratio": data.get("rows_per_track_short_ratio"),
            }
        )
    return rows


def _global_id_summary_rows(stage_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    global_stage = stage_results.get("global_association", {})
    for version_key in ["baseline_v1", "baseline_v2"]:
        data = global_stage.get(version_key, {})
        rows.append(
            {
                "version": version_key,
                "global_tracks": data.get("global_tracks"),
                "multi_camera_tracks": data.get("multi_camera_tracks"),
                "singleton_tracks": data.get("singleton_tracks"),
                "singleton_ratio": data.get("singleton_ratio"),
                "fragmentation_approx": data.get("fragmentation_approx"),
            }
        )
    return rows


def _transition_edge_rows(stage_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    global_stage = stage_results.get("global_association", {})
    for version_key in ["baseline_v1", "baseline_v2"]:
        data = global_stage.get(version_key, {})
        rows.append(
            {
                "version": version_key,
                "accepted_edges": data.get("accepted_edges"),
                "overlap_edges_accepted": data.get("overlap_edges_accepted"),
                "transition_edges_accepted": data.get("transition_edges_accepted"),
                "rejected_edges": data.get("rejected_edges"),
                "edge_reject_reasons": data.get("edge_reject_reasons"),
            }
        )
    return rows


def _local_to_global_rows(stage_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for version_key in ["baseline_v1", "baseline_v2"]:
        local_data = stage_results.get("local_tracking", {}).get(version_key, {})
        global_data = stage_results.get("global_association", {}).get(version_key, {})
        rows.append(
            {
                "version": version_key,
                "local_tracks": local_data.get("num_tracks"),
                "global_tracks": global_data.get("global_tracks"),
                "local_short_ratio": local_data.get("short_ratio"),
                "global_singleton_ratio": global_data.get("singleton_ratio"),
            }
        )
    return rows


def _worst_fragmented_rows(stage_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for version_key in ["baseline_v1", "baseline_v2"]:
        local_data = stage_results.get("local_tracking", {}).get(version_key, {})
        global_data = stage_results.get("global_association", {}).get(version_key, {})
        rows.append(
            {
                "version": version_key,
                "local_gt_fragmentation_approx": local_data.get("gt_fragmentation_approx"),
                "global_fragmentation_approx": global_data.get("fragmentation_approx"),
                "global_purity_mean": global_data.get("global_purity_mean"),
                "false_merge_rate": global_data.get("false_merge_rate"),
            }
        )
    return rows


def _metric_rows(stage_results: Dict[str, Dict[str, Any]], metrics: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for stage, result in stage_results.items():
        for metric in metrics:
            rows.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "baseline_v1": result.get("baseline_v1", {}).get(metric),
                    "baseline_v2_fullcam": result.get("baseline_v2", {}).get(metric),
                    "delta": _delta(result.get("baseline_v1", {}).get(metric), result.get("baseline_v2", {}).get(metric)),
                }
            )
    return rows


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _delta(left: Any, right: Any) -> Any:
    lval = safe_float(left)
    rval = safe_float(right)
    if lval is None or rval is None:
        return None
    return rval - lval


def _read_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run V1 vs V2 fullcam fragmentation audit.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", default="all", choices=STAGES + ["all"])
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def build_single_stage_arg_parser(stage: str) -> argparse.ArgumentParser:
    """Build parser for single-stage wrappers."""
    parser = argparse.ArgumentParser(description="Run %s fragmentation audit." % stage)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def run_stage_cli(stage: str) -> None:
    """Entry point for single-stage wrapper scripts."""
    parser = build_single_stage_arg_parser(stage)
    args = parser.parse_args()
    run_fragmentation_audit(args.config, stages=[stage], show_progress=args.progress, overwrite=args.overwrite)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    run_fragmentation_audit(args.config, stages=stages, show_progress=args.progress, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
