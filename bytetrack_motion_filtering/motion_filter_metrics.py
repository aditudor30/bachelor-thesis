"""Metrics and comparison outputs for Step 21E."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_comparison import collect_run_metrics
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import output_root, variant_names, variant_root
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import read_csv, read_json, write_csv, write_json
from deep_oc_sort_3d.bytetrack_tuning.tuning_metrics import collect_global_fragmentation_by_class


def collect_motion_filter_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect variant and baseline metrics using one common schema."""
    variants = {}
    for name in variant_names(config):
        variants[name] = collect_variant_pipeline_metrics(config, name)
    baselines = {
        "baseline_v2_current": _collect_baseline(config, "baseline_v2_current"),
        "bytetrack_21b": _collect_baseline(config, "bytetrack_21b"),
        "bytetrack_21c_best": _collect_baseline(config, "bytetrack_21c_best"),
    }
    rows = [_summary_row(name, value, baselines) for name, value in variants.items()]
    root = output_root(config)
    comparison = root / "comparison"
    diagnostics = root / "diagnostics"
    write_csv(comparison / "motion_filter_sweep_summary.csv", rows)
    write_json(comparison / "motion_filter_sweep_summary.json", {"variants": variants, "baselines": baselines, "rows": rows})
    _write_deltas(comparison, variants, baselines)
    diagnostic_tables = _write_diagnostics(config, diagnostics)
    return {"variants": variants, "baselines": baselines, "rows": rows, "diagnostics": diagnostic_tables}


def collect_variant_pipeline_metrics(config: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    """Collect all required metrics for one Step 21E output root."""
    root = variant_root(config, variant_name)
    source = config.get("paths", {}).get("bytetrack_21c_best", {})
    high = collect_run_metrics(
        {
            "name": variant_name,
            "pipeline_root": Path(str(config.get("paths", {}).get("v2_observations_root", ""))).parent,
            "local_tracks_root": source.get("local_tracks_root"),
            "tracklets_root": source.get("tracklets_root"),
            "candidates_root": source.get("candidates_root"),
            "motion_clean_root": root / "motion_clean",
            "global_root": root / "global_mtmc",
            "final_export_root": root / "final_export",
            "track1_root": root / "track1_submission",
        }
    )
    motion = read_json(root / "summaries" / "motion_quality_summary.json")
    validation = read_json(root / "validation" / "track1_validation_summary.json")
    fragmentation = collect_global_fragmentation_by_class(root / "global_mtmc")
    global_metrics = high.get("global_association", {})
    return {
        "variant_name": variant_name,
        "status": _status(root, motion, validation),
        "motion": motion,
        "global": global_metrics,
        "final_export_rows": high.get("final_export", {}).get("generic_rows"),
        "track1": high.get("track1", {}),
        "validation": validation,
        "per_class_fragmentation": fragmentation,
        "person_fragmentation": int(fragmentation.get("Person", 0) or 0),
        "non_person_fragmentation": sum(int(value or 0) for key, value in fragmentation.items() if key != "Person"),
    }


def _collect_baseline(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    spec = config.get("paths", {}).get(name, {})
    high = collect_run_metrics(
        {
            "name": name,
            "pipeline_root": Path(str(config.get("paths", {}).get("v2_observations_root", ""))).parent,
            "local_tracks_root": spec.get("local_tracks_root"),
            "tracklets_root": spec.get("tracklets_root"),
            "candidates_root": spec.get("candidates_root"),
            "motion_clean_root": spec.get("motion_clean_root"),
            "global_root": spec.get("global_root"),
            "final_export_root": spec.get("final_export_root"),
            "track1_root": spec.get("track1_root"),
        }
    )
    motion = high.get("motion_clean", {})
    input_candidates = _input_candidates(motion, high.get("candidates", {}))
    clean = _motion_clean_count(motion)
    fragmentation = collect_global_fragmentation_by_class(Path(str(spec.get("global_root", ""))))
    return {
        "name": name,
        "motion": {
            "input_candidates": input_candidates,
            "motion_clean_candidates": clean,
            "motion_clean_retention": _ratio(clean, input_candidates),
        },
        "global": high.get("global_association", {}),
        "final_export_rows": high.get("final_export", {}).get("generic_rows"),
        "track1": high.get("track1", {}),
        "per_class_fragmentation": fragmentation,
        "person_fragmentation": int(fragmentation.get("Person", 0) or 0),
        "non_person_fragmentation": sum(int(value or 0) for key, value in fragmentation.items() if key != "Person"),
    }


def _summary_row(name: str, value: Dict[str, Any], baselines: Dict[str, Any]) -> Dict[str, Any]:
    motion = value.get("motion", {})
    global_metrics = value.get("global", {})
    track1 = value.get("track1", {})
    v2 = baselines.get("baseline_v2_current", {})
    bt21c = baselines.get("bytetrack_21c_best", {})
    track1_rows = track1.get("rows")
    multi = global_metrics.get("multi_camera_tracks")
    return {
        "variant_name": name,
        "status": value.get("status"),
        "input_candidates": motion.get("input_candidates"),
        "motion_clean_candidates": motion.get("motion_clean_candidates"),
        "motion_clean_retention": motion.get("motion_clean_retention"),
        "rejected_candidates": motion.get("rejected_candidates"),
        "rejection_rate": motion.get("rejection_rate"),
        "motion_good": motion.get("motion_good"),
        "motion_suspicious": motion.get("motion_suspicious"),
        "motion_invalid": motion.get("motion_invalid"),
        "motion_unknown": motion.get("motion_unknown"),
        "step_p95": motion.get("step_p95"),
        "step_p99": motion.get("step_p99"),
        "step_max": motion.get("step_max"),
        "jump_count": motion.get("jump_count"),
        "jump_ratio": motion.get("jump_ratio"),
        "global_tracks": global_metrics.get("global_tracks"),
        "multi_camera_tracks": multi,
        "singleton_tracks": global_metrics.get("singleton_tracks"),
        "accepted_edges": global_metrics.get("accepted_edges"),
        "transition_edges_accepted": global_metrics.get("transition_edges_accepted"),
        "global_purity_mean": global_metrics.get("global_purity_mean"),
        "false_merge_rate": global_metrics.get("false_merge_rate"),
        "fragmentation_approx": global_metrics.get("fragmentation_approx"),
        "person_fragmentation": value.get("person_fragmentation"),
        "non_person_fragmentation": value.get("non_person_fragmentation"),
        "final_export_rows": value.get("final_export_rows"),
        "track1_rows": track1_rows,
        "track1_validation_errors": track1.get("validation_errors"),
        "track1_rows_retention_vs_v2_current": _ratio(track1_rows, v2.get("track1", {}).get("rows")),
        "track1_rows_retention_vs_bytetrack_21c_best": _ratio(track1_rows, bt21c.get("track1", {}).get("rows")),
        "multi_camera_retention_vs_v2_current": _ratio(multi, v2.get("global", {}).get("multi_camera_tracks")),
        "multi_camera_retention_vs_bytetrack_21c_best": _ratio(multi, bt21c.get("global", {}).get("multi_camera_tracks")),
    }


def _write_deltas(root: Path, variants: Dict[str, Any], baselines: Dict[str, Any]) -> None:
    for baseline_name, filename in (
        ("baseline_v2_current", "metric_deltas_vs_v2_current.csv"),
        ("bytetrack_21b", "metric_deltas_vs_bytetrack_21b.csv"),
        ("bytetrack_21c_best", "metric_deltas_vs_bytetrack_21c_best.csv"),
    ):
        rows = []
        baseline = baselines.get(baseline_name, {})
        for name, value in variants.items():
            rows.extend(_delta_rows(name, value, baseline))
        write_csv(root / filename, rows)


def _delta_rows(name: str, value: Dict[str, Any], baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = {
        "motion_clean_retention": (value.get("motion", {}).get("motion_clean_retention"), baseline.get("motion", {}).get("motion_clean_retention")),
        "global_tracks": (value.get("global", {}).get("global_tracks"), baseline.get("global", {}).get("global_tracks")),
        "multi_camera_tracks": (value.get("global", {}).get("multi_camera_tracks"), baseline.get("global", {}).get("multi_camera_tracks")),
        "global_purity_mean": (value.get("global", {}).get("global_purity_mean"), baseline.get("global", {}).get("global_purity_mean")),
        "false_merge_rate": (value.get("global", {}).get("false_merge_rate"), baseline.get("global", {}).get("false_merge_rate")),
        "fragmentation_approx": (value.get("global", {}).get("fragmentation_approx"), baseline.get("global", {}).get("fragmentation_approx")),
        "track1_rows": (value.get("track1", {}).get("rows"), baseline.get("track1", {}).get("rows")),
    }
    return [{"variant_name": name, "metric": key, "baseline_value": base, "variant_value": current, "delta": _delta(base, current)} for key, (current, base) in pairs.items()]


def _write_diagnostics(config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    all_rows = []
    summary_rows = []
    for name in variant_names(config):
        variant = variant_root(config, name)
        rows = read_csv(variant / "summaries" / "candidate_diagnostics.csv")
        all_rows.extend(rows)
        motion = read_json(variant / "summaries" / "motion_quality_summary.json")
        summary_rows.append({key: value for key, value in motion.items() if key != "files"})
    write_csv(root / "motion_filter_retention_summary.csv", summary_rows)
    write_json(root / "motion_filter_retention_summary.json", {"rows": summary_rows})
    tables = {
        "rejected_by_gap_bucket": _group_retention(all_rows, "gap_bucket", keep=False),
        "rejected_by_class": _group_retention(all_rows, "class_name", keep=False),
        "rejected_by_scene": _group_retention(all_rows, "scene_name", keep=False),
        "rejected_by_camera_pair": _group_retention(all_rows, "camera_id", keep=False),
        "retained_by_gap_bucket": _group_retention(all_rows, "gap_bucket", keep=True),
        "retained_by_class": _group_retention(all_rows, "class_name", keep=True),
        "retained_by_scene": _group_retention(all_rows, "scene_name", keep=True),
        "retained_by_camera_pair": _group_retention(all_rows, "camera_id", keep=True),
        "bbox_jump_tolerance_analysis": _jump_rows(all_rows, "bbox"),
        "pseudo3d_jump_analysis": _jump_rows(all_rows, "pseudo3d"),
    }
    for name, rows in tables.items():
        write_csv(root / (name + ".csv"), rows)
    warnings = []
    if not all_rows:
        warnings.append("No candidate diagnostics found; run variants before comparison.")
    write_json(root / "warnings.json", {"warnings": warnings})
    return tables


def _group_retention(rows: List[Dict[str, Any]], key: str, keep: bool) -> List[Dict[str, Any]]:
    groups = {}
    for row in rows:
        group_key = _gap_bucket(row.get("max_gap")) if key == "gap_bucket" else str(row.get(key, "unknown"))
        compound = (str(row.get("variant_name", "")), group_key)
        target = groups.setdefault(compound, {"total": 0, "retained": 0, "rejected": 0})
        target["total"] += 1
        if _truthy(row.get("is_motion_clean")):
            target["retained"] += 1
        else:
            target["rejected"] += 1
    output = []
    for (variant, group), counts in sorted(groups.items()):
        selected = counts["retained"] if keep else counts["rejected"]
        output.append({"variant_name": variant, key: group, "total_candidates": counts["total"], "selected_candidates": selected, "retained_candidates": counts["retained"], "rejected_candidates": counts["rejected"], "rate": _ratio(selected, counts["total"])})
    return output


def _jump_rows(rows: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        tolerated = int(float(row.get("tolerated_jump_count", 0) or 0))
        extreme = int(float(row.get("extreme_jump_count", 0) or 0))
        if tolerated <= 0 and extreme <= 0:
            continue
        output.append({
            "variant_name": row.get("variant_name"),
            "candidate_id": row.get("candidate_id"),
            "scene_name": row.get("scene_name"),
            "camera_id": row.get("camera_id"),
            "class_name": row.get("class_name"),
            "analysis_type": kind,
            "is_motion_clean": row.get("is_motion_clean"),
            "tolerated_jump_count": tolerated,
            "extreme_jump_count": extreme,
            "max_step_distance_3d": row.get("max_step_distance_3d"),
            "violations_json": row.get("violations_json"),
        })
    return output


def _input_candidates(motion: Dict[str, Any], candidates: Dict[str, Any]) -> int:
    direct = motion.get("input_candidates", motion.get("total_candidates"))
    if direct is not None:
        return int(direct or 0)
    files = motion.get("files", [])
    if isinstance(files, list):
        total = sum(int(row.get("num_candidates", row.get("input_candidates", 0)) or 0) for row in files if isinstance(row, dict))
        if total:
            return total
    return int(candidates.get("kept_candidates", candidates.get("total_candidates", 0)) or 0)


def _motion_clean_count(motion: Dict[str, Any]) -> int:
    return int(motion.get("clean_count", motion.get("motion_clean_candidates", 0)) or 0)


def _status(root: Path, motion: Dict[str, Any], validation: Dict[str, Any]) -> str:
    if motion.get("status") == "error":
        return "error"
    if not (root / "track1_submission" / "track1.txt").exists():
        return "incomplete"
    if validation.get("num_errors") not in (0, "0"):
        return "invalid"
    return "ok"


def _gap_bucket(value: Any) -> str:
    try:
        gap = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if gap <= 1:
        return "gap_0_or_1"
    if gap <= 5:
        return "gap_2_5"
    if gap <= 15:
        return "gap_6_15"
    if gap <= 30:
        return "gap_16_30"
    return "gap_gt_30"


def _ratio(a: Any, b: Any) -> Optional[float]:
    try:
        denominator = float(b)
        return None if denominator <= 0.0 else float(a) / denominator
    except (TypeError, ValueError):
        return None


def _delta(a: Any, b: Any) -> Optional[float]:
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() in ("true", "1", "yes")
