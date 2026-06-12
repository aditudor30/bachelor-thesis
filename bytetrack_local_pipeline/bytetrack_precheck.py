"""Extended Step 21B precheck built on the isolated Step 21A benchmark."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import write_csv, write_json, write_yaml
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import load_bytetrack_pipeline_config
from deep_oc_sort_3d.local_tracker_benchmark.benchmark_runner import run_local_tracker_benchmark


PASS_VERDICTS = (
    "bytetrack_precheck_pass_full_rerun_recommended",
    "bytetrack_precheck_pass_with_warnings",
)


def run_bytetrack_precheck(
    config_path: Path,
    progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run current/ByteTrack/BoT-SORT-no-ReID on val and holdout."""
    config = load_bytetrack_pipeline_config(config_path)
    root = Path(str(config.get("paths", {}).get("output_precheck_root")))
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    root.mkdir(parents=True, exist_ok=True)
    benchmark_config = _benchmark_config(config, root)
    generated_config = root / "configs" / "precheck_benchmark.yaml"
    write_yaml(generated_config, benchmark_config)
    result = run_local_tracker_benchmark(
        generated_config,
        subset_name="all_available",
        requested_trackers=["current", "bytetrack", "botsort_no_reid"],
        progress=progress,
        overwrite=False,
        skip_existing=False,
    )
    rows = result.get("comparison", {}).get("rows", [])
    verdict = decide_precheck_verdict(rows)
    summary = {"rows": rows, "verdict": verdict, "benchmark_root": str(root)}
    write_json(root / "precheck_summary.json", summary)
    write_csv(root / "precheck_summary.csv", rows)
    write_json(root / "precheck_verdict.json", verdict)
    return summary


def decide_precheck_verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply continuity, purity and non-Person safeguards."""
    current = _find(rows, "current_local_tracker")
    bytetrack = _find(rows, "bytetrack_style_yolo11m")
    botsort = _find(rows, "botsort_style_no_reid_yolo11m")
    if current is None or bytetrack is None:
        return {"label": "precheck_invalid_fix_required", "reasons": ["required_tracker_metrics_missing"]}
    byte_reasons = _candidate_failures(bytetrack, current)
    bot_reasons = [] if botsort is None else _candidate_failures(botsort, current)
    if not byte_reasons:
        warning_reasons = []
        false_delta = _delta(bytetrack.get("false_merge_suspicion_rate"), current.get("false_merge_suspicion_rate"))
        coverage = _ratio(bytetrack.get("gt_matched_records"), current.get("gt_matched_records"))
        if false_delta is not None and false_delta > 0.0:
            warning_reasons.append("false_merge_suspicion_increased")
        if coverage is not None and coverage < 0.90:
            warning_reasons.append("gt_matched_record_retention_below_0.90")
        label = "bytetrack_precheck_pass_with_warnings" if warning_reasons else "bytetrack_precheck_pass_full_rerun_recommended"
        return {"label": label, "reasons": warning_reasons, "selected_tracker": "bytetrack_style_yolo11m"}
    if botsort is not None and not bot_reasons:
        return {
            "label": "botsort_no_reid_precheck_preferred",
            "reasons": byte_reasons,
            "selected_tracker": "botsort_style_no_reid_yolo11m",
        }
    if _continuity_better(bytetrack, current):
        return {"label": "precheck_inconclusive", "reasons": byte_reasons}
    return {"label": "current_tracker_still_best", "reasons": byte_reasons}


def precheck_allows_full_rerun(verdict: Dict[str, Any]) -> bool:
    """Return whether the primary ByteTrack rerun is allowed."""
    return str(verdict.get("label")) in PASS_VERDICTS


def _benchmark_config(config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    paths = config.get("paths", {})
    precheck = config.get("precheck", {})
    benchmark = {
        "internal_holdout_scenes": {"train": list(precheck.get("internal_holdout", {}).get("scenes", []))},
        "val_scenes": {"val": list(precheck.get("official_val", {}).get("scenes", []))},
        "test_scenes": {"test": list(precheck.get("test_probe", {}).get("scenes", []))},
    }
    return {
        "local_tracker_benchmark": {"output_root": str(root), "progress": True, "random_seed": 42},
        "paths": {
            "dataset_root": paths.get("dataset_root"),
            "yolo_pipeline_root": paths.get("yolo_pipeline_root"),
            "v2_pipeline_root": paths.get("yolo_pipeline_root"),
            "observations_root": paths.get("v2_observations_root"),
            "current_local_tracks_root": paths.get("current_local_tracks_root"),
        },
        "benchmark": benchmark,
        "trackers": {
            "run_current_local_tracker": True,
            "run_bytetrack_style_yolo11m": True,
            "run_botsort_style_no_reid_yolo11m": True,
            "run_botsort_sbs_mot17_yolo11m": False,
            "run_botsort_sbs_mot20_yolo11m": False,
            "run_botsort_osnet_finetuned_yolo11m": False,
        },
        "detections": {"min_confidence_for_input": config.get("bytetrack_style", {}).get("min_confidence_for_input", 0.001)},
        "bytetrack_style": dict(config.get("bytetrack_style", {})),
        "botsort_style": dict(config.get("bytetrack_style", {})),
        "figures": {"enabled": True},
    }


def _candidate_failures(candidate: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    reasons = []
    if _number(candidate.get("median_track_length")) < 2.0 * max(1.0, _number(current.get("median_track_length"))):
        reasons.append("median_track_length_gain_too_small")
    if _number(candidate.get("short_track_ratio_le3")) >= 0.75 * _number(current.get("short_track_ratio_le3")):
        reasons.append("short_track_ratio_reduction_too_small")
    if _number(candidate.get("approx_fragmentation")) >= 0.75 * _number(current.get("approx_fragmentation")):
        reasons.append("fragmentation_reduction_too_small")
    if _number(candidate.get("approx_id_switches")) >= 0.75 * _number(current.get("approx_id_switches")):
        reasons.append("id_switch_reduction_too_small")
    purity_delta = _delta(candidate.get("local_purity_mean"), current.get("local_purity_mean"))
    if purity_delta is not None and purity_delta < -0.01:
        reasons.append("purity_drop_above_0.01")
    nonperson_delta = _delta(candidate.get("nonperson_short_track_ratio_le3"), current.get("nonperson_short_track_ratio_le3"))
    if nonperson_delta is not None and nonperson_delta > 0.05:
        reasons.append("non_person_degradation")
    false_merge_delta = _delta(candidate.get("false_merge_suspicion_rate"), current.get("false_merge_suspicion_rate"))
    if false_merge_delta is not None and false_merge_delta > 0.05:
        reasons.append("false_merge_suspicion_increase_above_0.05")
    runtime = _number(candidate.get("runtime_seconds"))
    if runtime > 3600.0:
        reasons.append("runtime_above_one_hour_for_precheck")
    return reasons


def _continuity_better(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    return (
        _number(candidate.get("median_track_length")) > _number(current.get("median_track_length"))
        and _number(candidate.get("short_track_ratio_le3")) < _number(current.get("short_track_ratio_le3"))
    )


def _find(rows: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("tracker_name") == name and row.get("status") == "ok":
            return row
    return None


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _delta(candidate: Any, baseline: Any) -> Optional[float]:
    try:
        return float(candidate) - float(baseline)
    except (TypeError, ValueError):
        return None


def _ratio(value: Any, baseline: Any) -> Optional[float]:
    try:
        denominator = float(baseline)
        return None if denominator <= 0.0 else float(value) / denominator
    except (TypeError, ValueError):
        return None
