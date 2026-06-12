"""Metric collection for ByteTrack coverage tuning variants and baselines."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_comparison import collect_run_metrics
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_comparison import is_local_track_csv
from deep_oc_sort_3d.bytetrack_tuning.tuning_config import tuning_output_root, variant_root
from deep_oc_sort_3d.bytetrack_tuning.tuning_io import progress_iter, read_json, write_json
from deep_oc_sort_3d.local_tracker_benchmark.gt_local_diagnostics import compute_gt_diagnostics
from deep_oc_sort_3d.local_tracker_benchmark.local_track_metrics import compute_track_metrics


STAGE_NAMES = [
    "observations",
    "local_records",
    "local_tracks",
    "tracklets",
    "valid_tracklets",
    "mtmc_candidates",
    "motion_clean_candidates",
    "global_tracks",
    "multi_camera_tracks",
    "final_export_rows",
    "track1_rows",
]


def collect_all_tuning_metrics(
    config: Dict[str, Any],
    names: Optional[List[str]] = None,
    include_baselines: bool = True,
    progress: bool = True,
) -> Dict[str, Any]:
    """Collect all available variant and baseline metrics."""
    variant_names = names or sorted(config.get("variants", {}).keys())
    variants = {}
    for name in progress_iter(variant_names, progress, "collect ByteTrack tuning metrics"):
        variants[str(name)] = collect_variant_metrics(config, str(name))
    baselines = collect_baseline_metrics(config) if include_baselines else {}
    output = {"variants": variants, "baselines": baselines}
    write_json(tuning_output_root(config) / "comparison" / "raw_tuning_metrics.json", output)
    return output


def collect_variant_metrics(config: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    """Collect local-to-Track1 metrics for one isolated variant."""
    root = variant_root(config, variant_name)
    return _collect_run(
        name=variant_name,
        observations_root=Path(str(config.get("paths", {}).get("v2_observations_root"))),
        local_root=root / "local_tracks",
        tracklets_root=root / "tracklets",
        candidates_root=root / "candidates",
        motion_root=root / "motion_clean",
        global_root=root / "global_mtmc",
        final_root=root / "final_export",
        track1_root=root / "track1_submission",
        validation_root=root / "validation",
        variant_root_path=root,
    )


def collect_baseline_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect V2 current, 21B and V1 using the same metric schema."""
    paths = config.get("paths", {})
    observations_root = Path(str(paths.get("v2_observations_root")))
    specs = {
        "baseline_v2_current": {
            "local": paths.get("baseline_v2_local_tracks_root"),
            "tracklets": paths.get("baseline_v2_tracklets_root"),
            "candidates": paths.get("baseline_v2_candidates_root"),
            "motion": paths.get("baseline_v2_motion_clean_root"),
            "global": paths.get("baseline_v2_global_root"),
            "final": paths.get("baseline_v2_final_export_root"),
            "track1": paths.get("baseline_v2_track1_root"),
        },
        "baseline_21b_bytetrack": {
            "local": paths.get("baseline_21b_local_tracks_root"),
            "tracklets": paths.get("baseline_21b_tracklets_root"),
            "candidates": paths.get("baseline_21b_candidates_root"),
            "motion": paths.get("baseline_21b_motion_clean_root"),
            "global": paths.get("baseline_21b_global_root"),
            "final": paths.get("baseline_21b_final_export_root"),
            "track1": paths.get("baseline_21b_track1_root"),
        },
        "baseline_v1_geometry_only": {
            "local": paths.get("baseline_v1_local_tracks_root"),
            "tracklets": paths.get("baseline_v1_tracklets_root"),
            "candidates": paths.get("baseline_v1_candidates_root"),
            "motion": paths.get("baseline_v1_motion_clean_root"),
            "global": paths.get("baseline_v1_global_root"),
            "final": paths.get("baseline_v1_final_export_root"),
            "track1": paths.get("baseline_v1_track1_root"),
        },
    }
    output = {}
    phase_a_scenes = _configured_scenes(config, "sweep_eval")
    for name, spec in specs.items():
        output[name] = _collect_run(
            name=name,
            observations_root=observations_root,
            local_root=_path(spec.get("local")),
            tracklets_root=_path(spec.get("tracklets")),
            candidates_root=_path(spec.get("candidates")),
            motion_root=_path(spec.get("motion")),
            global_root=_path(spec.get("global")),
            final_root=_path(spec.get("final")),
            track1_root=_path(spec.get("track1")),
            validation_root=_path(spec.get("track1")) / "validation",
            variant_root_path=None,
        )
        output[name]["phase_a_local_tracking"] = collect_local_metrics(
            _path(spec.get("local")),
            scene_filter=phase_a_scenes,
        )
        phase_a_observations = count_jsonl_records(observations_root, scene_filter=phase_a_scenes)
        output[name]["phase_a_stage_counts"] = {
            "observations": phase_a_observations.get("total", 0),
            "local_records": output[name]["phase_a_local_tracking"].get("num_records", 0),
            "local_tracks": output[name]["phase_a_local_tracking"].get("num_tracks", 0),
        }
        output[name]["phase_a_dimensions"] = {
            "observations": phase_a_observations,
            "local_records": output[name]["phase_a_local_tracking"].get("dimensions", {}),
        }
    return output


def _collect_run(
    name: str,
    observations_root: Path,
    local_root: Path,
    tracklets_root: Path,
    candidates_root: Path,
    motion_root: Path,
    global_root: Path,
    final_root: Path,
    track1_root: Path,
    validation_root: Path,
    variant_root_path: Optional[Path],
) -> Dict[str, Any]:
    local = collect_local_metrics(local_root)
    tracklets = read_json(tracklets_root / "summaries" / "tracklet_summary.json")
    candidates = read_json(candidates_root / "summaries" / "candidate_summary.json")
    motion = read_json(motion_root / "summaries" / "motion_quality_summary.json")
    high_level = collect_run_metrics(
        {
            "name": name,
            "pipeline_root": observations_root.parent,
            "local_tracks_root": local_root,
            "tracklets_root": tracklets_root,
            "candidates_root": candidates_root,
            "motion_clean_root": motion_root,
            "global_root": global_root,
            "final_export_root": final_root,
            "track1_root": track1_root,
        }
    )
    validation = read_json(validation_root / "track1_validation_summary.json")
    if not validation:
        validation = read_json(track1_root / "validation" / "track1_validation_summary.json")
    observed_scenes = sorted(local.get("dimensions", {}).get("per_scene", {}).keys())
    observations = count_jsonl_records(observations_root, scene_filter=observed_scenes or None)
    global_metrics = dict(high_level.get("global_association", {}))
    per_class_fragmentation = collect_global_fragmentation_by_class(global_root)
    global_metrics["per_class_fragmentation"] = per_class_fragmentation
    global_metrics["person_fragmentation"] = int(per_class_fragmentation.get("Person", 0) or 0)
    global_metrics["non_person_fragmentation"] = sum(
        int(value or 0) for key, value in per_class_fragmentation.items() if str(key) != "Person"
    )
    final_metrics = high_level.get("final_export", {})
    track1_metrics = high_level.get("track1", {})
    stage_counts = {
        "observations": observations.get("total", 0),
        "local_records": local.get("num_records", 0),
        "local_tracks": local.get("num_tracks", 0),
        "tracklets": int(tracklets.get("total_tracklets", 0) or 0),
        "valid_tracklets": int(tracklets.get("valid_tracklets", 0) or 0),
        "mtmc_candidates": int(candidates.get("kept_candidates", 0) or 0),
        "motion_clean_candidates": int(motion.get("clean_count", 0) or 0),
        "global_tracks": int(global_metrics.get("global_tracks", 0) or 0),
        "multi_camera_tracks": int(global_metrics.get("multi_camera_tracks", 0) or 0),
        "final_export_rows": int(final_metrics.get("generic_rows", 0) or 0),
        "track1_rows": int(track1_metrics.get("rows", 0) or 0),
    }
    tracklet_dimensions = collect_tracklet_dimensions(tracklets_root, valid_only=False)
    valid_tracklet_dimensions = collect_tracklet_dimensions(tracklets_root, valid_only=True)
    motion_dimensions = collect_tabular_dimensions(motion_root, "*_clean_candidates.csv")
    global_dimensions = collect_global_dimensions(global_root)
    final_dimensions = collect_tabular_dimensions(final_root / "generic_tracking_export", "*.csv")
    return {
        "name": name,
        "status": _run_status(local_root, variant_root_path),
        "runtime_seconds": _runtime_seconds(variant_root_path),
        "stage_counts": stage_counts,
        "local_tracking": local,
        "tracklets": tracklets,
        "candidates": candidates,
        "motion_clean": motion,
        "global_association": global_metrics,
        "final_export": final_metrics,
        "track1": {
            "rows": stage_counts["track1_rows"],
            "validation_errors": validation.get("num_errors", track1_metrics.get("validation_errors")),
            "validation_status": validation.get("status", track1_metrics.get("validation_status")),
            "duplicate_keys": _check_value(validation, "duplicate_key_count"),
            "sorting_issues": _check_value(validation, "sorting_issues"),
            "nan_inf_count": _check_value(validation, "nan_or_inf_values"),
            "non_positive_dimensions": _check_value(validation, "non_positive_dimensions"),
            "per_scene_rows": _distribution(validation, "per_scene_rows", track1_metrics.get("per_scene_rows", {})),
            "per_class_rows": _distribution(validation, "per_class_rows", track1_metrics.get("per_class_rows", {})),
        },
        "dimensions": {
            "observations": observations,
            "local_records": local.get("dimensions", {}),
            "tracklets": tracklet_dimensions,
            "valid_tracklets": valid_tracklet_dimensions,
            "mtmc_candidates": _candidate_dimensions(candidates),
            "motion_clean_candidates": motion_dimensions,
            "global_tracks": global_dimensions,
            "final_export_rows": final_dimensions,
            "track1_rows": {
                "per_scene": _distribution(validation, "per_scene_rows", track1_metrics.get("per_scene_rows", {})),
                "per_class": _distribution(validation, "per_class_rows", track1_metrics.get("per_class_rows", {})),
            },
        },
    }


def collect_local_metrics(root: Path, scene_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """Aggregate local metrics while loading one camera CSV at a time."""
    all_lengths = []
    all_confidences = []
    all_durations = []
    gt_values = []
    total_records = 0
    per_scene = {}
    per_camera = {}
    per_class = {}
    person_records = 0
    non_person_records = 0
    files = 0
    allowed_scenes = None if scene_filter is None else set(scene_filter)
    for path in (sorted(root.rglob("*.csv")) if root.exists() else []):
        if not is_local_track_csv(path):
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        normalized = [_normalize_local_row(row, root, path) for row in rows]
        if allowed_scenes is not None:
            normalized = [row for row in normalized if str(row.get("scene_name", "")) in allowed_scenes]
        if not normalized:
            continue
        files += 1
        metric = compute_track_metrics(normalized)
        gt = compute_gt_diagnostics(normalized)
        gt_values.append((metric, gt))
        total_records += len(normalized)
        all_confidences.extend(float(row.get("confidence", 0.0)) for row in normalized)
        groups = {}
        for row in normalized:
            key = (row.get("class_id"), row.get("track_id"))
            groups.setdefault(key, []).append(row)
            _increment(per_scene, str(row.get("scene_name", "")))
            _increment(per_camera, str(row.get("camera_id", "")))
            _increment(per_class, str(row.get("class_name", row.get("class_id", ""))))
            if int(float(row.get("class_id", -1))) == 0:
                person_records += 1
            else:
                non_person_records += 1
        for values in groups.values():
            frames = [int(float(row.get("frame_id", 0))) for row in values]
            all_lengths.append(len(values))
            all_durations.append(max(frames) - min(frames) + 1 if frames else 0)
    return {
        "files": files,
        "num_records": total_records,
        "num_tracks": len(all_lengths),
        "mean_track_length": _mean(all_lengths),
        "median_track_length": _percentile(all_lengths, 50),
        "p25_track_length": _percentile(all_lengths, 25),
        "p75_track_length": _percentile(all_lengths, 75),
        "p90_track_length": _percentile(all_lengths, 90),
        "num_length_1_tracks": _count_at_most(all_lengths, 1),
        "num_length_le_3_tracks": _count_at_most(all_lengths, 3),
        "num_length_le_5_tracks": _count_at_most(all_lengths, 5),
        "short_track_ratio_len1": _ratio_at_most(all_lengths, 1),
        "short_track_ratio_le3": _ratio_at_most(all_lengths, 3),
        "short_track_ratio_le5": _ratio_at_most(all_lengths, 5),
        "mean_confidence": _mean(all_confidences),
        "median_confidence": _percentile(all_confidences, 50),
        "track_duration_mean": _mean(all_durations),
        "track_duration_median": _percentile(all_durations, 50),
        "approx_id_switches": sum(int(gt.get("approx_id_switches", 0)) for _metric, gt in gt_values),
        "approx_fragmentation": sum(int(gt.get("approx_fragmentation", 0)) for _metric, gt in gt_values),
        "gt_matched_records": sum(int(gt.get("gt_matched_records", 0)) for _metric, gt in gt_values),
        "local_purity_mean": _weighted(gt_values, "local_purity_mean", "gt_matched_records"),
        "local_purity_median": _weighted(gt_values, "local_purity_median", "gt_matched_records"),
        "false_merge_suspicion_rate": _weighted(gt_values, "false_merge_suspicion_rate", "num_tracks"),
        "person_records": person_records,
        "non_person_records": non_person_records,
        "dimensions": {
            "per_scene": per_scene,
            "per_camera": per_camera,
            "per_class": per_class,
            "person_vs_nonperson": {"Person": person_records, "NonPerson": non_person_records},
        },
    }


def count_jsonl_records(root: Path, scene_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """Count observation JSONL records and common dimensions incrementally."""
    total = 0
    per_scene = {}
    per_camera = {}
    per_class = {}
    person = 0
    non_person = 0
    allowed_scenes = None if scene_filter is None else set(scene_filter)
    for path in (sorted(root.rglob("*.jsonl")) if root.exists() else []):
        relative = path.relative_to(root)
        scene_name = relative.parts[1] if len(relative.parts) > 2 else ""
        if allowed_scenes is not None and scene_name not in allowed_scenes:
            continue
        camera_id = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                _increment(per_scene, scene_name)
                _increment(per_camera, camera_id)
                try:
                    row = json.loads(line)
                except ValueError:
                    row = {}
                class_name = str(row.get("class_name", row.get("class_id", "unknown")))
                _increment(per_class, class_name)
                if int(row.get("class_id", -1)) == 0:
                    person += 1
                else:
                    non_person += 1
    return {
        "total": total,
        "per_scene": per_scene,
        "per_camera": per_camera,
        "per_class": per_class,
        "person_vs_nonperson": {"Person": person, "NonPerson": non_person},
    }


def collect_global_fragmentation_by_class(root: Path) -> Dict[str, int]:
    """Aggregate per-class fragmentation from scene-level global summaries."""
    output = {}
    for path in (sorted(root.rglob("summary.json")) if root.exists() else []):
        if "summaries" in set(path.parts):
            continue
        summary = read_json(path)
        diagnostic = summary.get("diagnostic_gt_metrics", {})
        values = diagnostic.get("per_class_fragmentation", {}) if isinstance(diagnostic, dict) else {}
        for key, value in (values.items() if isinstance(values, dict) else []):
            output[str(key)] = output.get(str(key), 0) + int(value or 0)
    return output


def collect_tracklet_dimensions(root: Path, valid_only: bool) -> Dict[str, Any]:
    """Count tracklets by common dimensions, optionally keeping valid tracklets only."""
    output = _empty_dimensions()
    for path in (sorted(root.rglob("*_tracklets.csv")) if root.exists() else []):
        for row in _read_csv(path):
            if valid_only and str(row.get("is_valid_for_mtmc", "")).lower() not in ("true", "1", "yes"):
                continue
            _increment_dimensions(output, row, path)
    return output


def collect_tabular_dimensions(root: Path, pattern: str) -> Dict[str, Any]:
    """Count rows by scene, camera, class and Person grouping."""
    output = _empty_dimensions()
    for path in (sorted(root.rglob(pattern)) if root.exists() else []):
        if "summaries" in set(path.parts):
            continue
        for row in _read_csv(path):
            _increment_dimensions(output, row, path)
    return output


def collect_global_dimensions(root: Path) -> Dict[str, Any]:
    """Aggregate scene/class/camera global-track counts from scene summaries."""
    output = _empty_dimensions()
    for path in (sorted(root.rglob("summary.json")) if root.exists() else []):
        if "summaries" in set(path.parts):
            continue
        summary = read_json(path)
        scene_name = path.parent.name
        global_tracks = int(summary.get("global_tracks", summary.get("num_global_tracks", 0)) or 0)
        output["per_scene"][scene_name] = output["per_scene"].get(scene_name, 0) + global_tracks
        for key, value in summary.get("per_class_tracks", {}).items():
            output["per_class"][str(key)] = output["per_class"].get(str(key), 0) + int(value or 0)
        for key, value in summary.get("per_camera_participation", {}).items():
            output["per_camera"][str(key)] = output["per_camera"].get(str(key), 0) + int(value or 0)
    output["person_vs_nonperson"] = _person_group_from_classes(output["per_class"])
    return output


def _normalize_local_row(row: Dict[str, Any], root: Path, path: Path) -> Dict[str, Any]:
    relative = path.relative_to(root)
    subset = relative.parts[0] if len(relative.parts) > 2 else str(row.get("split", ""))
    output = dict(row)
    output["subset"] = subset
    output["track_id"] = row.get("local_track_id")
    return output


def _summary_dimensions(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "per_scene": summary.get("per_scene", {}),
        "per_camera": summary.get("per_camera", {}),
        "per_class": summary.get("per_class", {}),
    }


def _candidate_dimensions(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "per_scene": summary.get("per_scene_counts", {}),
        "per_camera": summary.get("per_camera_counts", {}),
        "per_class": summary.get("per_class_kept_counts", {}),
    }


def _motion_dimensions(summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = summary.get("files", [])
    per_scene = {}
    per_camera = {}
    for row in (rows if isinstance(rows, list) else []):
        count = int(row.get("clean_count", 0) or 0)
        input_path = Path(str(row.get("input_path", "")))
        parts = list(input_path.parts)
        scene = parts[-2] if len(parts) >= 2 else ""
        camera = input_path.stem.replace("_candidates", "")
        per_scene[scene] = per_scene.get(scene, 0) + count
        per_camera[camera] = per_camera.get(camera, 0) + count
    return {"per_scene": per_scene, "per_camera": per_camera, "per_class": {}}


def _global_dimensions(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {"per_class": summary.get("per_class_tracks", {}), "per_scene": {}, "per_camera": {}}


def _empty_dimensions() -> Dict[str, Any]:
    return {"per_scene": {}, "per_camera": {}, "per_class": {}, "person_vs_nonperson": {}}


def _increment_dimensions(output: Dict[str, Any], row: Dict[str, Any], path: Path) -> None:
    scene_name = str(row.get("scene_name", ""))
    camera_id = str(row.get("camera_id", ""))
    class_name = str(row.get("class_name", row.get("class_id", "unknown")))
    if not scene_name and len(path.parts) >= 2:
        scene_name = path.parent.name
    if not camera_id:
        camera_id = path.stem.split("_")[0] if path.stem else ""
    _increment(output["per_scene"], scene_name)
    _increment(output["per_camera"], camera_id)
    _increment(output["per_class"], class_name)
    group = "Person" if class_name == "Person" or str(row.get("class_id", "")) == "0" else "NonPerson"
    _increment(output["person_vs_nonperson"], group)


def _person_group_from_classes(values: Dict[str, Any]) -> Dict[str, int]:
    person = int(values.get("Person", values.get("0", 0)) or 0)
    total = sum(int(value or 0) for value in values.values())
    return {"Person": person, "NonPerson": total - person}


def _distribution(report: Dict[str, Any], key: str, fallback: Any) -> Dict[str, Any]:
    direct = report.get(key)
    if isinstance(direct, dict):
        return direct
    distribution = report.get("distribution", {})
    nested = distribution.get(key) if isinstance(distribution, dict) else None
    return nested if isinstance(nested, dict) else (fallback if isinstance(fallback, dict) else {})


def _check_value(report: Dict[str, Any], key: str) -> Any:
    if report.get(key) is not None:
        return report.get(key)
    checks = report.get("checks", {})
    return checks.get(key) if isinstance(checks, dict) else None


def _run_status(local_root: Path, root: Optional[Path]) -> str:
    if root is not None:
        full = read_json(root / "summaries" / "full_run.json")
        local = read_json(root / "summaries" / "local_sweep_run.json")
        if full:
            return str(full.get("status", "unknown"))
        if local:
            return "phase_a_only" if local.get("status") == "ok" else str(local.get("status"))
    return "ok" if local_root.exists() else "missing"


def _runtime_seconds(root: Optional[Path]) -> Optional[float]:
    if root is None:
        return None
    full = read_json(root / "summaries" / "full_run.json")
    local = read_json(root / "summaries" / "local_sweep_run.json")
    value = full.get("runtime_seconds") if full else local.get("runtime_seconds")
    return None if value is None else float(value)


def _configured_scenes(config: Dict[str, Any], phase: str) -> List[str]:
    output = []
    groups = config.get("subsets", {}).get(phase, {})
    for payload in (groups.values() if isinstance(groups, dict) else []):
        if isinstance(payload, dict):
            output.extend(str(value) for value in payload.get("scenes", []) or [])
    return sorted(set(output))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _path(value: Any) -> Path:
    return Path(str(value or ""))


def _increment(values: Dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1


def _mean(values: List[Any]) -> Optional[float]:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: List[Any], percentile: float) -> Optional[float]:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _count_at_most(values: List[int], threshold: int) -> int:
    return sum(1 for value in values if int(value) <= threshold)


def _ratio_at_most(values: List[int], threshold: int) -> Optional[float]:
    return None if not values else float(_count_at_most(values, threshold)) / float(len(values))


def _weighted(values: List[Tuple[Dict[str, Any], Dict[str, Any]]], metric_key: str, weight_key: str) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for metric, diagnostic in values:
        value = diagnostic.get(metric_key)
        weight = diagnostic.get(weight_key, metric.get(weight_key, 0))
        if value is None:
            continue
        weight_value = max(1.0, float(weight or 0.0))
        numerator += float(value) * weight_value
        denominator += weight_value
    return None if denominator <= 0.0 else numerator / denominator
