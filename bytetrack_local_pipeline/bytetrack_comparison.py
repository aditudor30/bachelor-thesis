"""Collect and compare V1, V2 current and V2 ByteTrack-local metrics."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_comparison import collect_run_metrics
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import read_json, write_csv, write_json
from deep_oc_sort_3d.local_tracker_benchmark.gt_local_diagnostics import compute_gt_diagnostics
from deep_oc_sort_3d.local_tracker_benchmark.local_track_metrics import compute_track_metrics
from deep_oc_sort_3d.tracking.track_io import read_local_tracks_csv


def compare_bytetrack_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build comprehensive comparison and honest final verdict."""
    paths = config.get("paths", {})
    variants = {
        "baseline_v1_geometry_only": _collect_variant(
            "baseline_v1_geometry_only",
            paths.get("yolo_pipeline_root"),
            paths.get("baseline_v1_local_tracks_root", "output/local_tracks/yolo11m_medium_conf001"),
            paths.get("baseline_v1_tracklets_root", "output/tracklets/yolo11m_medium_conf001"),
            paths.get("baseline_v1_candidates_root", "output/mtmc_candidates/yolo11m_medium_conf001"),
            paths.get("baseline_v1_motion_clean_root", "output/mtmc_candidates_motion_clean/yolo11m_medium_conf001_mid_dense"),
            paths.get("baseline_v1_global_root"),
            paths.get("baseline_v1_final_export_root"),
            paths.get("baseline_v1_track1_root"),
        ),
        "baseline_v2_pseudo3d_fullcam": _collect_variant(
            "baseline_v2_pseudo3d_fullcam",
            str(Path(str(paths.get("v2_observations_root"))).parent),
            paths.get("current_local_tracks_root"),
            paths.get("baseline_v2_tracklets_root", "output/tracklets/baseline_v2_pseudo3d_fullcam"),
            paths.get("baseline_v2_candidates_root", "output/mtmc_candidates/baseline_v2_pseudo3d_fullcam"),
            paths.get("baseline_v2_motion_clean_root", "output/mtmc_candidates_motion_clean/baseline_v2_pseudo3d_fullcam"),
            paths.get("baseline_v2_global_root"),
            paths.get("baseline_v2_final_export_root"),
            paths.get("baseline_v2_track1_root"),
        ),
        "baseline_v2_pseudo3d_fullcam_bytetrack_local": _collect_variant(
            "baseline_v2_pseudo3d_fullcam_bytetrack_local",
            str(Path(str(paths.get("v2_observations_root"))).parent),
            paths.get("output_local_tracks_root"),
            paths.get("output_tracklets_root"),
            paths.get("output_candidates_root"),
            paths.get("output_motion_clean_root"),
            paths.get("output_global_root"),
            paths.get("output_final_export_root"),
            paths.get("output_track1_root"),
        ),
    }
    current = variants["baseline_v2_pseudo3d_fullcam"]
    candidate = variants["baseline_v2_pseudo3d_fullcam_bytetrack_local"]
    deltas_v2 = metric_deltas(current, candidate)
    deltas_v1 = metric_deltas(variants["baseline_v1_geometry_only"], candidate)
    combined = _combined_safe_metrics(paths.get("combined_safe_root"))
    deltas_combined = metric_deltas(combined, candidate) if combined else {}
    verdict = decide_final_verdict(candidate, current, deltas_v2, config)
    return {
        "variants": variants,
        "combined_safe_080": combined,
        "metric_deltas_vs_v1": deltas_v1,
        "metric_deltas_vs_v2_current": deltas_v2,
        "metric_deltas_vs_combined_safe_080": deltas_combined,
        "verdict": verdict,
    }


def metric_deltas(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compute named candidate-minus-baseline deltas."""
    metrics = {
        "local_median_track_length": ["local_tracking", "median_track_length"],
        "local_short_track_ratio_le3": ["local_tracking", "short_track_ratio_le3"],
        "local_fragmentation": ["local_tracking", "approx_fragmentation"],
        "local_id_switches": ["local_tracking", "approx_id_switches"],
        "local_purity_mean": ["local_tracking", "local_purity_mean"],
        "local_records": ["local_tracking", "num_records"],
        "gt_matched_records": ["local_tracking", "gt_matched_records"],
        "global_tracks": ["global_association", "global_tracks"],
        "multi_camera_tracks": ["global_association", "multi_camera_tracks"],
        "global_purity_mean": ["global_association", "global_purity_mean"],
        "false_merge_rate": ["global_association", "false_merge_rate"],
        "global_fragmentation": ["global_association", "fragmentation_approx"],
        "track1_rows": ["track1", "rows"],
        "track1_validation_errors": ["track1", "validation_errors"],
    }
    output = {}
    for name, path in metrics.items():
        output[name + "_delta"] = _delta(_nested(baseline, path), _nested(candidate, path))
    return output


def decide_final_verdict(
    candidate: Dict[str, Any],
    current: Dict[str, Any],
    deltas: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Choose one requested final verdict using validation and global safeguards."""
    errors = _nested(candidate, ["track1", "validation_errors"])
    if errors not in (0, "0"):
        return {"label": "baseline_v2_bytetrack_local_invalid_fix_required", "reasons": ["track1_validation_errors"]}
    selection = config.get("selection", {})
    purity_delta = deltas.get("global_purity_mean_delta")
    false_delta = deltas.get("false_merge_rate_delta")
    if purity_delta is not None and purity_delta < -float(selection.get("max_allowed_purity_drop", 0.01)):
        return {"label": "baseline_v2_bytetrack_local_valid_but_false_merges_too_high", "reasons": ["global_purity_drop"]}
    if false_delta is not None and false_delta > float(selection.get("max_allowed_false_merge_rate_delta", 0.01)):
        return {"label": "baseline_v2_bytetrack_local_valid_but_false_merges_too_high", "reasons": ["false_merge_rate_increase"]}
    local_fragment_delta = deltas.get("local_fragmentation_delta")
    global_fragment_delta = deltas.get("global_fragmentation_delta")
    if local_fragment_delta is not None and local_fragment_delta < 0:
        if global_fragment_delta is not None and global_fragment_delta < 0:
            return {"label": "baseline_v2_bytetrack_local_ready_for_submission_candidate", "reasons": ["local_and_global_fragmentation_improved"]}
        return {"label": "baseline_v2_bytetrack_local_valid_improves_tracking_needs_global_tuning", "reasons": ["local_gain_not_propagated_globally"]}
    return {"label": "baseline_v2_bytetrack_local_valid_but_not_better_than_v2_current", "reasons": ["no_clear_local_fragmentation_gain"]}


def write_comparison_outputs(summary: Dict[str, Any], output_root: Path) -> None:
    """Write requested JSON/CSV artifacts."""
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "baseline_v2_bytetrack_full_summary.json", summary)
    write_csv(output_root / "baseline_v2_bytetrack_full_summary.csv", _flatten_variants(summary.get("variants", {})))
    precheck = summary.get("precheck", {})
    write_json(output_root / "bytetrack_local_precheck_summary.json", precheck)
    write_csv(output_root / "bytetrack_local_precheck_summary.csv", _dict_rows(precheck))
    write_csv(output_root / "metric_deltas_vs_v1.csv", _dict_rows(summary.get("metric_deltas_vs_v1", {})))
    write_csv(output_root / "metric_deltas_vs_v2_current.csv", _dict_rows(summary.get("metric_deltas_vs_v2_current", {})))
    write_csv(output_root / "metric_deltas_vs_combined_safe_080.csv", _dict_rows(summary.get("metric_deltas_vs_combined_safe_080", {})))
    write_json(output_root / "verdict.json", summary.get("verdict", {}))
    _write_breakdowns(summary, output_root)


def _collect_variant(
    name: str,
    pipeline_root: Any,
    local_root: Any,
    tracklets_root: Any,
    candidates_root: Any,
    motion_root: Any,
    global_root: Any,
    final_root: Any,
    track1_root: Any,
) -> Dict[str, Any]:
    metrics = collect_run_metrics(
        {
            "name": name,
            "pipeline_root": pipeline_root,
            "local_tracks_root": local_root,
            "tracklets_root": tracklets_root,
            "candidates_root": candidates_root,
            "motion_clean_root": motion_root,
            "global_root": global_root,
            "final_export_root": final_root,
            "track1_root": track1_root,
        }
    )
    metrics["local_tracking"] = collect_local_tracking_metrics(Path(str(local_root)))
    return metrics


def collect_local_tracking_metrics(root: Path) -> Dict[str, Any]:
    """Aggregate camera-local metrics while holding one camera in memory."""
    lengths = []
    confidences = []
    total_records = 0
    diagnostics = []
    person_lengths = []
    nonperson_lengths = []
    per_camera = []
    for path in sorted(root.rglob("*.csv")) if root.exists() else []:
        if "summaries" in set(path.parts):
            continue
        records = read_local_tracks_csv(path)
        rows = [_record_row(record, root, path) for record in records]
        metric = compute_track_metrics(rows)
        gt = compute_gt_diagnostics(rows)
        diagnostics.append((metric, gt))
        camera_metric = dict(metric)
        camera_metric.update(gt)
        camera_metric.update(
            {
                "subset": rows[0].get("subset") if rows else "",
                "scene_name": rows[0].get("scene_name") if rows else "",
                "camera_id": rows[0].get("camera_id") if rows else path.stem,
            }
        )
        per_camera.append(camera_metric)
        total_records += len(rows)
        grouped = {}
        for row in rows:
            grouped.setdefault((str(row.get("class_id")), str(row.get("track_id"))), []).append(row)
        for (class_id, _track_id), values in grouped.items():
            length = len(values)
            lengths.append(length)
            if int(class_id) == 0:
                person_lengths.append(length)
            else:
                nonperson_lengths.append(length)
        confidences.extend(float(row.get("confidence", 0.0)) for row in rows)
    return {
        "num_records": total_records,
        "num_tracks": len(lengths),
        "mean_track_length": _mean(lengths),
        "median_track_length": _percentile(lengths, 50),
        "short_track_ratio_len1": _ratio_count(lengths, 1),
        "short_track_ratio_le3": _ratio_count(lengths, 3),
        "short_track_ratio_le5": _ratio_count(lengths, 5),
        "mean_confidence": _mean(confidences),
        "approx_id_switches": sum(int(gt.get("approx_id_switches", 0)) for _metric, gt in diagnostics),
        "approx_fragmentation": sum(int(gt.get("approx_fragmentation", 0)) for _metric, gt in diagnostics),
        "gt_matched_records": sum(int(gt.get("gt_matched_records", 0)) for _metric, gt in diagnostics),
        "local_purity_mean": _weighted_diagnostic(diagnostics, "local_purity_mean", "gt_matched_records"),
        "false_merge_suspicion_rate": _weighted_diagnostic(diagnostics, "false_merge_suspicion_rate", "num_tracks"),
        "person_mean_track_length": _mean(person_lengths),
        "person_median_track_length": _percentile(person_lengths, 50),
        "non_person_mean_track_length": _mean(nonperson_lengths),
        "non_person_median_track_length": _percentile(nonperson_lengths, 50),
        "per_camera": per_camera,
    }


def _record_row(record: Any, root: Path, path: Path) -> Dict[str, Any]:
    relative = path.relative_to(root)
    subset = relative.parts[0] if len(relative.parts) > 2 else ""
    return {
        "subset": subset,
        "scene_name": record.scene_name,
        "camera_id": record.camera_id,
        "class_id": record.class_id,
        "track_id": record.local_track_id,
        "frame_id": record.frame_id,
        "confidence": record.confidence,
        "matched_gt_object_id": record.matched_gt_object_id,
    }


def _combined_safe_metrics(root_value: Any) -> Dict[str, Any]:
    if root_value in (None, ""):
        return {}
    root = Path(str(root_value))
    candidates = list(root.rglob("summary.json")) + list(root.rglob("*.summary.json"))
    return read_json(candidates[0]) if candidates else {}


def _flatten_variants(variants: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for variant, payload in variants.items():
        for section, values in payload.items():
            if not isinstance(values, dict):
                continue
            for metric, value in values.items():
                if not isinstance(value, (dict, list)):
                    rows.append({"variant": variant, "section": section, "metric": metric, "value": value})
    return rows


def _write_breakdowns(summary: Dict[str, Any], output_root: Path) -> None:
    variants = summary.get("variants", {})
    rows = []
    for name, payload in variants.items():
        track1 = payload.get("track1", {})
        for scene, count in track1.get("per_scene_rows", {}).items():
            rows.append({"variant": name, "key": scene, "rows": count})
    write_csv(output_root / "per_scene_comparison.csv", rows)
    rows = []
    for name, payload in variants.items():
        track1 = payload.get("track1", {})
        for class_id, count in track1.get("per_class_rows", {}).items():
            rows.append({"variant": name, "key": class_id, "rows": count})
    write_csv(output_root / "per_class_comparison.csv", rows)
    rows = []
    for name, payload in variants.items():
        for camera in payload.get("local_tracking", {}).get("per_camera", []):
            row = dict(camera)
            row["variant"] = name
            rows.append(row)
    write_csv(output_root / "per_camera_comparison.csv", rows)


def _dict_rows(values: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"metric": key, "value": value} for key, value in values.items()]


def _nested(data: Dict[str, Any], path: Sequence[str]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _delta(left: Any, right: Any) -> Optional[float]:
    try:
        return float(right) - float(left)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: Sequence[float], value: float) -> Optional[float]:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), value))


def _ratio_count(values: Sequence[int], threshold: int) -> Optional[float]:
    if not values:
        return None
    return float(sum(1 for value in values if int(value) <= threshold)) / float(len(values))


def _weighted_diagnostic(values: Sequence[Any], metric_key: str, weight_key: str) -> Optional[float]:
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
