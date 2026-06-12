"""Local comparison for frozen Track 1 upload candidates."""

from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_comparison import collect_run_metrics
from deep_oc_sort_3d.final_export.track1_final_checks import read_track1_txt
from deep_oc_sort_3d.freeze_upload.freeze_config import candidate_output_root, candidate_specs, output_root
from deep_oc_sort_3d.freeze_upload.freeze_io import progress_iter, read_json, write_csv, write_json


CLASS_NAMES = {
    0: "Person",
    1: "Forklift",
    2: "PalletTruck",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "NovaCarter",
}


def compare_frozen_candidates(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Build local summaries, comparison tables, readiness and verdict files."""
    summaries = {}
    per_track_rows = []
    for spec in candidate_specs(config):
        name = str(spec.get("candidate_name"))
        summary, track_rows = build_candidate_summary(config, spec, progress=progress)
        summaries[name] = summary
        per_track_rows.extend(track_rows)
        write_json(candidate_output_root(config, name) / "local_summary.json", summary)

    comparison_root = output_root(config) / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    v2 = summaries.get("v2_current", {})
    v3 = summaries.get("v3_gap_aware_soft", {})
    metrics = _comparison_metrics(v2, v3)
    verdict = _build_verdict(config, summaries)
    readiness = _build_upload_readiness(config, summaries)
    result = {
        "candidates": summaries,
        "metrics": metrics,
        "distribution_deltas": {
            "per_scene_rows": _delta_mapping(v2.get("per_scene_rows", {}), v3.get("per_scene_rows", {})),
            "per_class_rows": _delta_mapping(v2.get("per_class_rows", {}), v3.get("per_class_rows", {})),
        },
        "verdict": verdict,
        "upload_readiness": readiness,
        "recommendation": (
            "Upload V2 current and V3 gap-aware soft as separate submissions. "
            "Use the official evaluation server to decide the final ranking."
        ),
    }
    write_json(comparison_root / "v2_vs_v3_freeze_summary.json", result)
    write_csv(comparison_root / "v2_vs_v3_freeze_summary.csv", metrics, _metric_fields())
    write_csv(
        comparison_root / "per_scene_comparison.csv",
        _breakdown_rows(v2, v3, "per_scene_rows", "scene_id"),
        _breakdown_fields("scene_id"),
    )
    write_csv(
        comparison_root / "per_class_comparison.csv",
        _class_breakdown_rows(v2, v3),
        _breakdown_fields("class_id") + ["class_name"],
    )
    write_csv(
        comparison_root / "per_track_statistics.csv",
        per_track_rows,
        ["candidate_name", "scene_id", "class_id", "class_name", "object_id", "num_rows"],
    )
    write_json(comparison_root / "upload_readiness.json", readiness)
    write_json(comparison_root / "verdict.json", verdict)
    return result


def build_candidate_summary(
    config: Dict[str, Any],
    spec: Dict[str, Any],
    progress: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Summarize one frozen file and its already-produced tracking artifacts."""
    name = str(spec.get("candidate_name"))
    root = candidate_output_root(config, name)
    rows = read_track1_txt(root / "track1.txt")
    track_counts = defaultdict(int)
    per_scene_rows = defaultdict(int)
    per_class_rows = defaultdict(int)
    per_scene_class_rows = defaultdict(int)
    tracks_per_scene = defaultdict(set)
    tracks_per_class = defaultdict(set)
    for row in progress_iter(rows, progress, "summarize %s" % name):
        parsed = _parse_track_key(row)
        if parsed is None:
            continue
        scene_id, class_id, object_id = parsed
        track_counts[(scene_id, class_id, object_id)] += 1
        per_scene_rows[str(scene_id)] += 1
        per_class_rows[str(class_id)] += 1
        per_scene_class_rows["%d/%d" % (scene_id, class_id)] += 1
        tracks_per_scene[str(scene_id)].add((class_id, object_id))
        tracks_per_class[str(class_id)].add((scene_id, object_id))
    lengths = list(track_counts.values())
    length_summary = _numeric_summary(lengths)
    validation = read_json(root / "validation_summary.json")
    manifest = read_json(root / "manifest.json")
    tracking_metrics = _collect_tracking_metrics(spec)
    summary = {
        "candidate_name": name,
        "description": spec.get("description"),
        "frozen_track1_path": str(root / "track1.txt"),
        "track1_rows": len(rows),
        "unique_tracks": len(track_counts),
        "unique_scene_ids": sorted([int(value) for value in per_scene_rows.keys()]),
        "num_unique_scene_ids": len(per_scene_rows),
        "unique_classes": sorted([int(value) for value in per_class_rows.keys()]),
        "num_unique_classes": len(per_class_rows),
        "rows_per_track": length_summary,
        "rows_per_track_mean": length_summary.get("mean"),
        "rows_per_track_median": length_summary.get("median"),
        "rows_per_track_p25": length_summary.get("p25"),
        "rows_per_track_p75": length_summary.get("p75"),
        "rows_per_track_p90": length_summary.get("p90"),
        "per_scene_rows": dict(sorted(per_scene_rows.items())),
        "per_class_rows": dict(sorted(per_class_rows.items(), key=lambda item: int(item[0]))),
        "per_scene_class_rows": dict(sorted(per_scene_class_rows.items())),
        "scene_distribution": dict(sorted(per_scene_rows.items())),
        "class_distribution": dict(sorted(per_class_rows.items(), key=lambda item: int(item[0]))),
        "rows_per_scene_class": dict(sorted(per_scene_class_rows.items())),
        "tracks_per_scene": {key: len(value) for key, value in sorted(tracks_per_scene.items())},
        "tracks_per_class": {
            key: len(value) for key, value in sorted(tracks_per_class.items(), key=lambda item: int(item[0]))
        },
        "validation": {
            "status": validation.get("status"),
            "num_errors": validation.get("num_errors"),
            "checks": validation.get("checks", {}),
        },
        "sha256": manifest.get("sha256"),
        "size_bytes": manifest.get("file_size_bytes"),
        "tracking_metrics": tracking_metrics,
    }
    per_track = []
    for key, count in sorted(track_counts.items()):
        per_track.append(
            {
                "candidate_name": name,
                "scene_id": key[0],
                "class_id": key[1],
                "class_name": CLASS_NAMES.get(key[1], "class_%d" % key[1]),
                "object_id": key[2],
                "num_rows": count,
            }
        )
    return summary, per_track


def _collect_tracking_metrics(spec: Dict[str, Any]) -> Dict[str, Any]:
    missing_root = "output/__freeze_upload_missing__"
    values = collect_run_metrics(
        {
            "name": spec.get("candidate_name", ""),
            "pipeline_root": spec.get("pipeline_root", missing_root),
            "local_tracks_root": spec.get("local_tracks_root", missing_root),
            "tracklets_root": spec.get("tracklets_root", missing_root),
            "candidates_root": spec.get("candidates_root", missing_root),
            "motion_clean_root": spec.get("motion_clean_root", missing_root),
            "global_root": spec.get("global_root", missing_root),
            "final_export_root": missing_root,
            "track1_root": spec.get("source_track1_root", missing_root),
        }
    )
    global_association = values.get("global_association", {})
    if int(global_association.get("summary_files", 0) or 0) <= 0:
        global_association = {
            "availability": "not_available",
            "summary_files": 0,
            "global_tracks": None,
            "multi_camera_tracks": None,
            "singleton_tracks": None,
            "accepted_edges": None,
            "transition_edges_accepted": None,
            "fragmentation_approx": None,
            "global_purity_mean": None,
            "false_merge_rate": None,
        }
    else:
        global_association["availability"] = "available"
    return {
        "global_association": global_association,
        "final_export": values.get("final_export", {}),
    }


def _comparison_metrics(v2: Dict[str, Any], v3: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = {
        "track1_rows": ["track1_rows"],
        "unique_tracks": ["unique_tracks"],
        "rows_per_track_mean": ["rows_per_track", "mean"],
        "rows_per_track_median": ["rows_per_track", "median"],
        "global_tracks": ["tracking_metrics", "global_association", "global_tracks"],
        "multi_camera_tracks": ["tracking_metrics", "global_association", "multi_camera_tracks"],
        "singleton_tracks": ["tracking_metrics", "global_association", "singleton_tracks"],
        "accepted_edges": ["tracking_metrics", "global_association", "accepted_edges"],
        "transition_edges_accepted": ["tracking_metrics", "global_association", "transition_edges_accepted"],
        "global_purity_mean": ["tracking_metrics", "global_association", "global_purity_mean"],
        "false_merge_rate": ["tracking_metrics", "global_association", "false_merge_rate"],
        "fragmentation_approx": ["tracking_metrics", "global_association", "fragmentation_approx"],
        "validation_errors": ["validation", "num_errors"],
        "duplicate_keys": ["validation", "checks", "duplicate_key_count"],
        "nan_inf_count": ["validation", "checks", "nan_or_inf_values"],
        "non_positive_dimensions": ["validation", "checks", "non_positive_dimensions"],
    }
    rows = []
    for metric, path in paths.items():
        left = _nested(v2, path)
        right = _nested(v3, path)
        rows.append(
            {
                "metric": metric,
                "v2_current": left,
                "v3_gap_aware_soft": right,
                "delta_v3_minus_v2": _delta(left, right),
                "ratio_v3_over_v2": _ratio(right, left),
            }
        )
    return rows


def _build_upload_readiness(config: Dict[str, Any], summaries: Dict[str, Any]) -> Dict[str, Any]:
    packages_root = output_root(config) / "packages"
    package_manifest = read_json(packages_root / "package_manifest.json")
    package_rows = {
        str(row.get("candidate_name")): row
        for row in package_manifest.get("packages", [])
        if isinstance(row, dict)
    }
    candidates = {}
    for name, summary in summaries.items():
        validation = summary.get("validation", {})
        zip_path = packages_root / (name + "_track1.zip")
        frozen_path = candidate_output_root(config, name) / "track1.txt"
        package_row = package_rows.get(name, {})
        package_verified = package_row.get("status") == "ok" and package_row.get("verification", {}).get("valid") is True
        ready = (
            validation.get("status") == "ok"
            and frozen_path.exists()
            and frozen_path.stat().st_size > 0
            and zip_path.exists()
            and zip_path.stat().st_size > 0
            and package_verified
        )
        candidates[name] = {
            "ready": ready,
            "validation_status": validation.get("status"),
            "validation_errors": validation.get("num_errors"),
            "track1_path": str(frozen_path),
            "zip_path": str(zip_path),
            "zip_sha256": package_row.get("zip_sha256"),
            "package_verified": package_verified,
            "sha256": summary.get("sha256"),
            "line_count": summary.get("track1_rows"),
        }
    output = dict(candidates)
    output["status"] = "ok" if candidates and all(row.get("ready") for row in candidates.values()) else "not_ready"
    output["recommendation"] = "Upload both as separate submissions: V2 first, V3 second."
    output["upload_instruction"] = "Do not combine V2 and V3 into one archive."
    return output


def _build_verdict(config: Dict[str, Any], summaries: Dict[str, Any]) -> Dict[str, Any]:
    readiness = _build_upload_readiness(config, summaries)
    ready = [name for name in ["v2_current", "v3_gap_aware_soft"] if readiness.get(name, {}).get("ready")]
    frozen_missing = any(
        not (candidate_output_root(config, name) / "track1.txt").exists()
        for name in ["v2_current", "v3_gap_aware_soft"]
    )
    if frozen_missing:
        label = "freeze_invalid_missing_files"
        reasons = ["one_or_more_frozen_track1_files_missing"]
    elif len(ready) == 2:
        label = "both_candidates_ready_for_upload"
        reasons = ["both_track1_files_valid", "both_zip_packages_verified"]
    elif ready == ["v2_current"]:
        label = "v2_ready_v3_not_ready"
        reasons = ["v3_validation_or_package_check_failed"]
    elif ready == ["v3_gap_aware_soft"]:
        label = "v3_ready_v2_not_ready"
        reasons = ["v2_validation_or_package_check_failed"]
    else:
        label = "no_candidate_ready_fix_required"
        reasons = ["validation_or_package_check_failed"]
    return {
        "label": label,
        "reasons": reasons,
        "official_winner": None,
        "official_winner_reason": "Only the official evaluation server can rank the two candidates.",
        "recommended_action": "Upload V2 current first, then upload V3 gap_aware_soft as a separate submission.",
    }


def _breakdown_rows(
    v2: Dict[str, Any],
    v3: Dict[str, Any],
    metric: str,
    key_name: str,
) -> List[Dict[str, Any]]:
    left = v2.get(metric, {}) if isinstance(v2.get(metric), dict) else {}
    right = v3.get(metric, {}) if isinstance(v3.get(metric), dict) else {}
    keys = sorted(set(list(left.keys()) + list(right.keys())), key=_sort_key)
    output = []
    for key in keys:
        a = int(left.get(key, 0) or 0)
        b = int(right.get(key, 0) or 0)
        output.append(
            {
                key_name: key,
                "v2_current": a,
                "v3_gap_aware_soft": b,
                "delta_v3_minus_v2": b - a,
                "ratio_v3_over_v2": _ratio(b, a),
            }
        )
    return output


def _class_breakdown_rows(v2: Dict[str, Any], v3: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = _breakdown_rows(v2, v3, "per_class_rows", "class_id")
    for row in rows:
        try:
            class_id = int(row.get("class_id"))
        except (TypeError, ValueError):
            class_id = -1
        row["class_name"] = CLASS_NAMES.get(class_id, "class_%d" % class_id)
    return rows


def _delta_mapping(left: Any, right: Any) -> Dict[str, int]:
    left_values = left if isinstance(left, dict) else {}
    right_values = right if isinstance(right, dict) else {}
    keys = sorted(set(list(left_values.keys()) + list(right_values.keys())), key=_sort_key)
    return {
        str(key): int(right_values.get(key, 0) or 0) - int(left_values.get(key, 0) or 0)
        for key in keys
    }


def _parse_track_key(row: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    try:
        return int(float(row.get("scene_id"))), int(float(row.get("class_id"))), int(float(row.get("object_id")))
    except (TypeError, ValueError):
        return None


def _numeric_summary(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "median": median(ordered),
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(values: List[int], fraction: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = fraction * float(len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return float(values[low]) * (1.0 - weight) + float(values[high]) * weight


def _nested(data: Dict[str, Any], keys: List[str]) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _delta(left: Any, right: Any) -> Optional[float]:
    try:
        return float(right) - float(left)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        denominator_value = float(denominator)
        if denominator_value == 0.0:
            return None
        return float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None


def _sort_key(value: Any) -> Tuple[int, str]:
    try:
        return 0, "%012d" % int(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _metric_fields() -> List[str]:
    return ["metric", "v2_current", "v3_gap_aware_soft", "delta_v3_minus_v2", "ratio_v3_over_v2"]


def _breakdown_fields(key: str) -> List[str]:
    return [key, "v2_current", "v3_gap_aware_soft", "delta_v3_minus_v2", "ratio_v3_over_v2"]
