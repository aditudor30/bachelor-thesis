"""Compare baseline_v1_geometry_only with baseline_v2_pseudo3d_fullcam."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists


def compare_fullcam_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compare configured baseline V1 and fullcam V2 roots."""
    paths = config.get("paths", {})
    v1 = collect_run_metrics(
        {
            "name": "baseline_v1_geometry_only",
            "pipeline_root": paths.get("baseline_v1_pipeline_root", ""),
            "local_tracks_root": paths.get("baseline_v1_local_tracks_root", "output/local_tracks/yolo11m_medium_conf001"),
            "tracklets_root": paths.get("baseline_v1_tracklets_root", "output/tracklets/yolo11m_medium_conf001"),
            "candidates_root": paths.get("baseline_v1_candidates_root", "output/mtmc_candidates/yolo11m_medium_conf001"),
            "motion_clean_root": paths.get("baseline_v1_motion_clean_root", "output/mtmc_candidates_motion_clean/yolo11m_medium_conf001_mid_dense"),
            "global_root": paths.get("baseline_v1_global_root", "output/global_mtmc_transition/yolo11m_medium_conf001_transition"),
            "final_export_root": paths.get("baseline_v1_final_export_root", "output/final_mvp_exports/yolo11m_medium_conf001_transition"),
            "track1_root": paths.get("baseline_v1_track1_root", "output/track1_submission/yolo11m_medium_conf001_transition"),
        }
    )
    v2 = collect_run_metrics(
        {
            "name": "baseline_v2_pseudo3d_fullcam",
            "pipeline_root": paths.get("output_pipeline_root", "output/pipeline_runs/baseline_v2_pseudo3d_fullcam"),
            "local_tracks_root": paths.get("output_local_tracks_root", "output/local_tracks/baseline_v2_pseudo3d_fullcam"),
            "tracklets_root": paths.get("output_tracklets_root", "output/tracklets/baseline_v2_pseudo3d_fullcam"),
            "candidates_root": paths.get("output_candidates_root", "output/mtmc_candidates/baseline_v2_pseudo3d_fullcam"),
            "motion_clean_root": paths.get("output_motion_clean_root", "output/mtmc_candidates_motion_clean/baseline_v2_pseudo3d_fullcam"),
            "global_root": paths.get("output_global_root", "output/global_mtmc_transition/baseline_v2_pseudo3d_fullcam"),
            "final_export_root": paths.get("output_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam"),
            "track1_root": paths.get("output_track1_root", "output/track1_submission/baseline_v2_pseudo3d_fullcam"),
        }
    )
    comparison = {
        "baseline_v1": v1,
        "baseline_v2_fullcam": v2,
        "deltas": compute_metric_deltas(v1, v2),
    }
    comparison["verdict"] = decide_fullcam_verdict(comparison)
    return comparison


def collect_run_metrics(paths: Dict[str, Any]) -> Dict[str, Any]:
    """Collect available high-level metrics for one run."""
    pipeline_root = Path(str(paths.get("pipeline_root", "")))
    local_root = Path(str(paths.get("local_tracks_root", "")))
    tracklet_root = Path(str(paths.get("tracklets_root", "")))
    candidate_root = Path(str(paths.get("candidates_root", "")))
    motion_root = Path(str(paths.get("motion_clean_root", "")))
    global_root = Path(str(paths.get("global_root", "")))
    final_root = Path(str(paths.get("final_export_root", "")))
    track1_root = Path(str(paths.get("track1_root", "")))
    observations = read_json_if_exists(pipeline_root / "summaries" / "pseudo3d_observation_summary.json")
    validation = _first_json(
        [
            track1_root / "validation" / "track1_validation_summary.json",
            track1_root / "validation" / "final_validation_report.json",
            track1_root / "track1_validation.json",
            track1_root / "validation.json",
        ]
    )
    global_summary = _aggregate_global_summaries(global_root)
    final_summary = _final_export_summary(final_root)
    return {
        "name": paths.get("name", ""),
        "paths": {key: str(value) for key, value in paths.items()},
        "observations": _observation_metrics(observations),
        "local_tracking": _local_tracking_metrics(local_root),
        "tracklets": _summary_json(tracklet_root / "summaries" / "tracklet_summary.json"),
        "candidates": _summary_json(candidate_root / "summaries" / "candidate_summary.json"),
        "motion_clean": _motion_metrics(motion_root),
        "global_association": global_summary,
        "final_export": final_summary,
        "track1": {
            "track1_path": str(track1_root / "track1.txt"),
            "rows": _line_count(track1_root / "track1.txt"),
            "validation_errors": validation.get("num_errors"),
            "validation_status": validation.get("status"),
            "duplicate_key_count": validation.get("duplicate_key_count"),
            "sorting_issues": validation.get("sorting_issues"),
            "per_scene_rows": validation.get("per_scene_rows", {}),
            "per_class_rows": validation.get("per_class_rows", {}),
        },
    }


def compute_metric_deltas(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    """Compute compact deltas from v1 to v2."""
    return {
        "track1_rows_delta": _delta(_nested(v1, ["track1", "rows"]), _nested(v2, ["track1", "rows"])),
        "track1_validation_errors_delta": _delta(
            _nested(v1, ["track1", "validation_errors"]),
            _nested(v2, ["track1", "validation_errors"]),
        ),
        "pseudo3d_used_rate_delta": _delta(
            _nested(v1, ["observations", "pseudo3d_used_rate"]),
            _nested(v2, ["observations", "pseudo3d_used_rate"]),
        ),
        "fallback_original_used_rate_delta": _delta(
            _nested(v1, ["observations", "fallback_original_used_rate"]),
            _nested(v2, ["observations", "fallback_original_used_rate"]),
        ),
        "global_tracks_delta": _delta(
            _nested(v1, ["global_association", "global_tracks"]),
            _nested(v2, ["global_association", "global_tracks"]),
        ),
        "multi_camera_tracks_delta": _delta(
            _nested(v1, ["global_association", "multi_camera_tracks"]),
            _nested(v2, ["global_association", "multi_camera_tracks"]),
        ),
        "global_purity_mean_delta": _delta(
            _nested(v1, ["global_association", "global_purity_mean"]),
            _nested(v2, ["global_association", "global_purity_mean"]),
        ),
        "false_merge_rate_delta": _delta(
            _nested(v1, ["global_association", "false_merge_rate"]),
            _nested(v2, ["global_association", "false_merge_rate"]),
        ),
        "fragmentation_approx_delta": _delta(
            _nested(v1, ["global_association", "fragmentation_approx"]),
            _nested(v2, ["global_association", "fragmentation_approx"]),
        ),
    }


def decide_fullcam_verdict(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return final Step 15H verdict."""
    v2 = summary.get("baseline_v2_fullcam", {})
    track1_errors = _nested(v2, ["track1", "validation_errors"])
    pseudo_rate = _nested(v2, ["observations", "pseudo3d_used_rate"])
    fallback_rate = _nested(v2, ["observations", "fallback_original_used_rate"])
    metadata_rate = _nested(v2, ["observations", "metadata_complete_rate"])
    deltas = summary.get("deltas", {})
    reasons = []
    if track1_errors not in (0, "0"):
        return {"label": "baseline_v2_fullcam_invalid_fix_required", "reasons": ["track1_validation_errors"]}
    if pseudo_rate is None or float(pseudo_rate) < 0.95:
        return {"label": "baseline_v2_fullcam_invalid_fix_required", "reasons": ["pseudo3d_used_rate_below_0.95"]}
    if fallback_rate is not None and float(fallback_rate) >= 0.05:
        reasons.append("fallback_original_used_rate_not_below_0.05")
    if metadata_rate is not None and float(metadata_rate) < 0.95:
        reasons.append("metadata_completeness_below_0.95")
    purity_delta = deltas.get("global_purity_mean_delta")
    fragmentation_delta = deltas.get("fragmentation_approx_delta")
    if purity_delta is not None and float(purity_delta) < -0.01:
        return {"label": "baseline_v2_fullcam_needs_tracking_tuning", "reasons": ["global_purity_drop"] + reasons}
    if fragmentation_delta is not None and float(fragmentation_delta) > 50:
        return {"label": "baseline_v2_fullcam_needs_tracking_tuning", "reasons": ["fragmentation_increase"] + reasons}
    if reasons:
        return {"label": "baseline_v2_fullcam_valid_but_not_submission_candidate", "reasons": reasons}
    return {"label": "baseline_v2_fullcam_ready_for_submission", "reasons": ["track1_valid_and_pseudo3d_fullcam_coverage_ok"]}


def write_fullcam_comparison_outputs(summary: Dict[str, Any], output_root: Path) -> None:
    """Write comparison JSON/CSV diagnostics and verdict."""
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(summary, output_root / "baseline_v1_vs_v2_fullcam_summary.json")
    _write_metric_csv(summary, output_root / "baseline_v1_vs_v2_fullcam_summary.csv")
    _write_dict_csv(summary.get("deltas", {}), output_root / "metric_deltas.csv")
    _write_json(summary.get("verdict", {}), output_root / "verdict.json")
    _write_breakdown_csv(summary, "subset", output_root / "per_subset_comparison.csv")
    _write_breakdown_csv(summary, "scene", output_root / "per_scene_comparison.csv")
    _write_breakdown_csv(summary, "class", output_root / "per_class_comparison.csv")
    diagnostics = output_root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    _write_json(_nested(summary.get("baseline_v2_fullcam", {}), ["observations"], {}), diagnostics / "pseudo3d_usage_fullcam.json")
    _write_json(_nested(summary.get("baseline_v2_fullcam", {}), ["track1"], {}), diagnostics / "track1_validation_summary.json")
    _write_json({}, diagnostics / "projection_fullcam_audit.json")
    _write_json({}, diagnostics / "smoothness_fullcam_comparison.json")


def _observation_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    total = int(data.get("output_observations", 0) or 0)
    pseudo = int(data.get("pseudo3d_used", 0) or 0)
    fallback = int(data.get("fallback_original_used", 0) or 0)
    metadata = data.get("source_metadata_completeness_fullcam", data.get("source_metadata_completeness", {}))
    metadata_rate = None
    if isinstance(metadata, dict):
        metadata_rate = metadata.get("overall_required_metadata_complete_rate")
    return {
        "output_observations": total,
        "pseudo3d_used": pseudo,
        "pseudo3d_used_rate": data.get("pseudo3d_used_rate", _rate(pseudo, total)),
        "fallback_original_used": fallback,
        "fallback_original_used_rate": data.get("fallback_original_used_rate", _rate(fallback, total)),
        "no_3d_records": data.get("no_3d_records"),
        "metadata_complete_rate": metadata_rate,
        "source_metadata_completeness": metadata,
        "center_3d_source_distribution": data.get("per_source", {}),
    }


def _local_tracking_metrics(root: Path) -> Dict[str, Any]:
    rows = _read_csv(root / "summaries" / "local_tracking_summary.csv")
    return {
        "files": len(rows),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
        "local_track_records": sum(_safe_int(row.get("num_track_records")) for row in rows),
        "active_tracks": sum(_safe_int(row.get("num_active_tracks")) for row in rows),
    }


def _motion_metrics(root: Path) -> Dict[str, Any]:
    data = _summary_json(root / "summaries" / "motion_quality_summary.json")
    if not data:
        data = _summary_json(root / "summaries" / "candidate_motion_summary.json")
    return data


def _aggregate_global_summaries(root: Path) -> Dict[str, Any]:
    aggregate = {
        "summary_files": 0,
        "global_tracks": 0,
        "multi_camera_tracks": 0,
        "singleton_tracks": 0,
        "accepted_edges": 0,
        "transition_edges_accepted": 0,
        "fragmentation_approx": 0,
        "per_class_tracks": {},
        "per_class_multi_camera_tracks": {},
        "global_purity_values": [],
        "false_merge_values": [],
    }
    transition_relation_found = False
    for path in sorted(root.rglob("summary.json")):
        if "summaries" in set(path.parts):
            continue
        data = read_json_if_exists(path)
        aggregate["summary_files"] += 1
        aggregate["global_tracks"] += _safe_int(data.get("global_tracks", data.get("num_global_tracks")))
        aggregate["multi_camera_tracks"] += _safe_int(data.get("multi_camera_tracks", data.get("num_multi_camera_tracks")))
        aggregate["singleton_tracks"] += _safe_int(data.get("singleton_tracks", data.get("num_singleton_tracks")))
        aggregate["accepted_edges"] += _safe_int(data.get("accepted_edges"))
        relations = data.get("accepted_edge_temporal_relations", {})
        if isinstance(relations, dict):
            for key, value in relations.items():
                if str(key) != "overlap":
                    aggregate["transition_edges_accepted"] += _safe_int(value)
                    transition_relation_found = True
        _merge_count_dict(aggregate["per_class_tracks"], data.get("per_class_tracks", {}))
        _merge_count_dict(
            aggregate["per_class_multi_camera_tracks"],
            data.get("per_class_multi_camera_tracks", data.get("per_class_multi_camera", {})),
        )
        metrics = data.get("diagnostic_gt_metrics", {})
        if isinstance(metrics, dict):
            if metrics.get("global_purity_mean") is not None:
                aggregate["global_purity_values"].append(float(metrics.get("global_purity_mean")))
            if metrics.get("false_merge_rate") is not None:
                aggregate["false_merge_values"].append(float(metrics.get("false_merge_rate")))
            aggregate["fragmentation_approx"] += _safe_int(metrics.get("fragmentation_approx"))
    if not transition_relation_found:
        aggregate["transition_edges_accepted"] = _count_accepted_transition_edges(root)
    aggregate["global_purity_mean"] = _mean(aggregate.pop("global_purity_values"))
    aggregate["false_merge_rate"] = _mean(aggregate.pop("false_merge_values"))
    return aggregate


def _final_export_summary(root: Path) -> Dict[str, Any]:
    generic_root = root / "generic_tracking_export"
    frame_root = root / "frame_global_records"
    return {
        "generic_rows": _csv_row_count(generic_root),
        "frame_record_rows": _csv_row_count(frame_root),
        "validation": _summary_json(root / "validation" / "export_validation.json"),
    }


def _first_json(paths: List[Path]) -> Dict[str, Any]:
    for path in paths:
        data = read_json_if_exists(path)
        if data:
            return data
    return {}


def _summary_json(path: Path) -> Dict[str, Any]:
    return read_json_if_exists(path)


def _line_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _csv_row_count(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in sorted(root.rglob("*.csv")):
        total += len(_read_csv(path))
    return total


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_metric_csv(summary: Dict[str, Any], path: Path) -> None:
    rows = []
    for section_name in ["baseline_v1", "baseline_v2_fullcam"]:
        section = summary.get(section_name, {})
        rows.extend(_flatten_rows(section_name, section))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_dict_csv(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in data.items():
            writer.writerow({"metric": key, "value": value})


def _write_breakdown_csv(summary: Dict[str, Any], name: str, path: Path) -> None:
    rows = _breakdown_rows(summary, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["breakdown", "source", "key", "baseline_v1", "baseline_v2_fullcam", "delta"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _breakdown_rows(summary: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    if name == "scene":
        return _dict_breakdown_rows(
            name,
            "track1_rows",
            _nested(summary, ["baseline_v1", "track1", "per_scene_rows"], {}),
            _nested(summary, ["baseline_v2_fullcam", "track1", "per_scene_rows"], {}),
        )
    if name == "class":
        rows = _dict_breakdown_rows(
            name,
            "track1_rows",
            _nested(summary, ["baseline_v1", "track1", "per_class_rows"], {}),
            _nested(summary, ["baseline_v2_fullcam", "track1", "per_class_rows"], {}),
        )
        rows.extend(
            _dict_breakdown_rows(
                name,
                "global_tracks",
                _nested(summary, ["baseline_v1", "global_association", "per_class_tracks"], {}),
                _nested(summary, ["baseline_v2_fullcam", "global_association", "per_class_tracks"], {}),
            )
        )
        return rows
    return _aggregate_breakdown_rows(summary, name)


def _dict_breakdown_rows(name: str, source: str, v1: Dict[str, Any], v2: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    if not isinstance(v1, dict):
        v1 = {}
    if not isinstance(v2, dict):
        v2 = {}
    keys = sorted(set([str(key) for key in v1.keys()] + [str(key) for key in v2.keys()]))
    for key in keys:
        left = _safe_float(v1.get(key))
        right = _safe_float(v2.get(key))
        rows.append(
            {
                "breakdown": name,
                "source": source,
                "key": key,
                "baseline_v1": left,
                "baseline_v2_fullcam": right,
                "delta": _delta(left, right),
            }
        )
    if rows:
        return rows
    return [
        {
            "breakdown": name,
            "source": source,
            "key": "not_available",
            "baseline_v1": "",
            "baseline_v2_fullcam": "",
            "delta": "",
        }
    ]


def _aggregate_breakdown_rows(summary: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    metric_paths = {
        "track1_rows": ["track1", "rows"],
        "track1_validation_errors": ["track1", "validation_errors"],
        "global_tracks": ["global_association", "global_tracks"],
        "multi_camera_tracks": ["global_association", "multi_camera_tracks"],
        "global_purity_mean": ["global_association", "global_purity_mean"],
        "fragmentation_approx": ["global_association", "fragmentation_approx"],
        "pseudo3d_used_rate": ["observations", "pseudo3d_used_rate"],
        "fallback_original_used_rate": ["observations", "fallback_original_used_rate"],
    }
    rows = []
    for key, metric_path in metric_paths.items():
        left = _nested(summary.get("baseline_v1", {}), metric_path)
        right = _nested(summary.get("baseline_v2_fullcam", {}), metric_path)
        rows.append(
            {
                "breakdown": name,
                "source": "aggregate_metric",
                "key": key,
                "baseline_v1": left,
                "baseline_v2_fullcam": right,
                "delta": _delta(left, right),
            }
        )
    return rows


def _flatten_rows(prefix: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            for child in _flatten_rows("%s.%s" % (prefix, key), value):
                rows.append(child)
        elif not isinstance(value, list):
            rows.append({"scope": prefix, "metric": key, "value": value})
    return rows


def _nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current.get(key)
    return current


def _delta(a: Any, b: Any) -> Any:
    if a in (None, "") or b in (None, ""):
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def _rate(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        denom = float(denominator)
        if denom <= 0.0:
            return None
        return float(numerator) / denom
    except (TypeError, ValueError):
        return None


def _mean(values: List[float]) -> Any:
    if not values:
        return None
    return sum(values) / float(len(values))


def _merge_count_dict(target: Dict[str, Any], data: Any) -> None:
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        target[str(key)] = _safe_int(target.get(str(key))) + _safe_int(value)


def _count_accepted_transition_edges(root: Path) -> int:
    csv_files = sorted(root.rglob("transition_edges.csv"))
    jsonl_files = sorted(root.rglob("transition_edges.jsonl"))
    if csv_files:
        return _count_accepted_transition_edges_csv(csv_files)
    return _count_accepted_transition_edges_jsonl(jsonl_files)


def _count_accepted_transition_edges_csv(paths: List[Path]) -> int:
    count = 0
    for path in paths:
        if "summaries" in set(path.parts):
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if _truthy(row.get("accepted")):
                    count += 1
    return count


def _count_accepted_transition_edges_jsonl(paths: List[Path]) -> int:
    count = 0
    for path in paths:
        if "summaries" in set(path.parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if _truthy(data.get("accepted")):
                count += 1
    return count


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in ("true", "1", "yes")


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
