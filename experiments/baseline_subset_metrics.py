"""Read-only metric collectors for baseline subset comparisons."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from deep_oc_sort_3d.audit3d.audit3d_io import numeric_stats, read_json_if_exists
from deep_oc_sort_3d.experiments.sixcam_subset import SixCamItem


def collect_observation_metrics_for_subset(root: Union[str, Path], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect Observation3D metrics for the selected cameras."""
    rows = []
    for item in subset_items:
        path = _path_with_optional_observations(root, item, "%s.jsonl" % item.camera_id)
        records = _read_jsonl(path)
        rows.append(_observation_row(item, path, records))
    return _summary("observations", version_name, rows, ["num_observations", "pseudo3d_used", "fallback_original_used", "no_3d_records"])


def collect_local_tracking_metrics_for_subset(root: Union[str, Path], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect local tracking metrics for the selected cameras."""
    rows = []
    for item in subset_items:
        path = Path(root) / item.subset / item.scene_name / ("%s.csv" % item.camera_id)
        records = _read_csv(path)
        rows.append(_track_records_row(item, path, records))
    return _summary("local_tracking", version_name, rows, ["num_records", "active_tracks"])


def collect_tracklet_metrics_for_subset(root: Union[str, Path], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect local tracklet metrics for the selected cameras."""
    rows = []
    for item in subset_items:
        path = _first_existing([Path(root) / item.subset / item.scene_name / ("%s_tracklets.csv" % item.camera_id), Path(root) / item.subset / item.scene_name / ("%s_tracklets.jsonl" % item.camera_id)])
        records = _read_csv_or_jsonl(path)
        rows.append(_tracklet_row(item, path, records))
    return _summary("tracklets", version_name, rows, ["total_tracklets", "valid_tracklets"])


def collect_candidate_metrics_for_subset(root: Union[str, Path], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect MTMC candidate metrics for the selected cameras."""
    rows = []
    for item in subset_items:
        path = _first_existing([Path(root) / item.subset / item.scene_name / ("%s_candidates.csv" % item.camera_id), Path(root) / item.subset / item.scene_name / ("%s_candidates.jsonl" % item.camera_id)])
        records = _read_csv_or_jsonl(path)
        rows.append(_candidate_row(item, path, records))
    return _summary("candidates", version_name, rows, ["total_candidates", "kept_candidates", "rejected_candidates"])


def collect_motion_clean_metrics_for_subset(root: Union[str, Path], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect motion-clean metrics for selected cameras."""
    rows = []
    for item in subset_items:
        scene_root = Path(root) / item.subset / item.scene_name
        metrics_path = scene_root / ("%s_motion_metrics.csv" % item.camera_id)
        records = _read_csv(metrics_path)
        rows.append(_motion_row(item, metrics_path, records, scene_root))
    return _summary("motion_clean", version_name, rows, ["total_candidates", "motion_good", "motion_suspicious", "motion_invalid", "motion_unknown"])


def collect_global_association_metrics_for_subset(root: Union[str, Path], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect scene-level global association metrics for the selected scenes."""
    seen = set()
    rows = []
    for item in subset_items:
        scene_key = (item.subset, item.scene_name)
        if scene_key in seen:
            continue
        seen.add(scene_key)
        summary_path = Path(root) / item.subset / item.scene_name / "summary.json"
        rows.append(_global_row(item, summary_path, read_json_if_exists(summary_path)))
    return _summary("global_association", version_name, rows, ["global_tracks", "accepted_edges", "transition_edges_accepted"])


def collect_final_export_metrics_for_subset(root: Union[str, Path], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect final frame-level export metrics for selected cameras."""
    rows = []
    for item in subset_items:
        path = Path(root) / "frame_global_records" / item.subset / item.scene_name / ("%s_global_records.csv" % item.camera_id)
        records = _read_csv(path)
        rows.append(_final_export_row(item, path, records))
    return _summary("final_export", version_name, rows, ["rows", "unique_tracks"])


def collect_all_subset_metrics(paths: Dict[str, Any], subset_items: List[SixCamItem], version_name: str) -> Dict[str, Any]:
    """Collect all available subset metrics for one baseline version."""
    return {
        "version_name": version_name,
        "observations": collect_observation_metrics_for_subset(paths.get("pipeline_root", ""), subset_items, version_name),
        "local_tracking": collect_local_tracking_metrics_for_subset(paths.get("local_tracks_root", ""), subset_items, version_name),
        "tracklets": collect_tracklet_metrics_for_subset(paths.get("tracklets_root", ""), subset_items, version_name),
        "candidates": collect_candidate_metrics_for_subset(paths.get("candidates_root", ""), subset_items, version_name),
        "motion_clean": collect_motion_clean_metrics_for_subset(paths.get("motion_clean_root", ""), subset_items, version_name),
        "global_association": collect_global_association_metrics_for_subset(paths.get("global_root", ""), subset_items, version_name),
        "final_export": collect_final_export_metrics_for_subset(paths.get("final_export_root", ""), subset_items, version_name),
    }


def _observation_row(item: SixCamItem, path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    pseudo_used = sum(1 for row in records if _bool(row.get("pseudo3d_used")))
    fallback = sum(1 for row in records if _bool(row.get("fallback_original_used")))
    no_3d = sum(1 for row in records if not _bool(row.get("has_3d", row.get("center_3d") not in (None, "", []))))
    return _base_row(item, path, total, {
        "num_observations": total,
        "pseudo3d_used": pseudo_used,
        "pseudo3d_used_rate": _rate(pseudo_used, total),
        "fallback_original_used": fallback,
        "fallback_original_used_rate": _rate(fallback, total),
        "no_3d_records": no_3d,
        "center_3d_source_distribution": _count(records, "center_3d_source"),
        "dimensions_3d_source_distribution": _count(records, "dimensions_3d_source"),
        "yaw_source_distribution": _count(records, "yaw_source"),
        "depth_source_distribution": _count(records, "depth_source"),
        "metadata_completeness": _source_metadata_completeness(records),
    })


def _track_records_row(item: SixCamItem, path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = _lengths_by_key(records, "local_track_id")
    return _base_row(item, path, len(records), {
        "num_records": len(records),
        "active_tracks": len(lengths),
        "track_length_stats": numeric_stats(list(lengths.values())),
        "matched_gt_records": sum(1 for row in records if _bool(row.get("matched_gt"))),
    })


def _tracklet_row(item: SixCamItem, path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = sum(1 for row in records if _bool(row.get("is_valid_for_mtmc")))
    lengths = [_optional_float(row.get("length")) for row in records]
    purity = [_optional_float(row.get("gt_purity")) for row in records]
    return _base_row(item, path, len(records), {
        "total_tracklets": len(records),
        "valid_tracklets": valid,
        "valid_tracklet_rate": _rate(valid, len(records)),
        "quality_flag_distribution": _count(records, "quality_flag"),
        "length_stats": numeric_stats(lengths),
        "purity_stats": numeric_stats(purity),
        "per_class": _count(records, "class_id"),
    })


def _candidate_row(item: SixCamItem, path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = sum(1 for row in records if _bool(row.get("is_candidate", True)))
    total = len(records)
    return _base_row(item, path, total, {
        "total_candidates": total,
        "kept_candidates": kept,
        "rejected_candidates": total - kept,
        "kept_rate": _rate(kept, total),
        "length_stats": numeric_stats([_optional_float(row.get("length")) for row in records]),
        "mean_confidence_stats": numeric_stats([_optional_float(row.get("mean_confidence")) for row in records]),
        "has_3d_count": sum(1 for row in records if _bool(row.get("has_3d"))),
        "quality_flag_distribution": _count(records, "quality_flag"),
        "per_class": _count(records, "class_id"),
    })


def _motion_row(item: SixCamItem, path: Path, records: List[Dict[str, Any]], scene_root: Path) -> Dict[str, Any]:
    good = sum(1 for row in records if str(row.get("motion_quality_flag")) == "motion_good")
    suspicious = sum(1 for row in records if str(row.get("motion_quality_flag")) == "motion_suspicious")
    invalid = sum(1 for row in records if str(row.get("motion_quality_flag")) == "motion_invalid")
    unknown = sum(1 for row in records if str(row.get("motion_quality_flag")) == "motion_unknown")
    clean_rows = _read_csv(scene_root / ("%s_clean_candidates.csv" % item.camera_id))
    return _base_row(item, path, len(records), {
        "total_candidates": len(records),
        "clean_count": len(clean_rows),
        "motion_good": good,
        "motion_suspicious": suspicious,
        "motion_invalid": invalid,
        "motion_unknown": unknown,
        "invalid_rate": _rate(invalid, len(records)),
        "step_p95_stats": numeric_stats([_optional_float(row.get("p95_step_distance_3d")) for row in records]),
        "step_max_stats": numeric_stats([_optional_float(row.get("max_step_distance_3d")) for row in records]),
        "jump_count_stats": numeric_stats([_optional_float(row.get("jump_count")) for row in records]),
        "jump_ratio_stats": numeric_stats([_optional_float(row.get("jump_ratio")) for row in records]),
    })


def _global_row(item: SixCamItem, path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    return _base_row(item, path, 1 if data else 0, {
        "global_tracks": _first_value(data, ["num_global_tracks", "global_tracks", "tracks"], None),
        "multi_camera_tracks": _first_value(data, ["multi_camera_tracks", "num_multi_camera_tracks"], "not_available"),
        "singleton_tracks": _first_value(data, ["singleton_tracks", "num_singleton_tracks"], "not_available"),
        "accepted_edges": _first_value(data, ["accepted_edges", "num_accepted_edges"], "not_available"),
        "transition_edges_accepted": _first_value(data, ["transition_edges_accepted", "num_transition_edges_accepted"], "not_available"),
        "global_purity_mean": _nested_first_value(data, ["diagnostic_gt_metrics"], ["global_purity_mean", "purity_mean"]),
    })


def _final_export_row(item: SixCamItem, path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = _lengths_by_key(records, "global_track_id")
    return _base_row(item, path, len(records), {
        "rows": len(records),
        "unique_tracks": len(lengths),
        "rows_per_track_stats": numeric_stats(list(lengths.values())),
        "scene_distribution": _count(records, "scene_name"),
        "class_distribution": _count(records, "class_id"),
        "duplicate_frame_track_keys": _duplicate_count(records, ["scene_name", "camera_id", "frame_id", "global_track_id"]),
        "non_positive_dimensions": _non_positive_dimensions(records),
    })


def _summary(section: str, version_name: str, rows: List[Dict[str, Any]], sum_fields: List[str]) -> Dict[str, Any]:
    summary = {
        "section": section,
        "version_name": version_name,
        "camera_count": len(rows),
        "missing_files": sum(1 for row in rows if row.get("status") == "missing"),
        "rows": rows,
        "detail_rows": rows,
    }
    for field in sum_fields:
        summary[field] = sum(_safe_int(row.get(field, 0)) for row in rows)
    if summary.get("num_observations") is not None:
        summary["pseudo3d_used_rate"] = _rate(summary.get("pseudo3d_used", 0), summary.get("num_observations", 0))
        summary["fallback_original_used_rate"] = _rate(summary.get("fallback_original_used", 0), summary.get("num_observations", 0))
    if summary.get("total_tracklets") is not None:
        summary["valid_tracklet_rate"] = _rate(summary.get("valid_tracklets", 0), summary.get("total_tracklets", 0))
    if summary.get("total_candidates") is not None:
        summary["kept_rate"] = _rate(summary.get("kept_candidates", 0), summary.get("total_candidates", 0))
        summary["invalid_rate"] = _rate(summary.get("motion_invalid", 0), summary.get("total_candidates", 0))
    return summary


def _base_row(item: SixCamItem, path: Path, record_count: int, values: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "subset": item.subset,
        "split": item.split,
        "scene_name": item.scene_name,
        "camera_id": item.camera_id,
        "path": str(path),
        "status": "ok" if path.exists() else "missing",
        "record_count": record_count,
    }
    row.update(values)
    return row


def _path_with_optional_observations(root: Union[str, Path], item: SixCamItem, filename: str) -> Path:
    root_path = Path(root)
    if (root_path / "observations3d").exists():
        root_path = root_path / "observations3d"
    return root_path / item.subset / item.scene_name / filename


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_csv_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path is None:
        return []
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return _read_csv(path)


def _first_existing(paths: List[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _count(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        value = row.get(field)
        key = _count_key(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_key(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str) and value == "":
        return "unknown"
    if isinstance(value, (list, dict, tuple)):
        try:
            return json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _lengths_by_key(rows: List[Dict[str, Any]], key_field: str) -> Dict[str, int]:
    lengths = {}
    for row in rows:
        key = str(row.get(key_field, ""))
        if key in ("", "None"):
            continue
        lengths[key] = lengths.get(key, 0) + 1
    return lengths


def _source_metadata_completeness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    fields = ["center_3d_source", "dimensions_3d_source", "yaw_source", "depth_source", "pseudo3d_method", "pseudo3d_version"]
    out = {"total": total}
    for field in fields:
        count = sum(1 for row in rows if row.get(field) not in (None, "", "unknown"))
        out["%s_complete" % field] = count
        out["%s_complete_rate" % field] = _rate(count, total)
    return out


def _duplicate_count(rows: List[Dict[str, Any]], fields: List[str]) -> int:
    seen = set()
    duplicate = 0
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            duplicate += 1
        seen.add(key)
    return duplicate


def _non_positive_dimensions(rows: List[Dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        dims = [_optional_float(row.get("width_3d")), _optional_float(row.get("length_3d")), _optional_float(row.get("height_3d"))]
        if any(value is not None and value <= 0.0 for value in dims):
            count += 1
    return count


def _first_value(data: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for key in keys:
        if key in data:
            return data.get(key)
    return default


def _nested_first_value(data: Dict[str, Any], parents: List[str], keys: List[str]) -> Any:
    for parent in parents:
        nested = data.get(parent, {})
        if isinstance(nested, dict):
            value = _first_value(nested, keys, None)
            if value is not None:
                return value
    return "not_available"


def _rate(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        denominator_value = float(denominator)
        if denominator_value == 0.0:
            return None
        return float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
