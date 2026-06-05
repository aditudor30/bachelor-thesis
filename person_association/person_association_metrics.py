"""Metrics for Person-aware association experiments."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.final_export.export_eval import evaluate_global_frame_records
from deep_oc_sort_3d.final_export.generic_export import read_global_frame_records_file
from deep_oc_sort_3d.person_association.person_association_io import (
    count_by,
    frame_record_csv_files,
    generic_csv_files,
    mean,
    percentile,
    read_csv_rows,
    read_json,
    row_track_key,
    safe_int,
    write_json,
)


def collect_person_association_metrics(
    run_name: str,
    final_export_root: Path,
    track1_root: Path,
    merge_summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect comparable metrics for one association run."""
    generic_rows = _load_generic_rows(final_export_root / "generic_tracking_export")
    person_rows = [row for row in generic_rows if safe_int(row.get("class_id"), -1) == 0]
    non_person_rows = [row for row in generic_rows if safe_int(row.get("class_id"), -1) != 0]
    frame_records = _load_frame_records(final_export_root / "frame_global_records")
    global_eval = evaluate_global_frame_records(frame_records) if frame_records else {}
    frame_rows = _load_frame_rows(final_export_root / "frame_global_records")
    person_diag = person_gt_diagnostic([row for row in frame_rows if safe_int(row.get("class_id"), -1) == 0])
    person_lengths = _rows_per_track(person_rows)
    track1_summary = read_json(track1_root / "track1_export_summary.json") or {}
    track1_rows = track1_summary.get("rows_written")
    if track1_rows is None:
        track1_rows = _count_text_rows(track1_root / "track1.txt")
    validation = _track1_validation(track1_root)
    merge_summary = read_json(merge_summary_path) if merge_summary_path is not None else {}
    if merge_summary is None:
        merge_summary = {}
    metrics = {
        "run_name": run_name,
        "final_export_root": str(final_export_root),
        "track1_root": str(track1_root),
        "generic_rows": len(generic_rows),
        "person_rows": len(person_rows),
        "non_person_rows": len(non_person_rows),
        "track1_rows": track1_rows,
        "track1_validation_status": validation.get("status"),
        "track1_validation_errors": validation.get("num_errors"),
        "person_unique_tracks": len(person_lengths),
        "person_rows_per_track_mean": mean(list(person_lengths.values())),
        "person_rows_per_track_median": percentile(list(person_lengths.values()), 50),
        "person_rows_per_track_p95": percentile(list(person_lengths.values()), 95),
        "person_singleton_tracks": len([value for value in person_lengths.values() if value <= 1]),
        "person_short_tracks_lte_3": len([value for value in person_lengths.values() if value <= 3]),
        "person_purity": person_diag.get("person_purity"),
        "person_false_merge_rate": person_diag.get("person_false_merge_rate"),
        "person_fragmentation_approx": person_diag.get("person_fragmentation_approx"),
        "multi_camera_tracks": _multi_camera_tracks(generic_rows),
        "per_class_rows": count_by(generic_rows, "class_name"),
        "per_scene_rows": count_by(generic_rows, "scene_name"),
        "global_purity_mean": global_eval.get("global_id_purity_mean"),
        "false_merge_rate": _false_merge_rate(global_eval),
        "fragmentation_approx": global_eval.get("fragmentation_approx"),
        "global_unique_tracks": global_eval.get("unique_global_tracks"),
        "applied_merge_mapping_size": merge_summary.get("mapping_size"),
        "applied_merge_components": merge_summary.get("merged_components"),
        "selected_edges_before_conflict_filter": merge_summary.get("selected_edges_before_conflict_filter"),
        "candidate_rows": merge_summary.get("candidate_rows"),
    }
    return metrics


def person_gt_diagnostic(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Person-specific GT diagnostic from matched_gt_object_id fields."""
    gt_rows = [row for row in rows if row.get("matched_gt_object_id") not in (None, "")]
    by_global: Dict[str, List[Dict[str, Any]]] = {}
    gt_to_global: Dict[str, set] = {}
    for row in gt_rows:
        gid = str(row.get("global_track_id", ""))
        gt = str(row.get("matched_gt_object_id", ""))
        if not gid or not gt:
            continue
        by_global.setdefault(gid, []).append(row)
        gt_to_global.setdefault(gt, set()).add(gid)
    purities = []
    false_merges = 0
    for grouped in by_global.values():
        counts: Dict[str, int] = {}
        for row in grouped:
            gt = str(row.get("matched_gt_object_id", ""))
            counts[gt] = counts.get(gt, 0) + 1
        if counts:
            purities.append(float(max(counts.values())) / float(sum(counts.values())))
        if len(counts) > 1:
            false_merges += 1
    fragmentation = sum([max(0, len(values) - 1) for values in gt_to_global.values()])
    return {
        "person_records_with_gt": len(gt_rows),
        "person_purity": mean(purities),
        "person_false_merge_count": false_merges,
        "person_false_merge_rate": float(false_merges) / float(len(by_global)) if by_global else None,
        "person_fragmentation_approx": fragmentation,
    }


def compute_association_deltas(run: Dict[str, Any], baseline: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Compute deltas against a baseline metrics row."""
    keys = [
        "generic_rows",
        "person_rows",
        "non_person_rows",
        "track1_rows",
        "person_unique_tracks",
        "person_fragmentation_approx",
        "person_purity",
        "person_false_merge_rate",
        "global_purity_mean",
        "false_merge_rate",
        "fragmentation_approx",
        "multi_camera_tracks",
    ]
    output = {}
    for key in keys:
        left = _number(run.get(key))
        right = _number(baseline.get(key))
        output["%s_%s_delta" % (prefix, key)] = None if left is None or right is None else left - right
    baseline_frag = _number(baseline.get("person_fragmentation_approx"))
    run_frag = _number(run.get("person_fragmentation_approx"))
    output["%s_person_fragmentation_reduction" % prefix] = None
    if baseline_frag is not None and baseline_frag != 0 and run_frag is not None:
        output["%s_person_fragmentation_reduction" % prefix] = (baseline_frag - run_frag) / baseline_frag
    return output


def write_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Write metrics JSON."""
    write_json(metrics, path)


def _load_generic_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in generic_csv_files(root):
        subset = path.parent.name
        file_rows, _fields = read_csv_rows(path)
        for row in file_rows:
            copied = dict(row)
            copied["subset"] = subset
            rows.append(copied)
    return rows


def _load_frame_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in frame_record_csv_files(root):
        file_rows, _fields = read_csv_rows(path)
        rows.extend(file_rows)
    return rows


def _load_frame_records(root: Path) -> List[Any]:
    records = []
    for path in frame_record_csv_files(root):
        try:
            records.extend(read_global_frame_records_file(path))
        except (OSError, ValueError, KeyError):
            continue
    return records


def _rows_per_track(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], int]:
    counts: Dict[Tuple[str, str, str, str], int] = {}
    for row in rows:
        key = row_track_key(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _multi_camera_tracks(rows: List[Dict[str, Any]]) -> int:
    cameras: Dict[Tuple[str, str, str, str], set] = {}
    for row in rows:
        key = row_track_key(row)
        cameras.setdefault(key, set()).add(str(row.get("camera_id", "")))
    return len([key for key, value in cameras.items() if len(value) > 1])


def _track1_validation(root: Path) -> Dict[str, Any]:
    for path in [
        root / "track1_validation_report.json",
        root / "validation_report.json",
        root / "validation" / "track1_validation_report.json",
    ]:
        data = read_json(path)
        if data is not None:
            return data
    return {}


def _count_text_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _false_merge_rate(global_eval: Dict[str, Any]) -> Optional[float]:
    count = _number(global_eval.get("false_merge_count"))
    unique = _number(global_eval.get("unique_global_tracks"))
    if count is None or unique is None or unique <= 0:
        return None
    return float(count) / float(unique)


def _number(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

