"""Metrics for Person cleanup runs."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import (
    count_by,
    generic_csv_files,
    mean,
    percentile,
    read_csv_rows,
    read_json,
    safe_int,
    track_key,
    write_json,
)
from deep_oc_sort_3d.person_cleanup.person_fragmentation_audit import person_gt_diagnostic


def collect_person_cleanup_metrics(
    run_name: str,
    final_export_root: Path,
    track1_root: Path,
    global_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect comparable metrics for one cleanup run."""
    rows = _load_generic_rows(final_export_root / "generic_tracking_export")
    person_rows = [row for row in rows if safe_int(row.get("class_id"), -1) == 0]
    non_person_rows = [row for row in rows if safe_int(row.get("class_id"), -1) != 0]
    person_lengths = _rows_per_track(person_rows)
    validation = _track1_validation(track1_root)
    summary = read_json(track1_root / "track1_export_summary.json") or {}
    track1_rows = summary.get("rows_written")
    if track1_rows is None:
        track1_rows = _count_text_rows(track1_root / "track1.txt")
    frame_rows = _load_frame_rows(final_export_root / "frame_global_records")
    gt_diag = person_gt_diagnostic([row for row in frame_rows if safe_int(row.get("class_id"), -1) == 0], [])
    global_metrics = collect_global_reference_metrics(global_root) if global_root is not None else {}
    metrics = {
        "run_name": run_name,
        "final_export_root": str(final_export_root),
        "track1_root": str(track1_root),
        "generic_rows": len(rows),
        "person_rows": len(person_rows),
        "non_person_rows": len(non_person_rows),
        "person_unique_tracks": len(person_lengths),
        "person_rows_per_track_mean": mean(list(person_lengths.values())),
        "person_rows_per_track_median": percentile(list(person_lengths.values()), 50),
        "person_rows_per_track_p95": percentile(list(person_lengths.values()), 95),
        "person_singleton_tracks": len([value for value in person_lengths.values() if value <= 1]),
        "person_short_tracks_lte_3": len([value for value in person_lengths.values() if value <= 3]),
        "per_class_rows": count_by(rows, "class_name"),
        "per_scene_rows": count_by(rows, "scene_name"),
        "track1_rows": track1_rows,
        "track1_validation_status": validation.get("status"),
        "track1_validation_errors": validation.get("num_errors"),
        "person_purity": gt_diag.get("person_purity"),
        "person_false_merge_rate": gt_diag.get("person_false_merge_rate"),
        "person_fragmentation_approx": gt_diag.get("person_fragmentation_approx"),
    }
    metrics.update(global_metrics)
    return metrics


def collect_global_reference_metrics(global_root: Path) -> Dict[str, Any]:
    """Collect coarse global metrics from global association summaries."""
    if global_root is None:
        return {}
    evals = [read_json(path) for path in sorted(global_root.rglob("eval.json"))]
    evals = [item for item in evals if item is not None]
    summaries = [read_json(path) for path in sorted(global_root.rglob("summary.json"))]
    summaries = [item for item in summaries if item is not None]
    return {
        "global_tracks": sum([int(item.get("global_tracks", 0)) for item in summaries]),
        "multi_camera_tracks": sum([int(item.get("multi_camera_tracks", 0)) for item in summaries]),
        "accepted_edges": sum([int(item.get("accepted_edges", 0)) for item in summaries]),
        "global_purity_mean": mean([item.get("global_purity_mean") for item in evals]),
        "false_merge_rate": mean([item.get("false_merge_rate") for item in evals]),
        "fragmentation_approx": sum([int(item.get("fragmentation_approx", 0)) for item in evals]),
    }


def compute_cleanup_deltas(run: Dict[str, Any], baseline: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Compute cleanup deltas against a baseline."""
    keys = [
        "generic_rows",
        "person_rows",
        "non_person_rows",
        "track1_rows",
        "person_unique_tracks",
        "person_singleton_tracks",
        "person_short_tracks_lte_3",
        "person_fragmentation_approx",
        "person_purity",
        "person_false_merge_rate",
        "global_purity_mean",
        "false_merge_rate",
        "fragmentation_approx",
    ]
    output = {}
    for key in keys:
        run_value = _number(run.get(key))
        base_value = _number(baseline.get(key))
        output["%s_%s_delta" % (prefix, key)] = None if run_value is None or base_value is None else run_value - base_value
    base_frag = _number(baseline.get("person_fragmentation_approx"))
    run_frag = _number(run.get("person_fragmentation_approx"))
    output["%s_person_fragmentation_reduction" % prefix] = None
    if base_frag is not None and base_frag != 0 and run_frag is not None:
        output["%s_person_fragmentation_reduction" % prefix] = (base_frag - run_frag) / base_frag
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
    for path in sorted(root.rglob("*_global_records.csv")) if root.exists() else []:
        file_rows, _fields = read_csv_rows(path)
        rows.extend(file_rows)
    return rows


def _rows_per_track(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], int]:
    counts = {}
    for row in rows:
        key = track_key(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _track1_validation(root: Path) -> Dict[str, Any]:
    for name in ["track1_validation_report.json", "validation_report.json"]:
        data = read_json(root / name)
        if data is not None:
            return data
    data = read_json(root / "validation" / "track1_validation_report.json")
    return data if data is not None else {}


def _count_text_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _number(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
