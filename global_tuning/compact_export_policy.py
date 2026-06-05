"""Experimental compact export policy for tuning runs."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.global_tuning.tuning_io import safe_float, write_json


def apply_compact_export_policy(
    generic_export_root: Path,
    output_root: Path,
    policy: Dict[str, Any],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Write a compact copy of generic exports and return a report.

    This is deliberately separate from the official Track 1 writer. It drops
    only rows from very short, low-confidence global tracks according to the
    run-specific policy.
    """
    files = sorted(generic_export_root.rglob("*.csv"))
    report_rows = []
    totals = {
        "files": len(files),
        "rows_before": 0,
        "rows_after": 0,
        "rows_dropped": 0,
        "tracks_dropped": 0,
        "per_class_dropped": {},
        "per_scene_dropped": {},
    }
    for path in files:
        relative = path.relative_to(generic_export_root)
        output_path = output_root / relative
        file_report = compact_generic_export_file(path, output_path, policy)
        report_rows.append(file_report)
        totals["rows_before"] += int(file_report.get("rows_before", 0))
        totals["rows_after"] += int(file_report.get("rows_after", 0))
        totals["rows_dropped"] += int(file_report.get("rows_dropped", 0))
        totals["tracks_dropped"] += int(file_report.get("tracks_dropped", 0))
        _merge_counts(totals["per_class_dropped"], file_report.get("per_class_dropped", {}))
        _merge_counts(totals["per_scene_dropped"], file_report.get("per_scene_dropped", {}))
    summary = dict(totals)
    summary["files_detail"] = report_rows
    write_json(summary, output_root.parent / "summaries" / "compact_export_summary.json")
    return summary


def compact_generic_export_file(input_path: Path, output_path: Path, policy: Dict[str, Any]) -> Dict[str, Any]:
    """Compact one generic export CSV."""
    rows, fieldnames = _read_rows_with_header(input_path)
    groups = _group_rows(rows)
    dropped_keys = set()
    per_class_dropped = {}
    per_scene_dropped = {}
    for key, group_rows in groups.items():
        if should_drop_track(group_rows, policy):
            dropped_keys.add(key)
            class_name = str(group_rows[0].get("class_name", "unknown")) if group_rows else "unknown"
            scene_name = str(group_rows[0].get("scene_name", "unknown")) if group_rows else "unknown"
            per_class_dropped[class_name] = per_class_dropped.get(class_name, 0) + len(group_rows)
            per_scene_dropped[scene_name] = per_scene_dropped.get(scene_name, 0) + len(group_rows)
    kept_rows = [row for row in rows if _group_key(row) not in dropped_keys]
    _write_rows(output_path, kept_rows, fieldnames)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before": len(rows),
        "rows_after": len(kept_rows),
        "rows_dropped": len(rows) - len(kept_rows),
        "tracks_before": len(groups),
        "tracks_dropped": len(dropped_keys),
        "per_class_dropped": per_class_dropped,
        "per_scene_dropped": per_scene_dropped,
    }


def should_drop_track(rows: List[Dict[str, Any]], policy: Dict[str, Any]) -> bool:
    """Return True when a track should be removed by the compact policy."""
    if not rows:
        return False
    class_id = str(rows[0].get("class_id", ""))
    protected = set([str(value) for value in policy.get("protected_class_ids", [])])
    min_rows = int(policy.get("min_rows_per_track", 3))
    min_conf = float(policy.get("min_mean_confidence", 0.20))
    single_frame_min_conf = float(policy.get("single_frame_min_confidence", 0.35))
    if class_id in protected:
        min_rows = int(policy.get("protected_min_rows_per_track", 1))
        min_conf = float(policy.get("protected_min_mean_confidence", 0.0))
        single_frame_min_conf = float(policy.get("protected_single_frame_min_confidence", 0.0))
    mean_conf = _mean_confidence(rows)
    unique_frames = set([str(row.get("frame_id", "")) for row in rows])
    if bool(policy.get("drop_single_frame_global_tracks", True)):
        if len(unique_frames) <= 1 and mean_conf < single_frame_min_conf:
            return True
    if bool(policy.get("drop_low_conf_short_tracks", True)):
        if len(rows) < min_rows and mean_conf < min_conf:
            return True
    return False


def _read_rows_with_header(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def _write_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _group_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    groups = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    return groups


def _group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("scene_name", "")),
        str(row.get("class_id", "")),
        str(row.get("global_track_id", "")),
    )


def _mean_confidence(rows: List[Dict[str, Any]]) -> float:
    values = [safe_float(row.get("confidence"), None) for row in rows]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return 0.0
    return float(sum(numeric)) / float(len(numeric))


def _merge_counts(left: Dict[str, int], right: Dict[str, Any]) -> None:
    for key, value in right.items():
        left[str(key)] = left.get(str(key), 0) + int(value)

