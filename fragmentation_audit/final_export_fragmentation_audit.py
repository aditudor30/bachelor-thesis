"""Final export and Track1 fragmentation audit."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.final_export.track1_final_checks import (
    compute_track1_distribution,
    read_track1_txt,
    validate_track1_rows,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    add_count,
    find_data_files,
    iter_table_rows_progress,
    iter_table_rows,
    progress_iter,
    rate,
    safe_int,
    write_csv,
    write_json,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import (
    add_scope_counts,
    length_distribution,
    update_track_accumulator,
)


def audit_final_export(
    final_export_root: Path,
    track1_root: Path,
    output_path: Path,
    diagnostics_root: Path,
    run_name: str,
    thresholds: FragmentationThresholds,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit frame-level export, generic export, and official Track1 text."""
    generic_root = final_export_root / "generic_tracking_export"
    frame_root = final_export_root / "frame_global_records"
    generic_files = find_data_files(generic_root, suffixes=[".csv"])
    frame_files = find_data_files(frame_root, suffixes=[".csv", ".jsonl"])
    tracks = {}
    generic_rows = 0
    frame_rows = 0
    output = {
        "run_name": run_name,
        "final_export_root": str(final_export_root),
        "track1_root": str(track1_root),
        "generic_files": len(generic_files),
        "frame_record_files": len(frame_files),
        "generic_rows": 0,
        "frame_records": 0,
        "unique_global_tracks": 0,
        "rows_per_track": {},
        "per_subset": {},
        "per_scene": {},
        "per_camera": {},
        "per_class": {},
        "track1": {},
    }
    for path in progress_iter(generic_files, show_progress, "%s generic export" % run_name):
        for row in iter_table_rows_progress(path, show_progress, "%s generic rows" % run_name):
            generic_rows += 1
            update_track_accumulator(tracks, row, "global_track_id")
            add_scope_counts(output, row)
    for path in progress_iter(frame_files, show_progress, "%s frame records" % run_name):
        for _row in iter_table_rows_progress(path, show_progress, "%s frame rows" % run_name):
            frame_rows += 1
    lengths = [safe_int(item.get("length")) for item in tracks.values()]
    output["generic_rows"] = generic_rows
    output["frame_records"] = frame_rows
    output["unique_global_tracks"] = len(tracks)
    output["rows_per_track"] = length_distribution(lengths, thresholds)
    output["rows_per_track_singleton_ratio"] = output["rows_per_track"].get("singleton_ratio")
    output["rows_per_track_short_ratio"] = output["rows_per_track"].get("short_ratio")
    output["track1"] = _track1_summary(track1_root / "track1.txt", show_progress)
    output["track1_rows"] = output["track1"].get("rows")
    output["track1_validation_errors"] = output["track1"].get("validation", {}).get("num_errors")
    write_json(output, output_path)
    write_csv(_rows_per_track_rows(tracks, thresholds), diagnostics_root / "%s_rows_per_track_distribution.csv" % run_name)
    write_csv(_global_id_fragmentation_rows(tracks, thresholds), diagnostics_root / "%s_global_id_fragmentation_analysis.csv" % run_name)
    return output


def _track1_summary(path: Path, show_progress: bool) -> Dict[str, Any]:
    if not path.exists():
        return {"track1_path": str(path), "missing": True}
    rows = read_track1_txt(path)
    validation = validate_track1_rows(rows, show_progress=show_progress)
    distribution = compute_track1_distribution(rows)
    return {
        "track1_path": str(path),
        "missing": False,
        "rows": len(rows),
        "validation": validation,
        "distribution": distribution,
    }


def _rows_per_track_rows(tracks: Dict[Tuple[str, str, str, str, int], Dict[str, Any]], thresholds: FragmentationThresholds) -> List[Dict[str, Any]]:
    rows = [_track_row(item, thresholds) for item in tracks.values()]
    return sorted(rows, key=lambda item: int(item.get("rows", 0)))


def _global_id_fragmentation_rows(tracks: Dict[Tuple[str, str, str, str, int], Dict[str, Any]], thresholds: FragmentationThresholds) -> List[Dict[str, Any]]:
    rows = []
    for item in tracks.values():
        if int(item.get("length", 0)) <= thresholds.very_short_track_length:
            rows.append(_track_row(item, thresholds))
    return sorted(rows, key=lambda item: int(item.get("rows", 0)))


def _track_row(item: Dict[str, Any], thresholds: FragmentationThresholds) -> Dict[str, Any]:
    length = safe_int(item.get("length"))
    return {
        "subset": item.get("subset"),
        "scene": item.get("scene"),
        "camera": item.get("camera"),
        "class": item.get("class"),
        "global_track_id": item.get("track_id"),
        "rows": length,
        "start_frame": item.get("start_frame"),
        "end_frame": item.get("end_frame"),
        "is_singleton": length <= thresholds.singleton_length,
        "is_short": length <= thresholds.short_track_length,
        "is_very_short": length <= thresholds.very_short_track_length,
    }
