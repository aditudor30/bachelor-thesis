"""Local tracking fragmentation audit."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.fragmentation_audit.fragmentation_io import (
    find_data_files,
    iter_table_rows,
    progress_iter,
    write_csv,
    write_json,
)
from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import (
    summarize_track_accumulator,
    update_track_accumulator,
)


def audit_local_tracking(
    local_tracks_root: Path,
    output_path: Path,
    diagnostics_root: Path,
    run_name: str,
    thresholds: FragmentationThresholds,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit frame-level local tracking records."""
    files = find_data_files(local_tracks_root)
    tracks = {}
    total_records = 0
    for path in progress_iter(files, show_progress, "%s local tracks" % run_name):
        for row in iter_table_rows(path):
            total_records += 1
            update_track_accumulator(tracks, row, "local_track_id")
    summary = summarize_track_accumulator(tracks, thresholds)
    summary.update(
        {
            "run_name": run_name,
            "root": str(local_tracks_root),
            "files": len(files),
            "missing_root": not local_tracks_root.exists(),
            "total_records": total_records,
            "active_tracks": summary.get("num_tracks"),
        }
    )
    write_json(summary, output_path)
    write_csv(_worst_tracks(tracks, thresholds), diagnostics_root / "%s_worst_fragmented_local_tracks.csv" % run_name)
    write_csv(_short_tracks(tracks, thresholds), diagnostics_root / "%s_short_local_tracks.csv" % run_name)
    return summary


def _worst_tracks(tracks: Dict[Any, Dict[str, Any]], thresholds: FragmentationThresholds) -> List[Dict[str, Any]]:
    rows = sorted(tracks.values(), key=lambda item: int(item.get("length", 0)))
    return [_track_row(item, thresholds) for item in rows[:200]]


def _short_tracks(tracks: Dict[Any, Dict[str, Any]], thresholds: FragmentationThresholds) -> List[Dict[str, Any]]:
    rows = [item for item in tracks.values() if int(item.get("length", 0)) <= thresholds.short_track_length]
    return [_track_row(item, thresholds) for item in rows[:5000]]


def _track_row(item: Dict[str, Any], thresholds: FragmentationThresholds) -> Dict[str, Any]:
    length = int(item.get("length", 0))
    return {
        "subset": item.get("subset"),
        "scene": item.get("scene"),
        "camera": item.get("camera"),
        "class": item.get("class"),
        "track_id": item.get("track_id"),
        "length": length,
        "start_frame": item.get("start_frame"),
        "end_frame": item.get("end_frame"),
        "is_singleton": length <= thresholds.singleton_length,
        "is_short": length <= thresholds.short_track_length,
        "num_gt_ids": len(item.get("gt_ids", {})),
    }

