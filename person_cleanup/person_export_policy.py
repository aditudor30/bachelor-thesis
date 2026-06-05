"""Apply Person cleanup policies to final export artifacts."""

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import (
    frame_record_csv_files,
    generic_csv_files,
    progress_iter,
    read_csv_rows,
    track_key,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.person_cleanup.person_pruning import TrackKey, keys_to_drop_for_policy
from deep_oc_sort_3d.person_cleanup.person_selective_merge import (
    apply_person_merge_mapping,
    build_safe_person_merge_mapping,
)
from deep_oc_sort_3d.person_cleanup.person_track_classifier import build_track_stats_from_rows, classify_person_track


def apply_person_cleanup_export_policy(
    source_final_export_root: Path,
    output_final_export_root: Path,
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Create a run-specific final export with Person cleanup applied."""
    generic_source = source_final_export_root / "generic_tracking_export"
    frame_source = source_final_export_root / "frame_global_records"
    generic_output = output_final_export_root / "generic_tracking_export"
    frame_output = output_final_export_root / "frame_global_records"
    subsets = _optional_list(config.get("apply_to_subsets"))
    scenes = _optional_list(config.get("apply_to_scenes"))
    all_generic_rows = _load_all_generic_rows(generic_source, subsets, scenes)
    person_stats = [classify_person_track(item, config.get("classification", {})) for item in build_track_stats_from_rows(all_generic_rows)]
    drop_keys, drop_audit_rows = keys_to_drop_for_policy(person_stats, config.get("pruning", {}))
    merge_mapping, merge_rows = build_safe_person_merge_mapping(all_generic_rows, config.get("selective_merge_safe", {}))
    generic_report = _write_filtered_generic_files(
        generic_source,
        generic_output,
        drop_keys,
        merge_mapping,
        subsets,
        scenes,
        show_progress,
    )
    frame_report = _write_filtered_frame_record_files(
        frame_source,
        frame_output,
        drop_keys,
        merge_mapping,
        subsets,
        scenes,
        show_progress,
    )
    summary = {
        "source_final_export_root": str(source_final_export_root),
        "output_final_export_root": str(output_final_export_root),
        "person_tracks_analyzed": len(person_stats),
        "drop_keys": len(drop_keys),
        "merge_mapping_size": len(merge_mapping),
        "merge_candidates": len(merge_rows),
        "generic_report": generic_report,
        "frame_report": frame_report,
    }
    summaries_root = output_final_export_root.parent / "summaries"
    write_json(summary, summaries_root / "person_cleanup_export_summary.json")
    write_csv_rows(drop_audit_rows, summaries_root / "dropped_tracks_analysis.csv")
    write_csv_rows(merge_rows, summaries_root / "merged_tracks_analysis.csv")
    return summary


def _load_all_generic_rows(root: Path, subsets: Any, scenes: Any) -> List[Dict[str, Any]]:
    rows = []
    for path in generic_csv_files(root, subsets=subsets, scenes=scenes):
        subset = path.parent.name
        file_rows, _fieldnames = read_csv_rows(path)
        for row in file_rows:
            copied = dict(row)
            copied["subset"] = subset
            rows.append(copied)
    return rows


def _write_filtered_generic_files(
    source_root: Path,
    output_root: Path,
    drop_keys: Set[TrackKey],
    merge_mapping: Dict[TrackKey, str],
    subsets: Any,
    scenes: Any,
    show_progress: bool,
) -> Dict[str, Any]:
    rows_before = 0
    rows_after = 0
    non_person_dropped = 0
    per_class_dropped = {}
    per_scene_dropped = {}
    files = generic_csv_files(source_root, subsets=subsets, scenes=scenes)
    for path in progress_iter(files, show_progress, "person cleanup generic files", "file"):
        subset = path.parent.name
        rows, fieldnames = read_csv_rows(path)
        working = []
        for row in rows:
            copied = dict(row)
            copied["subset"] = subset
            working.append(copied)
        kept = []
        for row in working:
            key = track_key(row, subset=subset)
            rows_before += 1
            if key in drop_keys:
                _count_drop(row, per_class_dropped, per_scene_dropped)
                if str(row.get("class_id", "")) != "0":
                    non_person_dropped += 1
                continue
            kept.append(dict(row))
            rows_after += 1
        kept = apply_person_merge_mapping(kept, merge_mapping)
        kept = [{field: row.get(field, "") for field in fieldnames} for row in kept]
        output_path = output_root / subset / path.name
        write_csv_rows(kept, output_path, fieldnames)
    return {
        "files": len(files),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_before - rows_after,
        "non_person_rows_dropped": non_person_dropped,
        "per_class_dropped": per_class_dropped,
        "per_scene_dropped": per_scene_dropped,
    }


def _write_filtered_frame_record_files(
    source_root: Path,
    output_root: Path,
    drop_keys: Set[TrackKey],
    merge_mapping: Dict[TrackKey, str],
    subsets: Any,
    scenes: Any,
    show_progress: bool,
) -> Dict[str, Any]:
    rows_before = 0
    rows_after = 0
    files = frame_record_csv_files(source_root, subsets=subsets, scenes=scenes)
    for path in progress_iter(files, show_progress, "person cleanup frame files", "file"):
        rows, fieldnames = read_csv_rows(path)
        kept = []
        for row in rows:
            key = track_key(row)
            rows_before += 1
            if key in drop_keys:
                continue
            kept.append(row)
            rows_after += 1
        kept = apply_person_merge_mapping(kept, merge_mapping)
        relative = path.relative_to(source_root)
        write_csv_rows(kept, output_root / relative, fieldnames)
    return {"files": len(files), "rows_before": rows_before, "rows_after": rows_after, "rows_dropped": rows_before - rows_after}


def _count_drop(row: Dict[str, Any], per_class: Dict[str, int], per_scene: Dict[str, int]) -> None:
    class_name = str(row.get("class_name", row.get("class_id", "")))
    scene_name = str(row.get("scene_name", ""))
    per_class[class_name] = per_class.get(class_name, 0) + 1
    per_scene[scene_name] = per_scene.get(scene_name, 0) + 1


def _optional_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]
