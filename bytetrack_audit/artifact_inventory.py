"""Artifact inventory with explicit stage units and common dimensions."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.bytetrack_audit.audit_config import VARIANT_NAMES, audit_scenes, output_root, variant_paths
from deep_oc_sort_3d.bytetrack_audit.audit_io import (
    count_nonempty_lines,
    iter_csv,
    iter_jsonl,
    progress_iter,
    safe_bool,
    safe_int,
    write_csv,
    write_json,
)
from deep_oc_sort_3d.bytetrack_audit.unit_keys import STAGE_UNITS


STAGES = [
    "observations",
    "local_records",
    "tracklets",
    "candidates",
    "motion_clean_candidates",
    "global_tracks",
    "final_export_rows",
    "track1_rows",
]


def build_artifact_inventories(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Inventory all compared variants without loading whole datasets."""
    root = output_root(config)
    scene_selection = audit_scenes(config, include_test=True)
    common_observations = Path(str(config.get("paths", {}).get("v2_observations_root", "")))
    results = {}
    dimension_rows = []
    for variant_name in progress_iter(VARIANT_NAMES, progress, "artifact inventories"):
        result = inventory_variant(
            variant_name,
            common_observations,
            variant_paths(config, variant_name),
            scene_selection,
            progress,
        )
        results[variant_name] = result
        write_csv(root / "inventories" / _inventory_name(variant_name), result.get("stage_rows", []))
        dimension_rows.extend(result.get("dimension_rows", []))
    input_rows = results.get("v2_current", {}).get("observation_file_rows", [])
    write_csv(root / "inventories" / "input_observation_inventory.csv", input_rows)
    write_csv(root / "inventories" / "scene_camera_class_inventory.csv", dimension_rows)
    write_json(root / "inventories" / "inventory_summary.json", results)
    return results


def inventory_variant(
    variant_name: str,
    observations_root: Path,
    paths: Dict[str, Path],
    scene_selection: Sequence[Tuple[str, str, str]],
    progress: bool,
) -> Dict[str, Any]:
    """Inventory one complete run using one file at a time."""
    allowed = set((subset, scene) for subset, _split, scene in scene_selection)
    stage_rows = []
    dimension_rows = []
    observation_files = _selected_files(observations_root, "*.jsonl", allowed)
    observation_count, observation_dims, observation_file_rows = _count_record_files(
        observation_files, "observations", variant_name, observations_root, "jsonl", None, progress
    )
    _append_stage(stage_rows, variant_name, "observations", observation_count)
    dimension_rows.extend(_dimension_rows(variant_name, "observations", observation_dims))

    local_root = paths.get("local_tracks_root", Path(""))
    local_files = _selected_files(local_root, "*.csv", allowed, exclude_summaries=True)
    local_count, local_dims, _files = _count_record_files(
        local_files, "local_records", variant_name, local_root, "csv", None, progress
    )
    _append_stage(stage_rows, variant_name, "local_records", local_count)
    dimension_rows.extend(_dimension_rows(variant_name, "local_records", local_dims))

    tracklet_root = paths.get("tracklets_root", Path(""))
    tracklet_files = _prefer_jsonl(tracklet_root, "*_tracklets", allowed)
    tracklet_count, tracklet_dims, _files = _count_record_files(
        tracklet_files, "tracklets", variant_name, tracklet_root, _format(tracklet_files), None, progress
    )
    _append_stage(stage_rows, variant_name, "tracklets", tracklet_count)
    dimension_rows.extend(_dimension_rows(variant_name, "tracklets", tracklet_dims))

    candidate_root = paths.get("candidates_root", Path(""))
    candidate_files = _prefer_jsonl(candidate_root, "*_candidates", allowed)
    candidate_count, candidate_dims, _files = _count_record_files(
        candidate_files,
        "candidates",
        variant_name,
        candidate_root,
        _format(candidate_files),
        _candidate_kept,
        progress,
    )
    _append_stage(stage_rows, variant_name, "candidates", candidate_count)
    dimension_rows.extend(_dimension_rows(variant_name, "candidates", candidate_dims))

    motion_root = paths.get("motion_clean_root", Path(""))
    motion_files = _prefer_jsonl(motion_root, "*_clean_candidates", allowed)
    motion_count, motion_dims, _files = _count_record_files(
        motion_files,
        "motion_clean_candidates",
        variant_name,
        motion_root,
        _format(motion_files),
        None,
        progress,
    )
    _append_stage(stage_rows, variant_name, "motion_clean_candidates", motion_count)
    dimension_rows.extend(_dimension_rows(variant_name, "motion_clean_candidates", motion_dims))

    global_root = paths.get("global_root", Path(""))
    global_files = _selected_named_files(global_root, "global_tracks.jsonl", allowed)
    if not global_files:
        global_files = _selected_named_files(global_root, "global_tracks.csv", allowed)
    global_count, global_dims, _files = _count_record_files(
        global_files, "global_tracks", variant_name, global_root, _format(global_files), None, progress
    )
    _append_stage(stage_rows, variant_name, "global_tracks", global_count)
    dimension_rows.extend(_dimension_rows(variant_name, "global_tracks", global_dims))

    final_root = paths.get("final_export_root", Path("")) / "generic_tracking_export"
    final_files = _selected_files(final_root, "*.csv", allowed, exclude_summaries=True)
    final_count, final_dims, _files = _count_record_files(
        final_files, "final_export_rows", variant_name, final_root, "csv", None, progress
    )
    _append_stage(stage_rows, variant_name, "final_export_rows", final_count)
    dimension_rows.extend(_dimension_rows(variant_name, "final_export_rows", final_dims))

    track1_root = paths.get("track1_root", Path(""))
    track1_path = track1_root / "track1.txt"
    track1_count = count_nonempty_lines(track1_path)
    _append_stage(stage_rows, variant_name, "track1_rows", track1_count)
    return {
        "variant_name": variant_name,
        "stage_rows": stage_rows,
        "stage_counts": {row["stage_name"]: row["count"] for row in stage_rows},
        "dimension_rows": dimension_rows,
        "observation_file_rows": observation_file_rows,
        "warnings": _missing_warnings(paths),
    }


def _count_record_files(
    files: List[Path],
    stage: str,
    variant: str,
    root: Path,
    file_format: str,
    predicate: Any,
    progress: bool,
) -> Tuple[int, Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    total = 0
    dimensions = {"scene": {}, "camera": {}, "class": {}, "person_group": {}}
    file_rows = []
    for path in progress_iter(files, progress, "%s %s" % (variant, stage)):
        subset, scene_name, camera_id = _path_identity(root, path)
        count = 0
        iterator = iter_jsonl(path) if file_format == "jsonl" else iter_csv(path)
        for row in iterator:
            if predicate is not None and not predicate(row):
                continue
            count += 1
            class_name = str(row.get("class_name", row.get("class_id", "unknown")))
            _inc(dimensions["scene"], str(row.get("scene_name", scene_name)))
            _inc(dimensions["camera"], str(row.get("camera_id", camera_id)))
            _inc(dimensions["class"], class_name)
            group = "Person" if class_name == "Person" or safe_int(row.get("class_id"), -1) == 0 else "NonPerson"
            _inc(dimensions["person_group"], group)
        total += count
        file_rows.append(
            {
                "variant_name": variant,
                "stage_name": stage,
                "subset": subset,
                "scene_name": scene_name,
                "camera_id": camera_id,
                "file_path": str(path),
                "count": count,
            }
        )
    return total, dimensions, file_rows


def _selected_files(
    root: Path,
    pattern: str,
    allowed: set,
    exclude_summaries: bool = False,
) -> List[Path]:
    if not root.exists():
        return []
    output = []
    for path in sorted(root.rglob(pattern)):
        if exclude_summaries and "summaries" in set(path.parts):
            continue
        subset, scene, _camera = _path_identity(root, path)
        if (subset, scene) in allowed:
            output.append(path)
    return output


def _selected_named_files(root: Path, filename: str, allowed: set) -> List[Path]:
    if not root.exists():
        return []
    output = []
    for path in sorted(root.rglob(filename)):
        subset, scene, _camera = _path_identity(root, path)
        if (subset, scene) in allowed:
            output.append(path)
    return output


def _prefer_jsonl(root: Path, stem_pattern: str, allowed: set) -> List[Path]:
    jsonl = _selected_files(root, stem_pattern + ".jsonl", allowed, exclude_summaries=True)
    return jsonl if jsonl else _selected_files(root, stem_pattern + ".csv", allowed, exclude_summaries=True)


def _path_identity(root: Path, path: Path) -> Tuple[str, str, str]:
    try:
        relative = path.relative_to(root)
        parts = relative.parts
    except ValueError:
        parts = path.parts
    subset = parts[0] if len(parts) >= 2 else ""
    scene = parts[1] if len(parts) >= 3 else (path.stem if len(parts) == 2 else "")
    camera = path.stem.replace("_tracklets", "").replace("_candidates", "").replace("_clean", "")
    return str(subset), str(scene), str(camera)


def _candidate_kept(row: Dict[str, Any]) -> bool:
    value = row.get("is_candidate")
    return True if value in (None, "") else safe_bool(value)


def _format(files: List[Path]) -> str:
    return "jsonl" if files and files[0].suffix.lower() == ".jsonl" else "csv"


def _append_stage(rows: List[Dict[str, Any]], variant: str, stage: str, count: int) -> None:
    rows.append(
        {
            "variant_name": variant,
            "stage_name": stage,
            "unit_type": STAGE_UNITS.get(stage, stage),
            "count": int(count),
        }
    )


def _dimension_rows(variant: str, stage: str, values: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    output = []
    for dimension, counts in values.items():
        for key, count in counts.items():
            output.append(
                {
                    "variant_name": variant,
                    "stage_name": stage,
                    "unit_type": STAGE_UNITS.get(stage, stage),
                    "dimension": dimension,
                    "key": key,
                    "count": count,
                }
            )
    return output


def _inc(values: Dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1


def _missing_warnings(paths: Dict[str, Path]) -> List[str]:
    return ["missing path: %s" % path for path in paths.values() if not path.exists()]


def _inventory_name(variant_name: str) -> str:
    return {
        "v2_current": "v2_current_inventory.csv",
        "bytetrack_21b": "bytetrack_21b_inventory.csv",
        "bytetrack_21c_best": "bytetrack_21c_best_inventory.csv",
    }.get(variant_name, "%s_inventory.csv" % variant_name)
