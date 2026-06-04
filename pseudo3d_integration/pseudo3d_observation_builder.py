"""Build baseline_v2 Observation3D JSONL files from baseline observations and pseudo-3D."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from deep_oc_sort_3d.audit3d.audit3d_io import iter_data_files, progress_iter, read_jsonl_dicts, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import load_pseudo3d_priors
from deep_oc_sort_3d.pseudo3d_integration.observation3d_pseudo3d_adapter import merge_observation_with_pseudo3d
from deep_oc_sort_3d.pseudo3d_integration.pseudo3d_prediction_lookup import Pseudo3DPredictionLookup
from deep_oc_sort_3d.pseudo3d_integration.pseudo3d_integration_summary import summarize_integrated_observations


def build_pseudo3d_observations_for_file(
    observations_path: Path,
    pseudo3d_predictions_path_or_lookup: Union[str, Path, Pseudo3DPredictionLookup],
    output_path: Path,
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Merge pseudo-3D predictions into one baseline Observation3D JSONL file."""
    lookup = _lookup(pseudo3d_predictions_path_or_lookup)
    records = read_jsonl_dicts(observations_path)
    merged = []
    rows_for_progress = progress_iter(records, show_progress, "build pseudo3D observations", "observation")
    for record in rows_for_progress:
        prediction = _match_prediction(record, lookup, config)
        merged.append(merge_observation_with_pseudo3d(record, prediction, config))
    _write_jsonl_dicts(merged, output_path)
    metadata_path = _metadata_output_path(config, output_path)
    if metadata_path is not None:
        _write_jsonl_dicts([_metadata_row(row) for row in merged], metadata_path)
    summary = summarize_integrated_observations(merged)
    summary.update(
        {
            "input_path": str(observations_path),
            "output_path": str(output_path),
            "input_observations": len(records),
            "output_observations": len(merged),
        }
    )
    return summary


def build_pseudo3d_observations_batch(
    input_observations_root: Path,
    pseudo3d_predictions_root: Path,
    output_observations_root: Path,
    config: Dict[str, Any],
    subsets: Optional[List[str]] = None,
    scenes: Optional[List[str]] = None,
    camera_ids: Optional[List[str]] = None,
    show_progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build pseudo-3D observations for all matching baseline files."""
    config = _with_class_priors(config)
    lookup = Pseudo3DPredictionLookup(pseudo3d_predictions_root)
    files = _observation_files(input_observations_root, subsets, scenes, camera_ids)
    rows = []
    for path in progress_iter(files, show_progress, "baseline_v2 observation files", "file"):
        output_path = output_observations_root / path.relative_to(input_observations_root)
        if output_path.exists() and not overwrite:
            rows.append({"input_path": str(path), "output_path": str(output_path), "status": "skipped_existing"})
            continue
        try:
            row = build_pseudo3d_observations_for_file(path, lookup, output_path, config, show_progress)
            row["status"] = "ok"
            rows.append(row)
        except Exception as exc:
            rows.append({"input_path": str(path), "output_path": str(output_path), "status": "error", "error_message": str(exc)})
    summary = _batch_summary(rows, lookup.summary())
    write_json(summary, output_observations_root.parent / "summaries" / "pseudo3d_observation_summary.json")
    write_csv(rows, output_observations_root.parent / "summaries" / "pseudo3d_observation_summary.csv")
    return summary


def _match_prediction(record: Dict[str, Any], lookup: Pseudo3DPredictionLookup, config: Dict[str, Any]) -> Any:
    subset = str(record.get("subset") or _subset_from_split(record.get("split")))
    scene_name = str(record.get("scene_name", ""))
    camera_id = str(record.get("camera_id", ""))
    frame_id = int(record.get("frame_id", 0))
    class_id = int(record.get("class_id", -1))
    local_track_id = _optional_int(record.get("local_track_id"))
    global_track_id = _optional_int(record.get("global_track_id"))
    prediction = lookup.get_by_exact_key(subset, scene_name, camera_id, frame_id, class_id, local_track_id, global_track_id)
    policy = config.get("pseudo3d_integration", config)
    if prediction is None and bool(policy.get("bbox_iou_fallback_enabled", True)):
        record_for_lookup = dict(record)
        record_for_lookup["subset"] = subset
        prediction = lookup.get_by_bbox_iou_fallback(record_for_lookup, float(policy.get("bbox_iou_fallback_threshold", 0.8)))
    return prediction


def _observation_files(root: Path, subsets: Optional[List[str]], scenes: Optional[List[str]], camera_ids: Optional[List[str]]) -> List[Path]:
    subset_filter = None if subsets is None else set(subsets)
    scene_filter = None if scenes is None else set(scenes)
    camera_filter = None if camera_ids is None else set(camera_ids)
    files = []
    for path in iter_data_files(root, [".jsonl"]):
        subset, scene_name, camera_id = _parse_observation_path(root, path)
        if subset_filter is not None and subset not in subset_filter:
            continue
        if scene_filter is not None and scene_name not in scene_filter:
            continue
        if camera_filter is not None and camera_id not in camera_filter:
            continue
        files.append(path)
    return files


def _parse_observation_path(root: Path, path: Path) -> Tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if len(parts) >= 3:
        return parts[0], parts[1], Path(parts[2]).stem
    return "unknown", "unknown", path.stem


def _with_class_priors(config: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(config)
    paths = output.get("paths", {})
    priors_path = paths.get("class_priors") if isinstance(paths, dict) else None
    if not priors_path:
        return output
    table = load_pseudo3d_priors(priors_path)
    priors = {}
    for class_id, prior in table.priors_by_class_id.items():
        priors[int(class_id)] = {"width": prior.width, "length": prior.length, "height": prior.height}
        priors[str(class_id)] = {"width": prior.width, "length": prior.length, "height": prior.height}
    output["class_priors_by_id"] = priors
    return output


def _metadata_output_path(config: Dict[str, Any], output_path: Path) -> Optional[Path]:
    policy = config.get("pseudo3d_integration", config)
    if not bool(policy.get("write_source_metadata", True)):
        return None
    paths = config.get("paths", {})
    root = paths.get("output_metadata_root") if isinstance(paths, dict) else None
    if root is None:
        root = str(output_path.parent.parent.parent / "observations3d_metadata")
    try:
        relative = output_path.relative_to(Path(paths.get("output_observations_root", output_path.parent.parent.parent)))
    except (ValueError, AttributeError):
        relative = Path(output_path.name)
    return Path(root) / relative


def _metadata_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "scene_id",
        "scene_name",
        "split",
        "camera_id",
        "frame_id",
        "detection_id",
        "class_id",
        "pseudo3d_matched",
        "pseudo3d_used",
        "fallback_original_used",
        "has_3d",
        "center_3d_source",
        "dimensions_3d_source",
        "yaw_source",
        "depth_source",
        "is_gt_derived",
        "is_estimated_for_test",
        "pseudo3d_method",
        "pseudo3d_version",
        "fallback_reason",
        "source_notes",
    ]
    return {key: row.get(key) for key in keys}


def _write_jsonl_dicts(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _lookup(value: Union[str, Path, Pseudo3DPredictionLookup]) -> Pseudo3DPredictionLookup:
    if isinstance(value, Pseudo3DPredictionLookup):
        return value
    return Pseudo3DPredictionLookup(value)


def _batch_summary(rows: List[Dict[str, Any]], lookup_summary: Dict[str, Any]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    total = sum(int(row.get("output_observations", 0) or 0) for row in ok_rows)
    used = sum(int(row.get("pseudo3d_used", 0) or 0) for row in ok_rows)
    return {
        "files": len(rows),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
        "skipped": sum(1 for row in rows if row.get("status") == "skipped_existing"),
        "input_observations": sum(int(row.get("input_observations", 0) or 0) for row in ok_rows),
        "output_observations": total,
        "pseudo3d_used": used,
        "pseudo3d_used_rate": float(used) / float(total) if total else None,
        "pseudo3d_missing": sum(int(row.get("pseudo3d_missing", 0) or 0) for row in ok_rows),
        "fallback_original_used": sum(int(row.get("fallback_original_used", 0) or 0) for row in ok_rows),
        "class_prior_dimensions_used": sum(int(row.get("class_prior_dimensions_used", 0) or 0) for row in ok_rows),
        "no_3d_records": sum(int(row.get("no_3d_records", 0) or 0) for row in ok_rows),
        "lookup": lookup_summary,
        "files_detail": rows,
    }


def _subset_from_split(split: Any) -> str:
    if split == "val":
        return "official_val"
    if split == "train":
        return "internal_holdout"
    return str(split or "")


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

