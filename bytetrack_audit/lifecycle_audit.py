"""Artifact-only and optional instrumented ByteTrack lifecycle audit."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.bytetrack_audit.audit_config import output_root, variant_paths, variant_tracker_settings
from deep_oc_sort_3d.bytetrack_audit.audit_io import iter_csv, iter_jsonl, progress_iter, safe_int, write_csv, write_json
from deep_oc_sort_3d.bytetrack_audit.instrumented_bytetrack_audit import run_instrumented_mini_rerun


def run_lifecycle_audit(
    config: Dict[str, Any],
    progress: bool = True,
    artifact_only: bool = True,
    instrumented_mini_rerun: bool = False,
) -> Dict[str, Any]:
    """Audit exported records and optionally collect actual lifecycle events."""
    rows = []
    warnings = []
    for variant_name in ["bytetrack_21b", "bytetrack_21c_best"]:
        variant_rows, variant_warnings = _artifact_variant_rows(config, variant_name, progress)
        rows.extend(variant_rows)
        warnings.extend(variant_warnings)
    instrumented = {}
    if instrumented_mini_rerun and not artifact_only:
        instrumented = run_instrumented_mini_rerun(config, progress=progress)
    combined = _aggregate(rows, ["variant_name"])
    per_scene = _aggregate(rows, ["variant_name", "scene_name"])
    per_camera = _aggregate(rows, ["variant_name", "scene_name", "camera_id"])
    per_class = _aggregate(rows, ["variant_name", "class_name"])
    root = output_root(config) / "lifecycle_audit"
    write_csv(root / "bytetrack_lifecycle_summary.csv", combined)
    write_json(
        root / "bytetrack_lifecycle_summary.json",
        {"artifact_only": True, "rows": combined, "instrumented": instrumented},
    )
    write_csv(root / "per_scene_lifecycle.csv", per_scene)
    write_csv(root / "per_camera_lifecycle.csv", per_camera)
    write_csv(root / "per_class_lifecycle.csv", per_class)
    write_csv(root / "matched_high_low_summary.csv", _select_columns(combined, [
        "variant_name", "high_conf_detections", "low_conf_detections",
        "matched_high_detections", "matched_low_detections",
        "unmatched_high_detections", "unmatched_low_detections",
    ]))
    write_csv(root / "tentative_confirmed_lost_removed_summary.csv", _select_columns(combined, [
        "variant_name", "tentative_tracks", "confirmed_tracks", "lost_tracks", "removed_tracks",
    ]))
    write_csv(root / "exported_vs_not_exported_summary.csv", _select_columns(combined, [
        "variant_name", "input_observations", "exported_local_records",
        "associated_but_not_exported_records", "export_retention",
    ]))
    write_json(root / "lifecycle_warnings.json", {"warnings": sorted(set(warnings))})
    return {"rows": combined, "camera_rows": rows, "warnings": warnings, "instrumented": instrumented}


def _artifact_variant_rows(
    config: Dict[str, Any],
    variant_name: str,
    progress: bool,
) -> Any:
    observations_root = Path(str(config.get("paths", {}).get("v2_observations_root", "")))
    local_root = variant_paths(config, variant_name).get("local_tracks_root", Path(""))
    settings = variant_tracker_settings(config, variant_name)
    high_thresh = float(settings.get("track_high_thresh", 0.3))
    low_thresh = float(settings.get("track_low_thresh", 0.05))
    rows = []
    warnings = []
    files = [path for path in sorted(local_root.rglob("*.csv")) if "summaries" not in set(path.parts)] if local_root.exists() else []
    for local_path in progress_iter(files, progress, "%s lifecycle artifacts" % variant_name):
        relative = local_path.relative_to(local_root)
        if len(relative.parts) < 3:
            continue
        subset, scene_name = relative.parts[0], relative.parts[1]
        camera_id = local_path.stem
        observation_path = observations_root / subset / scene_name / (camera_id + ".jsonl")
        observations = list(iter_jsonl(observation_path))
        local_rows = list(iter_csv(local_path))
        exported_keys = set((safe_int(row.get("frame_id"), -1), safe_int(row.get("detection_id"), -1)) for row in local_rows)
        class_names = sorted(set(
            [str(row.get("class_name", row.get("class_id", "unknown"))) for row in observations]
            + [str(row.get("class_name", row.get("class_id", "unknown"))) for row in local_rows]
        ))
        for class_name in class_names:
            input_rows = [row for row in observations if str(row.get("class_name", row.get("class_id", "unknown"))) == class_name]
            exported_rows = [row for row in local_rows if str(row.get("class_name", row.get("class_id", "unknown"))) == class_name]
            high = [row for row in input_rows if float(row.get("confidence", 0.0)) >= high_thresh]
            low = [row for row in input_rows if low_thresh <= float(row.get("confidence", 0.0)) < high_thresh]
            matched_high = [row for row in high if (safe_int(row.get("frame_id"), -1), safe_int(row.get("detection_id"), -1)) in exported_keys]
            matched_low = [row for row in low if (safe_int(row.get("frame_id"), -1), safe_int(row.get("detection_id"), -1)) in exported_keys]
            tentative = sum(1 for row in exported_rows if str(row.get("track_state", "")) == "tentative")
            confirmed = sum(1 for row in exported_rows if str(row.get("track_state", "")) == "confirmed")
            rows.append(
                {
                    "variant_name": variant_name,
                    "subset": subset,
                    "scene_name": scene_name,
                    "camera_id": camera_id,
                    "class_name": class_name,
                    "input_observations": len(input_rows),
                    "input_detections": len(input_rows),
                    "high_conf_detections": len(high),
                    "low_conf_detections": len(low),
                    "matched_high_detections": len(matched_high),
                    "matched_low_detections": len(matched_low),
                    "unmatched_high_detections": len(high) - len(matched_high),
                    "unmatched_low_detections": len(low) - len(matched_low),
                    "tentative_tracks": tentative,
                    "confirmed_tracks": confirmed,
                    "lost_tracks": "",
                    "removed_tracks": "",
                    "detections_associated_to_any_track": len(exported_rows),
                    "detections_associated_to_confirmed_track": confirmed,
                    "detections_associated_to_tentative_track": tentative,
                    "detections_associated_to_lost_recent_track": "",
                    "exported_local_records": len(exported_rows),
                    "associated_but_not_exported_records": "",
                    "export_retention": None if not input_rows else float(len(exported_rows)) / float(len(input_rows)),
                    "audit_mode": "artifact_only",
                }
            )
        if not observation_path.exists():
            warnings.append("missing observations: %s" % observation_path)
    warnings.append(
        "%s artifact-only mode cannot prove lost/removed or associated-but-not-exported counts" % variant_name
    )
    return rows, warnings


def _aggregate(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    groups = {}
    numeric = [
        "input_observations", "input_detections", "high_conf_detections", "low_conf_detections",
        "matched_high_detections", "matched_low_detections", "unmatched_high_detections",
        "unmatched_low_detections", "tentative_tracks", "confirmed_tracks", "lost_tracks",
        "removed_tracks", "detections_associated_to_any_track",
        "detections_associated_to_confirmed_track", "detections_associated_to_tentative_track",
        "detections_associated_to_lost_recent_track", "exported_local_records",
        "associated_but_not_exported_records",
    ]
    for row in rows:
        group_key = tuple(str(row.get(key, "")) for key in keys)
        target = groups.setdefault(group_key, {key: row.get(key, "") for key in keys})
        for field in numeric:
            value = row.get(field)
            if value in (None, ""):
                continue
            target[field] = int(target.get(field, 0) or 0) + int(value)
    output = []
    for target in groups.values():
        inputs = int(target.get("input_observations", 0) or 0)
        exported = int(target.get("exported_local_records", 0) or 0)
        target["export_retention"] = None if inputs <= 0 else float(exported) / float(inputs)
        output.append(target)
    return sorted(output, key=lambda row: tuple(str(row.get(key, "")) for key in keys))


def _select_columns(rows: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    return [{field: row.get(field, "") for field in fields} for row in rows]

