"""Audit V5 inputs, train/val sources and test camera-mapping safety."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import (
    calibration_scene_lookup,
    input_track1_path,
    input_variant_name,
    observation_source_roots,
    output_root,
)
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import (
    read_geometry_rows,
    unique_track_count,
    write_json,
    write_yaml,
)


def audit_calibration_sources(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Write source, mapping and frozen-input audits without reading test GT/depth."""
    root = output_root(config)
    input_path = input_track1_path(config)
    rows = read_geometry_rows(input_path, progress=progress)
    validation = validate_official_track1(input_path, config, progress=False)
    source_rows: List[Dict[str, Any]] = []
    scene_lookup = calibration_scene_lookup(config)
    available_scenes = set()
    for priority, source_root in enumerate(observation_source_roots(config)):
        files = list(source_root.rglob("*.jsonl")) if source_root.is_dir() else []
        scenes = sorted(set(part for path in files for part in path.parts if part in scene_lookup))
        available_scenes.update(scenes)
        source_rows.append({
            "priority": priority, "path": str(source_root), "exists": source_root.is_dir(),
            "jsonl_files": len(files), "calibration_scenes": scenes,
        })
    missing_scenes = sorted(set(scene_lookup.keys()) - available_scenes)
    camera_mapping = {
        "test_track1_contains_camera_id": False,
        "reliable_test_row_camera_mapping_available": False,
        "status": "not_applied_due_to_missing_camera_mapping",
        "reason": "Official Track1 keys do not contain camera_id and no explicit one-to-one test row sidecar is configured.",
        "depth_scale_test_application_allowed": False,
    }
    class_audit = {
        "official_to_internal": config.get("class_mapping", {}).get("official_to_internal", {}),
        "internal_to_official": config.get("class_mapping", {}).get("internal_to_official", {}),
        "track1_class_ids_are_never_modified": True,
    }
    input_summary = {
        "input_variant": input_variant_name(config), "track1_path": str(input_path),
        "rows": len(rows), "unique_tracks": unique_track_count(rows),
        "validation_status": validation.get("status"), "validation_errors": validation.get("num_errors"),
        "scene_distribution": validation.get("per_scene_rows"), "class_distribution": validation.get("per_class_rows"),
    }
    availability = {
        "dataset_root": str(config.get("paths", {}).get("dataset_root", "")),
        "input_track1_exists": input_path.is_file(), "input_variant": input_variant_name(config),
        "observation_sources": source_rows, "configured_calibration_scenes": sorted(scene_lookup.keys()),
        "available_calibration_scenes": sorted(available_scenes), "missing_calibration_scenes": missing_scenes,
    }
    write_yaml(root / "configs" / "resolved_config.yaml", {key: value for key, value in config.items() if not str(key).startswith("_")})
    write_json(root / "audit" / "input_availability_audit.json", availability)
    write_json(root / "audit" / "calibration_source_audit.json", {"sources": source_rows, "missing_scenes": missing_scenes})
    write_json(root / "audit" / "camera_mapping_audit.json", camera_mapping)
    write_json(root / "audit" / "class_mapping_audit.json", class_audit)
    write_json(root / "audit" / "v4_input_summary.json", input_summary)
    return {"availability": availability, "camera_mapping": camera_mapping, "input_summary": input_summary}
