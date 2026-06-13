"""Audit immutable inputs and calibration-source availability for Step 22E."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import (
    calibration_scene_lookup,
    input_track1_path,
    input_variant_name,
    output_root,
    source_roots,
)
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import (
    iter_jsonl,
    read_geometry_rows,
    unique_track_count,
    write_json,
    write_yaml,
)


def discover_source_files(config: Dict[str, Any]) -> List[Tuple[str, str, str, str, Path]]:
    """Return one highest-priority JSONL source per scene/camera."""
    scene_lookup = calibration_scene_lookup(config)
    selected: Dict[Tuple[str, str], Tuple[str, str, str, str, Path]] = {}
    for root in source_roots(config):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            scene = next((part for part in path.parts if part in scene_lookup), None)
            if scene is None:
                continue
            camera_id = source_camera_id(path)
            key = (scene, camera_id)
            if key in selected:
                continue
            info = scene_lookup[scene]
            selected[key] = (info["phase"], info["split"], scene, camera_id, path)
    return sorted(selected.values(), key=lambda item: (item[0], item[2], item[3]))


def source_camera_id(path: Path) -> str:
    """Read camera_id from a source row, with filename fallback."""
    for row in iter_jsonl(path):
        value = row.get("camera_id")
        if value not in (None, ""):
            return str(value)
        break
    stem = path.stem
    for suffix in ["_pseudo3d_stabilized", "_pseudo3d_predictions", "_observations"]:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def audit_v51_sources(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Write the requested availability, mapping, input and baseline audits."""
    root = output_root(config)
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    lookup = calibration_scene_lookup(config)
    files = discover_source_files(config)
    source_scenes = set(item[2] for item in files)
    complete_source_scenes = set()
    scene_rows: List[Dict[str, Any]] = []
    for scene, info in sorted(lookup.items()):
        scene_root = dataset_root / info["split"] / scene
        matching = [item for item in files if item[2] == scene]
        expected_cameras = set(load_calibration_json(scene_root / "calibration.json").keys()) if (scene_root / "calibration.json").is_file() else set()
        available_cameras = set(item[3] for item in matching)
        camera_complete = bool(available_cameras) and (not expected_cameras or expected_cameras.issubset(available_cameras))
        if camera_complete:
            complete_source_scenes.add(scene)
        scene_rows.append({
            "phase": info["phase"], "split": info["split"], "scene": scene,
            "source_available": bool(matching), "source_files": len(matching),
            "expected_cameras": sorted(expected_cameras), "available_cameras": sorted(available_cameras),
            "camera_coverage_complete": camera_complete,
            "ground_truth_exists": (scene_root / "ground_truth.json").is_file(),
            "calibration_exists": (scene_root / "calibration.json").is_file(),
            "source_paths": [str(item[4]) for item in matching],
        })
    fit_scenes = set(config.get("calibration_splits", {}).get("fit_train", {}).get("scenes", []))
    missing_fit = sorted(fit_scenes - complete_source_scenes)
    input_path = input_track1_path(config)
    input_rows = read_geometry_rows(input_path, progress=progress)
    input_validation = validate_official_track1(input_path, config, progress=False)
    input_summary = {
        "input_variant": input_variant_name(config), "track1_path": str(input_path),
        "rows": len(input_rows), "unique_tracks": unique_track_count(input_rows),
        "validation_status": input_validation.get("status"),
        "validation_errors": input_validation.get("num_errors"),
        "scene_distribution": input_validation.get("per_scene_rows"),
        "class_distribution": input_validation.get("per_class_rows"),
    }
    availability = {
        "dataset_root": str(dataset_root), "input_track1_exists": input_path.is_file(),
        "input_variant": input_variant_name(config), "source_roots": [str(path) for path in source_roots(config)],
        "source_files": len(files), "available_calibration_scenes": sorted(source_scenes),
        "missing_calibration_scenes": sorted(set(lookup.keys()) - source_scenes),
        "fit_train_complete": not missing_fit, "missing_fit_train_scenes": missing_fit,
        "scenes": scene_rows,
    }
    camera = {
        "calibration_sources_have_camera_id": all(bool(item[3]) for item in files),
        "test_track1_contains_camera_id": False,
        "reliable_test_row_camera_mapping_available": False,
        "status": "camera_specific_calibration_not_applied",
        "depth_scale_test_application_allowed": False,
        "reason": "Official Track1 rows have no camera_id and no explicit row-to-camera sidecar is configured.",
    }
    mapping = {
        "official_to_internal": config.get("class_mapping", {}).get("official_to_internal", {}),
        "internal_to_official": config.get("class_mapping", {}).get("internal_to_official", {}),
        "track1_class_ids_are_never_modified": True,
    }
    write_yaml(root / "configs" / "resolved_config.yaml", {key: value for key, value in config.items() if not str(key).startswith("_")})
    write_json(root / "audit" / "input_availability_audit.json", availability)
    write_json(root / "audit" / "train_source_availability.json", {"fit_train_complete": not missing_fit, "missing_scenes": missing_fit, "scenes": scene_rows})
    write_json(root / "audit" / "camera_mapping_audit.json", camera)
    write_json(root / "audit" / "class_mapping_audit.json", mapping)
    write_json(root / "audit" / "v4_input_summary.json", input_summary)
    write_json(root / "audit" / "v5_baseline_summary.json", _reference_summary(config, "v5_track1"))
    return {"availability": availability, "camera_mapping": camera, "input_summary": input_summary}


def _reference_summary(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    path = Path(str(config.get("paths", {}).get(key, "")))
    if not path.is_file():
        return {"path": str(path), "exists": False, "status": "not_available"}
    rows = read_geometry_rows(path, progress=False)
    report = validate_official_track1(path, config, progress=False)
    return {
        "path": str(path), "exists": True, "rows": len(rows), "unique_tracks": unique_track_count(rows),
        "validation_status": report.get("status"), "validation_errors": report.get("num_errors"),
    }
