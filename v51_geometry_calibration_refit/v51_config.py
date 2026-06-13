"""Configuration and immutable input selection for Step 22E."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1


VARIANT_NAMES = [
    "v51_dimension_scale_refit",
    "v51_center_bias_refit",
    "v51_depth_scale_refit",
    "v51_yaw_bias_refit",
    "v51_geometry_refit_balanced",
]

_VALIDATION_CACHE: Dict[Tuple[str, int], bool] = {}


def load_v51_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = ["v51_geometry_calibration_refit", "paths", "official_track1", "class_mapping", "calibration_splits", "selection"]
    if not isinstance(config, dict):
        raise ValueError("Step 22E config must be a mapping: %s" % path)
    missing = [key for key in required if not isinstance(config.get(key), dict)]
    if missing:
        raise ValueError("Step 22E config missing sections: %s" % ", ".join(missing))
    config["_config_path"] = str(path)
    return config


def output_root(config: Dict[str, Any]) -> Path:
    value = config.get("v51_geometry_calibration_refit", {}).get("output_root", "output/v51_geometry_calibration_refit_official_023_027")
    return Path(str(value))


def variant_root(config: Dict[str, Any], variant: str) -> Path:
    if variant not in VARIANT_NAMES:
        raise ValueError("Unknown V5.1 variant: %s" % variant)
    return output_root(config) / "variants" / variant


def input_track1_path(config: Dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    for key in ["v4_track1", "v31_track1"]:
        path = Path(str(paths.get(key, "")))
        if _valid_track1(path, config):
            return path
    raise FileNotFoundError("Neither the V4 nor V3.1 Track1 candidate is present and valid")


def input_variant_name(config: Dict[str, Any]) -> str:
    selected = input_track1_path(config)
    v4 = Path(str(config.get("paths", {}).get("v4_track1", "")))
    return "v4_geometry_refined_official" if selected == v4 else "v3_coverage_extended_official"


def progress_default(config: Dict[str, Any]) -> bool:
    return bool(config.get("v51_geometry_calibration_refit", {}).get("progress", True))


def calibration_scene_lookup(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for phase, section in config.get("calibration_splits", {}).items():
        for scene in section.get("scenes", []):
            result[str(scene)] = {"phase": str(phase), "split": str(section.get("split", ""))}
    return result


def source_roots(config: Dict[str, Any]) -> List[Path]:
    paths = config.get("paths", {})
    pipeline = Path(str(paths.get("pipeline_runs_root", "output/pipeline_runs")))
    pseudo = Path(str(paths.get("pseudo3d_fullcam_root", "output/pseudo3d/baseline_v2_pseudo3d_fullcam")))
    values = [
        str(output_root(config) / "generated_train_sources"),
        str(pseudo / "predictions_stabilized"),
        str(pseudo / "predictions_raw"),
    ]
    values.extend(list(paths.get("calibration_source_roots", [])))
    values.extend([
        str(pipeline / "yolo11m_medium_curriculum_conf001" / "observations3d"),
        str(pipeline / "baseline_v2_pseudo3d_fullcam" / "observations3d"),
    ])
    output: List[Path] = []
    seen = set()
    for value in values:
        path = Path(str(value))
        if str(path) not in seen:
            output.append(path)
            seen.add(str(path))
    return output


def official_to_internal(config: Dict[str, Any], class_id: int) -> Optional[int]:
    values = config.get("class_mapping", {}).get("official_to_internal", {})
    value = values.get(class_id, values.get(str(class_id)))
    return None if value is None else int(value)


def internal_to_official(config: Dict[str, Any], class_id: int) -> Optional[int]:
    values = config.get("class_mapping", {}).get("internal_to_official", {})
    value = values.get(class_id, values.get(str(class_id)))
    return None if value is None else int(value)


def _valid_track1(path: Path, config: Dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    key = (str(path.resolve()), int(path.stat().st_mtime_ns))
    if key not in _VALIDATION_CACHE:
        report = validate_official_track1(path, config, progress=False)
        _VALIDATION_CACHE[key] = report.get("status") == "ok" and int(report.get("num_errors", 1)) == 0
    return _VALIDATION_CACHE[key]
