"""Configuration helpers for Step 22D."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1


VARIANT_NAMES = [
    "v5_dimension_scale_calibrated",
    "v5_center_bias_calibrated",
    "v5_depth_scale_calibrated",
    "v5_yaw_bias_calibrated",
    "v5_geometry_calibrated_balanced",
]


_TRACK1_VALIDATION_CACHE: Dict[Tuple[str, int], bool] = {}


def load_geometry_calibration_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the V5 YAML configuration."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Step 22D config must be a mapping: %s" % path)
    required = ["v5_geometry_calibration", "paths", "official_track1", "class_mapping", "calibration_splits", "selection"]
    missing = [key for key in required if not isinstance(config.get(key), dict)]
    if missing:
        raise ValueError("Step 22D config missing sections: %s" % ", ".join(missing))
    config["_config_path"] = str(path)
    return config


def output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated V5 output root."""
    return Path(str(config.get("v5_geometry_calibration", {}).get("output_root", "output/v5_geometry_calibration_official_023_027")))


def variant_root(config: Dict[str, Any], variant: str) -> Path:
    """Return one V5 variant root."""
    if variant not in VARIANT_NAMES:
        raise ValueError("Unknown V5 variant: %s" % variant)
    return output_root(config) / "variants" / variant


def input_track1_path(config: Dict[str, Any]) -> Path:
    """Select validated V4 when available, otherwise V3.1."""
    paths = config.get("paths", {})
    v4_path = Path(str(paths.get("v4_track1", "")))
    if _is_valid_track1(v4_path, config):
        return v4_path
    v31_path = Path(str(paths.get("v31_track1", "")))
    if _is_valid_track1(v31_path, config):
        return v31_path
    raise FileNotFoundError("Neither the V4 nor V3.1 Track1 candidate is present and valid")


def input_variant_name(config: Dict[str, Any]) -> str:
    """Return the chosen immutable input label."""
    configured_v4 = Path(str(config.get("paths", {}).get("v4_track1", "")))
    return "v4_geometry_refined_official" if input_track1_path(config) == configured_v4 else "v3_coverage_extended_official"


def progress_default(config: Dict[str, Any]) -> bool:
    """Return configured progress behavior."""
    return bool(config.get("v5_geometry_calibration", {}).get("progress", True))


def calibration_scene_lookup(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Map configured calibration scenes to phase and dataset split."""
    output: Dict[str, Dict[str, str]] = {}
    for phase, section in config.get("calibration_splits", {}).items():
        for scene in section.get("scenes", []):
            output[str(scene)] = {"phase": str(phase), "split": str(section.get("split", ""))}
    return output


def observation_source_roots(config: Dict[str, Any]) -> List[Path]:
    """Return configured and auto-derived observation roots in priority order."""
    paths = config.get("paths", {})
    values = list(paths.get("observation_source_roots", []))
    pipeline_root = Path(str(paths.get("pipeline_runs_root", "output/pipeline_runs")))
    values.extend([
        str(pipeline_root / "baseline_v2_pseudo3d_fullcam" / "observations3d"),
        str(pipeline_root / "yolo11m_medium_curriculum_conf001" / "observations3d"),
    ])
    output: List[Path] = []
    seen = set()
    for value in values:
        path = Path(str(value))
        key = str(path)
        if key not in seen:
            output.append(path)
            seen.add(key)
    return output


def official_to_internal(config: Dict[str, Any], class_id: int) -> Optional[int]:
    """Map one official class ID to its internal ID."""
    values = config.get("class_mapping", {}).get("official_to_internal", {})
    value = values.get(class_id, values.get(str(class_id)))
    return None if value is None else int(value)


def internal_to_official(config: Dict[str, Any], class_id: int) -> Optional[int]:
    """Map one internal class ID to its official ID."""
    values = config.get("class_mapping", {}).get("internal_to_official", {})
    value = values.get(class_id, values.get(str(class_id)))
    return None if value is None else int(value)


def _is_valid_track1(path: Path, config: Dict[str, Any]) -> bool:
    """Validate and cache one immutable Track1 candidate before selecting it."""
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    cache_key = (str(path.resolve()), int(path.stat().st_mtime_ns))
    if cache_key not in _TRACK1_VALIDATION_CACHE:
        report = validate_official_track1(path, config, progress=False)
        _TRACK1_VALIDATION_CACHE[cache_key] = report.get("status") == "ok" and int(report.get("num_errors", 1)) == 0
    return _TRACK1_VALIDATION_CACHE[cache_key]
