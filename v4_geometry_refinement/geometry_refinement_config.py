"""Configuration helpers for Step 22C."""

from pathlib import Path
from typing import Any, Dict, List

import yaml


VARIANT_NAMES = [
    "v4_smooth_only",
    "v4_outlier_repair",
    "v4_dimension_consistency",
    "v4_yaw_refinement",
    "v4_geometry_refined_balanced",
]


def load_geometry_refinement_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the Step 22C YAML config."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Step 22C config must be a mapping: %s" % path)
    required = ["v4_geometry_refinement", "paths", "official_track1", "smoothing", "outlier_repair", "dimension_consistency", "yaw_refinement", "selection"]
    missing = [key for key in required if not isinstance(config.get(key), dict)]
    if missing:
        raise ValueError("Step 22C config missing sections: %s" % ", ".join(missing))
    config["_config_path"] = str(path)
    return config


def output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated V4 output root."""
    section = config.get("v4_geometry_refinement", {})
    return Path(str(section.get("output_root", "output/v4_geometry_refinement_official_023_027")))


def variant_root(config: Dict[str, Any], variant: str) -> Path:
    """Return one V4 variant directory."""
    if variant not in VARIANT_NAMES:
        raise ValueError("Unknown V4 variant: %s" % variant)
    return output_root(config) / "variants" / variant


def v31_track1_path(config: Dict[str, Any]) -> Path:
    """Return the frozen V3.1 input path."""
    return Path(str(config.get("paths", {}).get("v31_track1", "")))


def expected_scene_ids(config: Dict[str, Any]) -> List[int]:
    """Return expected official scene identifiers."""
    return [int(value) for value in config.get("official_track1", {}).get("valid_scene_ids", [23, 24, 25, 26, 27])]


def progress_default(config: Dict[str, Any]) -> bool:
    """Return configured progress behavior."""
    return bool(config.get("v4_geometry_refinement", {}).get("progress", True))

