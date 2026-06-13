"""Configuration helpers for the Step 22B coverage extension."""

from pathlib import Path
from typing import Any, Dict, List

import yaml


VARIANT_NAMES = [
    "v3_short_track_safe",
    "v3_associated_tentative_export",
    "v3_scene_class_targeted_recovery",
    "v3_single_camera_keep_clean",
    "v3_balanced_coverage_extension",
]


def load_coverage_extension_config(path: Path) -> Dict[str, Any]:
    """Load and validate a Step 22B YAML configuration."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Step 22B config must be a mapping: %s" % path)
    required = ["v3_coverage_extension", "paths", "official_track1", "class_mapping", "recovery_rules", "selection"]
    missing = [key for key in required if not isinstance(config.get(key), dict)]
    if missing:
        raise ValueError("Step 22B config missing sections: %s" % ", ".join(missing))
    config["_config_path"] = str(path)
    return config


def output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated Step 22B output root."""
    section = config.get("v3_coverage_extension", {})
    return Path(str(section.get("output_root", "output/v3_coverage_extension_official_023_027")))


def variant_root(config: Dict[str, Any], variant: str) -> Path:
    """Return one variant output directory."""
    if variant not in VARIANT_NAMES:
        raise ValueError("Unknown Step 22B variant: %s" % variant)
    return output_root(config) / "variants" / variant


def expected_scene_ids(config: Dict[str, Any]) -> List[int]:
    """Return official scene identifiers."""
    return [int(value) for value in config.get("official_track1", {}).get("valid_scene_ids", [23, 24, 25, 26, 27])]


def valid_class_ids(config: Dict[str, Any]) -> List[int]:
    """Return valid official class identifiers."""
    return [int(value) for value in config.get("official_track1", {}).get("valid_class_ids", range(7))]


def internal_to_official(config: Dict[str, Any]) -> Dict[int, int]:
    """Return the immutable internal-to-official class mapping."""
    values = config.get("class_mapping", {}).get("internal_to_official", {})
    return {int(key): int(value) for key, value in values.items()}


def official_to_internal(config: Dict[str, Any]) -> Dict[int, int]:
    """Return the inverse official class mapping."""
    return {value: key for key, value in internal_to_official(config).items()}


def id_offset(config: Dict[str, Any], variant: str) -> int:
    """Return the collision-resistant object-ID range for a variant."""
    common = config.get("recovery_rules", {}).get("common", {})
    offsets = common.get("id_offsets", {})
    return int(offsets.get(variant, 900000000))


def progress_default(config: Dict[str, Any]) -> bool:
    """Return the configured progress default."""
    return bool(config.get("v3_coverage_extension", {}).get("progress", True))

