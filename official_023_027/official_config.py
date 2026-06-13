"""Configuration helpers for Step 22A."""

from pathlib import Path
from typing import Any, Dict, List

import yaml


VARIANT_KEYS = ["v2_current", "v3_gap_aware_soft"]


def load_official_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the Step 22A configuration."""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Step 22A config must be a mapping: %s" % path)
    required = ["official_023_027", "paths", "test_scenes", "official_track1", "class_mapping"]
    missing = [key for key in required if not isinstance(value.get(key), dict)]
    if missing:
        raise ValueError("Step 22A config missing sections: %s" % ", ".join(missing))
    value["_config_path"] = str(path)
    return value


def output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated processing output root."""
    return Path(str(config.get("paths", {}).get("output_root", "output/official_023_027")))


def frozen_output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated official frozen-candidate root."""
    return Path(
        str(
            config.get("paths", {}).get(
                "frozen_output_root",
                "output/frozen_upload_candidates_official_023_027",
            )
        )
    )


def dataset_root(config: Dict[str, Any]) -> Path:
    """Return the configured dataset root."""
    return Path(str(config.get("paths", {}).get("dataset_root", "dataset/MTMC_Tracking_2026")))


def scene_names(config: Dict[str, Any], group: str = "all") -> List[str]:
    """Return configured test scene names."""
    return [str(value) for value in config.get("test_scenes", {}).get(group, [])]


def scene_ids(config: Dict[str, Any]) -> List[int]:
    """Return expected official scene identifiers."""
    return [int(value) for value in config.get("official_track1", {}).get("valid_scene_ids", [23, 24, 25, 26, 27])]


def class_remap(config: Dict[str, Any]) -> Dict[int, int]:
    """Return internal-to-official class mapping."""
    values = config.get("class_mapping", {}).get("internal_to_official", {})
    return {int(key): int(value) for key, value in values.items()}


def variant_extension_root(config: Dict[str, Any], variant: str) -> Path:
    """Return one extension root."""
    return output_root(config) / variant / "extension_026_027"


def variant_official_root(config: Dict[str, Any], variant: str) -> Path:
    """Return one processing-side official Track1 root."""
    return output_root(config) / variant / "official_track1"


def frozen_variant_name(variant: str) -> str:
    """Return frozen output directory name for a pipeline variant."""
    if variant == "v2_current":
        return "v2_current_official"
    return "v3_gap_aware_soft_official"


def frozen_variant_root(config: Dict[str, Any], variant: str) -> Path:
    """Return one frozen official candidate directory."""
    return frozen_output_root(config) / frozen_variant_name(variant)


def old_track1_path(config: Dict[str, Any], variant: str) -> Path:
    """Return the old frozen 023-025 Track1 path."""
    key = "v2_old_track1_023_025" if variant == "v2_current" else "v3_old_track1_023_025"
    return Path(str(config.get("paths", {}).get(key, "")))


def source_track1_path(config: Dict[str, Any], variant: str) -> Path:
    """Return the original 023-025 Track1 source used as a read-only fallback."""
    key = "v2_source_track1_023_025" if variant == "v2_current" else "v3_source_track1_023_025"
    return Path(str(config.get("paths", {}).get(key, "")))


def extension_track1_path(config: Dict[str, Any], variant: str) -> Path:
    """Return internal Track1 path produced for 026-027."""
    return variant_extension_root(config, variant) / "track1_internal_026_027.txt"


def progress_enabled(config: Dict[str, Any]) -> bool:
    """Return configured progress default."""
    return bool(config.get("official_023_027", {}).get("progress", True))
