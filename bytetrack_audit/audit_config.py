"""Configuration helpers for the Step 21D ByteTrack audit."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from deep_oc_sort_3d.bytetrack_audit.audit_io import write_yaml


VARIANT_NAMES = ["v2_current", "bytetrack_21b", "bytetrack_21c_best"]


def load_audit_config(path: Path) -> Dict[str, Any]:
    """Load and validate the audit configuration."""
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("ByteTrack audit config must be a mapping")
    if not isinstance(value.get("paths"), dict):
        raise ValueError("ByteTrack audit config requires paths")
    value["_config_path"] = str(path)
    return value


def output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated audit output root."""
    section = config.get("bytetrack_lifecycle_audit", {})
    return Path(str(section.get("output_root", "output/bytetrack_audit/baseline_v2_pseudo3d_fullcam")))


def audit_scenes(config: Dict[str, Any], include_test: bool = True) -> List[Tuple[str, str, str]]:
    """Return pipeline subset, dataset split and scene tuples."""
    output = []
    subsets = config.get("subsets", {})
    keys = ["audit_val", "audit_internal_holdout"]
    if include_test:
        keys.append("audit_test_probe_no_gt")
    for key in keys:
        groups = subsets.get(key, {})
        if not isinstance(groups, dict):
            continue
        for split, scenes in groups.items():
            subset = {"train": "internal_holdout", "val": "official_val", "test": "test"}.get(
                str(split), str(split)
            )
            for scene_name in scenes or []:
                output.append((subset, str(split), str(scene_name)))
    return sorted(set(output))


def instrumented_scenes(config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Return the optional mini-rerun scene selection."""
    output = []
    groups = config.get("lifecycle_audit", {}).get("instrumented_subset", {})
    for split, scenes in (groups.items() if isinstance(groups, dict) else []):
        subset = {"train": "internal_holdout", "val": "official_val", "test": "test"}.get(
            str(split), str(split)
        )
        for scene_name in scenes or []:
            output.append((subset, str(split), str(scene_name)))
    return sorted(set(output))


def variant_paths(config: Dict[str, Any], variant_name: str) -> Dict[str, Path]:
    """Resolve normalized stage roots for one compared variant."""
    paths = config.get("paths", {})
    if variant_name == "bytetrack_21c_best":
        root = Path(str(paths.get("bytetrack_21c_best", {}).get("variant_root", "")))
        return {
            "local_tracks_root": root / "local_tracks",
            "tracklets_root": root / "tracklets",
            "candidates_root": root / "candidates",
            "motion_clean_root": root / "motion_clean",
            "global_root": root / "global_mtmc",
            "final_export_root": root / "final_export",
            "track1_root": root / "track1_submission",
        }
    values = paths.get(variant_name, {})
    return {str(key): Path(str(value)) for key, value in values.items() if str(key).endswith("_root")}


def variant_tracker_settings(config: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    """Resolve ByteTrack thresholds from generated variant configs when available."""
    configured = config.get("lifecycle_audit", {}).get("tracker_settings_by_variant", {})
    if isinstance(configured, dict) and isinstance(configured.get(variant_name), dict):
        return configured.get(variant_name, {})
    if variant_name == "bytetrack_21c_best":
        root = Path(str(config.get("paths", {}).get("bytetrack_21c_best", {}).get("variant_root", "")))
        path = root / "variant_config.yaml"
    else:
        root = variant_paths(config, variant_name).get("local_tracks_root", Path(""))
        path = root / "variant_config.yaml"
    if path.exists():
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        settings = value.get("bytetrack_style", {}) if isinstance(value, dict) else {}
        if isinstance(settings, dict):
            return settings
    fallback = config.get("lifecycle_audit", {}).get("tracker_settings", {})
    return fallback if isinstance(fallback, dict) else {}


def write_resolved_config(config: Dict[str, Any]) -> Path:
    """Persist a copy of the effective audit configuration."""
    path = output_root(config) / "configs" / "resolved_config.yaml"
    write_yaml(path, {key: value for key, value in config.items() if not str(key).startswith("_")})
    return path
