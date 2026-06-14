"""Configuration helpers for Step 23A."""

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_failure_audit_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Step 23A config must be a YAML mapping: %s" % path)
    value["_config_path"] = str(path)
    return value


def output_root(config: Dict[str, Any]) -> Path:
    section = config.get("official_failure_audit_23a", {})
    return Path(str(section.get("output_root", "output/official_failure_audit_23a")))


def dataset_root(config: Dict[str, Any]) -> Path:
    return Path(str(config.get("paths", {}).get("dataset_root", "")))


def val_scenes(config: Dict[str, Any]) -> List[str]:
    return [str(value) for value in config.get("val_scenes", ["Warehouse_020", "Warehouse_021", "Warehouse_022"])]


def scene_id(scene_name: str) -> int:
    try:
        return int(str(scene_name).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return -1


def progress_default(config: Dict[str, Any]) -> bool:
    return bool(config.get("official_failure_audit_23a", {}).get("progress", True))


def resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def internal_to_official(config: Dict[str, Any]) -> Dict[int, int]:
    raw = config.get("class_mapping", {}).get("internal_to_official", {})
    return {int(key): int(value) for key, value in raw.items()}


def official_class_names(config: Dict[str, Any]) -> Dict[str, int]:
    values = config.get("class_mapping", {}).get("official", {})
    return {str(name): int(class_id) for class_id, name in values.items()}
