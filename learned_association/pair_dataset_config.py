"""Configuration loading for the Person association pair dataset."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_pair_dataset_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate a YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = ("person_association_pair_dataset", "paths", "splits")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("Missing config sections: %s" % ", ".join(missing))
    config["_config_path"] = str(path)
    return config


def apply_cli_overrides(
    config: Dict[str, Any],
    dataset_root: Optional[str] = None,
    output_root: Optional[str] = None,
    progress: Optional[bool] = None,
    max_pairs_per_scene: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply supported CLI overrides in-place and return the config."""
    if dataset_root is not None:
        config.setdefault("paths", {})["dataset_root"] = dataset_root
    if output_root is not None:
        config.setdefault("person_association_pair_dataset", {})["output_root"] = output_root
    if progress is not None:
        config.setdefault("person_association_pair_dataset", {})["progress"] = progress
    if max_pairs_per_scene is not None:
        config.setdefault("pair_generation", {})["max_pairs_per_scene"] = max_pairs_per_scene
    return config


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return the configured output root."""
    value = config.get("person_association_pair_dataset", {}).get(
        "output_root", "output/learned_association/person_pairs_v1"
    )
    return Path(value)


def configured_scenes(
    config: Dict[str, Any], debug_limit_scenes: Optional[int] = None
) -> List[Dict[str, str]]:
    """Flatten train/val scene configuration into ordered records."""
    records = []  # type: List[Dict[str, str]]
    for split_key in ("train", "val"):
        split_config = config.get("splits", {}).get(split_key, {})
        split_name = str(split_config.get("split_name", split_key))
        scenes = list(split_config.get("scenes", []))
        if debug_limit_scenes is not None:
            scenes = scenes[: max(0, debug_limit_scenes)]
        for scene_name in scenes:
            records.append({"split": split_name, "scene_name": str(scene_name)})
    return records


def progress_enabled(config: Dict[str, Any]) -> bool:
    """Return whether progress reporting is enabled."""
    return bool(config.get("person_association_pair_dataset", {}).get("progress", True))
