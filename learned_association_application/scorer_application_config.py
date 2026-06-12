"""Configuration loading for learned association application."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.learned_association_application.scorer_association_io import write_yaml


def load_application_config(path: Path) -> Dict[str, Any]:
    """Load the Step 20C YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Step 20C config must contain a YAML mapping")
    return value


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return the isolated Step 20C output root."""
    section = config.get("person_scorer_association", {})
    return Path(str(section.get("output_root", "output/learned_association/person_scorer_association_v1")))


def save_resolved_config(config: Dict[str, Any], output_root: Path) -> None:
    """Write the effective configuration."""
    write_yaml(output_root / "configs" / "resolved_config.yaml", config)


def variant_names(config: Dict[str, Any]) -> List[str]:
    """Return configured sweep names as a plain list for Python 3.9 callers."""
    return [str(value) for value in config.get("sweep", {}).get("variants", [])]
