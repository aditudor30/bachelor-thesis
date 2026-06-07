"""Config helpers for final freeze."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.final_freeze.freeze_io import load_yaml


def load_final_freeze_config(config_path: Path) -> Dict[str, Any]:
    """Load final freeze YAML config."""
    return load_yaml(config_path)


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return final freeze output root."""
    section = config.get("final_freeze", {})
    return Path(str(section.get("output_root", "output/final_freeze")))

