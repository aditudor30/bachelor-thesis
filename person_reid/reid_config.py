"""Configuration helpers for Step 16A Person ReID."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.person_reid.reid_utils import load_yaml


def load_person_reid_config(path: Path) -> Dict[str, Any]:
    """Load and normalize a Person ReID config."""
    data = load_yaml(path)
    section = data.get("reid_person", {})
    config = {
        "reid_person": section if isinstance(section, dict) else {},
        "paths": data.get("paths", {}),
        "backend": data.get("backend", {}),
        "crops": data.get("crops", {}),
        "embeddings": data.get("embeddings", {}),
        "diagnostics": data.get("diagnostics", {}),
    }
    return config


def output_root(config: Dict[str, Any]) -> Path:
    """Return configured output root."""
    return Path(str(config.get("reid_person", {}).get("output_root", "output/reid_person/baseline_v2_pseudo3d_fullcam")))

