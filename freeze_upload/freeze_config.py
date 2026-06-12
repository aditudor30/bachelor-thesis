"""Configuration helpers for Step 21F."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.freeze_upload.freeze_io import load_yaml, write_yaml


CANDIDATE_ORDER = ["v2_current", "v3_gap_aware_soft"]


def load_freeze_config(path: Path) -> Dict[str, Any]:
    """Load and validate the freeze configuration."""
    value = load_yaml(path)
    paths = value.get("paths", {})
    missing = [name for name in CANDIDATE_ORDER if not isinstance(paths.get(name), dict)]
    if missing:
        raise ValueError("Missing freeze candidate configs: %s" % ", ".join(missing))
    value["_config_path"] = str(path)
    return value


def output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated Step 21F output root."""
    section = config.get("freeze_upload_candidates", {})
    return Path(str(section.get("output_root", "output/frozen_upload_candidates")))


def candidate_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return normalized candidate specifications in stable order."""
    paths = config.get("paths", {})
    output = []
    for name in CANDIDATE_ORDER:
        value = dict(paths.get(name, {}))
        value["candidate_name"] = str(value.get("candidate_name", name))
        value["config_key"] = name
        output.append(value)
    return output


def candidate_output_root(config: Dict[str, Any], candidate_name: str) -> Path:
    """Return one frozen candidate directory."""
    return output_root(config) / str(candidate_name)


def write_resolved_config(config: Dict[str, Any]) -> Path:
    """Write the resolved config into the isolated output root."""
    path = output_root(config) / "configs" / "resolved_config.yaml"
    write_yaml(path, {key: value for key, value in config.items() if not str(key).startswith("_")})
    return path

