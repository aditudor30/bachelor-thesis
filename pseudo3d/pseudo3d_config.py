"""Config loader for the future pseudo-3D estimator."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_pseudo3d_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load pseudo-3D estimator YAML config."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def validate_pseudo3d_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the minimal config contract used by Step 15B tests."""
    errors = []
    pseudo3d = config.get("pseudo3d", {})
    method = config.get("method", {})
    metadata = config.get("metadata", {})
    if not isinstance(pseudo3d.get("enabled"), bool):
        errors.append("pseudo3d.enabled_must_be_boolean")
    if not method.get("primary"):
        errors.append("method.primary_missing")
    if not method.get("fallback_order"):
        errors.append("method.fallback_order_missing")
    if metadata.get("write_source_fields") is not True:
        errors.append("metadata.write_source_fields_must_be_true")
    return {"status": "error" if errors else "ok", "errors": errors}


def pseudo3d_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact config summary."""
    return {
        "name": config.get("pseudo3d", {}).get("name"),
        "version": config.get("pseudo3d", {}).get("version"),
        "enabled": config.get("pseudo3d", {}).get("enabled"),
        "primary_method": config.get("method", {}).get("primary"),
        "fallback_order": config.get("method", {}).get("fallback_order", []),
    }
