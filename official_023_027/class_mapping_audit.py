"""Audit internal and official class mappings before final remapping."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.official_023_027.official_config import class_remap, output_root
from deep_oc_sort_3d.official_023_027.official_track1_io import write_json


def audit_class_mapping(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compare configured mappings with known internal detector configuration."""
    mapping = config.get("class_mapping", {})
    internal = _normalize_names(mapping.get("internal", {}))
    official = _normalize_names(mapping.get("official", {}))
    remap = class_remap(config)
    discovered = []
    detector_config = Path(
        str(
            config.get("paths", {}).get(
                "detector_pipeline_config",
                "deep_oc_sort_3d/configs/pipeline_yolo11m_medium_conf001.yaml",
            )
        )
    )
    candidate_paths = [
        detector_config,
        Path("deep_oc_sort_3d/configs/track1_schema_confirmed.yaml"),
    ]
    mismatches = []
    for path in candidate_paths:
        found = _discover_mapping(path)
        if found:
            discovered.append({"path": str(path), "mapping": found})
            if path == detector_config and found != internal:
                mismatches.append({"path": str(path), "expected_internal": internal, "found": found})
    mapping_complete = set(remap.keys()) == set(range(7)) and set(remap.values()) == set(range(7))
    name_consistent = all(internal.get(source) == official.get(target) for source, target in remap.items())
    status = "ok" if mapping_complete and name_consistent and not mismatches else "error"
    summary = {
        "status": status,
        "internal_mapping": internal,
        "official_mapping": official,
        "internal_to_official": remap,
        "mapping_complete_bijection": mapping_complete,
        "class_names_consistent_after_remap": name_consistent,
        "discovered_mappings": discovered,
        "mismatches": mismatches,
        "application_scope": "official Track1 final/freeze only",
        "pipeline_internal_mapping_unchanged": True,
    }
    write_json(output_root(config) / "audit" / "class_mapping_audit.json", summary)
    return summary


def _discover_mapping(path: Path) -> Dict[int, str]:
    if not path.exists() or path.suffix.lower() not in (".yaml", ".yml"):
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        return {}
    classes = value.get("classes", {})
    return _normalize_names(classes) if isinstance(classes, dict) else {}


def _normalize_names(values: Any) -> Dict[int, str]:
    if not isinstance(values, dict):
        return {}
    return {int(key): str(value) for key, value in values.items()}
