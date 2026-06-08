"""Config loading and override helpers for SmartSpaces Person ReID dataset."""

from pathlib import Path
from typing import Any, Dict, Optional

from deep_oc_sort_3d.reid_training.reid_dataset_io import load_yaml


def load_person_reid_dataset_config(config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML config and apply CLI overrides."""
    config = load_yaml(config_path)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return output root for the ReID dataset."""
    section = config.get("person_reid_dataset", {})
    return Path(str(section.get("output_root", "output/reid_training/person_smartspaces_v1")))


def dataset_root_from_config(config: Dict[str, Any]) -> Path:
    """Return dataset root."""
    return Path(str(config.get("paths", {}).get("dataset_root", "/path/to/MTMC_Tracking_2026")))


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply non-None CLI overrides."""
    output = dict(config)
    paths = dict(output.get("paths", {}))
    dataset = dict(output.get("person_reid_dataset", {}))
    crops = dict(output.get("crop_extraction", {}))
    if overrides.get("dataset_root") is not None:
        paths["dataset_root"] = str(overrides["dataset_root"])
    if overrides.get("output_root") is not None:
        dataset["output_root"] = str(overrides["output_root"])
    if overrides.get("frame_stride") is not None:
        crops["frame_stride"] = int(overrides["frame_stride"])
    if overrides.get("max_crops_per_identity") is not None:
        crops["max_crops_per_identity"] = int(overrides["max_crops_per_identity"])
    if overrides.get("overwrite") is not None:
        dataset["overwrite"] = bool(overrides["overwrite"])
    if overrides.get("skip_existing") is not None:
        dataset["skip_existing"] = bool(overrides["skip_existing"])
    output["paths"] = paths
    output["person_reid_dataset"] = dataset
    output["crop_extraction"] = crops
    return output

