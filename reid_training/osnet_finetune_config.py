"""Config helpers for OSNet SmartSpaces Person ReID fine-tuning."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from deep_oc_sort_3d.reid_training.reid_dataset_io import load_yaml, write_json


def load_osnet_finetune_config(config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load OSNet fine-tuning config and apply CLI overrides."""
    config = load_yaml(config_path)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return fine-tuning output root."""
    section = config.get("osnet_person_smartspaces_finetune", {})
    return Path(str(section.get("output_root", "output/reid_training/osnet_person_smartspaces_v1")))


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply non-None CLI overrides."""
    output = dict(config)
    section = dict(output.get("osnet_person_smartspaces_finetune", {}))
    training = dict(output.get("training", {}))
    data = dict(output.get("data", {}))
    paths = dict(output.get("paths", {}))
    if overrides.get("output_root") is not None:
        section["output_root"] = str(overrides["output_root"])
    if overrides.get("epochs") is not None:
        training["epochs"] = int(overrides["epochs"])
    if overrides.get("batch_size") is not None:
        data["batch_size"] = int(overrides["batch_size"])
    if overrides.get("device") is not None:
        training["device"] = str(overrides["device"])
    if overrides.get("weights") is not None:
        paths["pretrained_osnet_weights"] = str(overrides["weights"])
    output["osnet_person_smartspaces_finetune"] = section
    output["training"] = training
    output["data"] = data
    output["paths"] = paths
    return output


def prepare_output_dirs(config: Dict[str, Any], overwrite: bool = False) -> Path:
    """Create output directories and optionally clear non-dataset outputs."""
    root = output_root_from_config(config)
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    for name in ["configs", "checkpoints", "logs", "embeddings", "evaluation", "figures", "reports"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def save_resolved_config(config: Dict[str, Any], output_root: Path) -> Path:
    """Save resolved YAML-like config as JSON for reproducibility."""
    path = output_root / "configs" / "osnet_person_smartspaces_v1_resolved.yaml"
    try:
        import yaml

        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except ImportError:
        write_json(config, path.with_suffix(".json"))
    return path


def summarize_environment(config: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    """Write minimal environment summary."""
    summary: Dict[str, Any] = {"torch_available": False, "torchreid_available": False, "cuda_available": False}
    try:
        import torch

        summary["torch_available"] = True
        summary["torch_version"] = getattr(torch, "__version__", "")
        summary["cuda_available"] = bool(torch.cuda.is_available())
        summary["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except ImportError:
        pass
    try:
        import torchreid

        summary["torchreid_available"] = True
        summary["torchreid_module"] = str(torchreid)
    except ImportError:
        pass
    summary["pretrained_osnet_weights"] = str(config.get("paths", {}).get("pretrained_osnet_weights", ""))
    write_json(summary, output_root / "logs" / "environment_summary.json")
    return summary

