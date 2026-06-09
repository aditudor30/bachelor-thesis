"""Configuration helpers for final freeze v2."""

import shutil
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import load_yaml, write_yaml


DEFAULT_OUTPUT_ROOT = "output/final_freeze_v2"


def load_final_freeze_v2_config(config_path: Path) -> Dict[str, Any]:
    """Load final freeze v2 config."""
    return load_yaml(Path(config_path))


def final_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return top-level final freeze section."""
    section = config.get("final_freeze_v2", config)
    return section if isinstance(section, dict) else {}


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return final freeze v2 output root."""
    section = final_section(config)
    return Path(str(section.get("output_root", DEFAULT_OUTPUT_ROOT)))


def prepare_output_root(config: Dict[str, Any], overwrite: bool = False) -> Path:
    """Create output subdirectories."""
    root = output_root_from_config(config)
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    dirs = [
        "baselines/v1_geometry_only",
        "baselines/v2_pseudo3d_fullcam",
        "baselines/v2_export_compact",
        "baselines/reid_pretrained_diagnostic",
        "baselines/reid_finetuned_threshold_080",
        "baselines/reid_finetuned_combined_safe_080",
        "tables",
        "figures/qualitative_2d",
        "figures/qualitative_3d",
        "figures/bev",
        "figures/reid_panels",
        "figures/charts",
        "figures/captions",
        "reports",
        "packages/submission_safe_v1",
        "packages/mvp_3d_v2",
        "packages/compact_safe_v2",
        "packages/reid_experimental_combined_safe_080",
        "packages/thesis_assets",
        "manifests",
        "configs",
        "summaries",
    ]
    for item in dirs:
        (root / item).mkdir(parents=True, exist_ok=True)
    return root


def save_resolved_config(config: Dict[str, Any], config_path: Path, output_root: Path) -> None:
    """Save config copies into output root."""
    write_yaml(config, output_root / "configs" / "final_freeze_v2_resolved.yaml")
    if Path(config_path).exists():
        target = output_root / "configs" / Path(config_path).name
        target.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

