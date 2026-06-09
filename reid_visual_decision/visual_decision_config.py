"""Configuration helpers for Step 18D visual merge decision."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import ensure_dirs, load_yaml, write_yaml


DEFAULT_OUTPUT_ROOT = "output/person_reid_visual_decision/baseline_v2_pseudo3d_fullcam"
DEFAULT_SOURCE_ROOT = "output/person_reid_finetuned_association/baseline_v2_pseudo3d_fullcam"


def load_visual_decision_config(config_path: Path) -> Dict[str, Any]:
    """Load the Step 18D visual decision config."""
    data = load_yaml(Path(config_path))
    return data if isinstance(data, dict) else {}


def visual_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the main visual-decision section."""
    section = config.get("person_reid_visual_decision", config)
    return section if isinstance(section, dict) else {}


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return the Step 18D output root."""
    section = visual_section(config)
    return Path(str(section.get("output_root", DEFAULT_OUTPUT_ROOT)))


def source_root_from_config(config: Dict[str, Any]) -> Path:
    """Return the fine-tuned association source root."""
    paths = config.get("paths", {})
    return Path(str(paths.get("finetuned_association_root", DEFAULT_SOURCE_ROOT)))


def dataset_root_from_config(config: Dict[str, Any]) -> Path:
    """Return dataset root from config."""
    paths = config.get("paths", {})
    return Path(str(paths.get("dataset_root", "/path/to/MTMC_Tracking_2026")))


def variants_from_config(config: Dict[str, Any]) -> List[str]:
    """Return variants requested for visual inspection."""
    section = visual_section(config)
    variants = section.get("variants", ["threshold_080", "combined_safe_080"])
    return [str(item) for item in variants]


def max_events_for_variant(config: Dict[str, Any], variant: str) -> int:
    """Return max visual-review events for a variant."""
    section = visual_section(config)
    per_variant = section.get("max_events_per_variant", {})
    if isinstance(per_variant, dict) and variant in per_variant:
        return int(per_variant[variant])
    return int(section.get("default_max_events", 40))


def prepare_visual_output(config: Dict[str, Any], overwrite: bool = False) -> Path:
    """Create the Step 18D output tree."""
    _unused_overwrite = overwrite
    root = output_root_from_config(config)
    labels = ["likely_good", "ambiguous", "suspicious", "likely_bad", "not_enough_visual_evidence"]
    dirs = [
        root / "merge_audit",
        root / "figures",
        root / "manual_review",
        root / "comparison",
        root / "reports",
        root / "configs",
    ]
    for variant in variants_from_config(config):
        for label in labels:
            dirs.append(root / "visual_panels" / variant / label)
    ensure_dirs(dirs)
    return root


def save_visual_config(config: Dict[str, Any], config_path: Path, output_root: Path) -> None:
    """Save a resolved config copy."""
    write_yaml(config, output_root / "configs" / "person_reid_visual_decision_resolved.yaml")
    if Path(config_path).exists():
        (output_root / "configs" / Path(config_path).name).write_text(
            Path(config_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def frame_records_root_from_config(config: Dict[str, Any]) -> Path:
    """Return baseline frame_global_records root."""
    paths = config.get("paths", {})
    default_root = "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam/frame_global_records"
    return Path(str(paths.get("v2_frame_global_records_root", default_root)))

