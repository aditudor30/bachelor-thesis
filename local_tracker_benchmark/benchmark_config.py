"""Configuration helpers for Step 21A."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


TRACKER_ALIASES = {
    "current": "current_local_tracker",
    "bytetrack": "bytetrack_style_yolo11m",
    "botsort_no_reid": "botsort_style_no_reid_yolo11m",
    "botsort_sbs_mot17": "botsort_sbs_mot17_yolo11m",
    "botsort_sbs_mot20": "botsort_sbs_mot20_yolo11m",
    "botsort_osnet": "botsort_osnet_finetuned_yolo11m",
}


def load_benchmark_config(path: Path) -> Dict[str, Any]:
    """Load benchmark YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Local tracker benchmark config must be a mapping")
    return value


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return isolated benchmark root."""
    return Path(str(config.get("local_tracker_benchmark", {}).get("output_root", "output/local_tracker_benchmark/v1")))


def resolve_scene_selection(config: Dict[str, Any], subset_name: str) -> List[Tuple[str, str, str]]:
    """Resolve CLI subset into pipeline subset, dataset split and scene triples."""
    benchmark = config.get("benchmark", {})
    mappings = []  # type: List[Tuple[str, str, str]]
    if subset_name == "quick":
        sections = benchmark.get("quick_scenes", {})
    elif subset_name == "val":
        sections = benchmark.get("val_scenes", {})
    elif subset_name == "internal_holdout":
        sections = benchmark.get("internal_holdout_scenes", {})
    elif subset_name == "test":
        sections = benchmark.get("test_scenes", {})
    elif subset_name == "all_available":
        sections = {}
        for key in ("internal_holdout_scenes", "val_scenes", "test_scenes"):
            for split, scenes in benchmark.get(key, {}).items():
                sections.setdefault(split, [])
                sections[split].extend(scenes)
    else:
        raise ValueError("Unknown benchmark subset: %s" % subset_name)
    for split, scenes in sections.items():
        pipeline_subset = _pipeline_subset(str(split))
        for scene in scenes:
            mappings.append((pipeline_subset, str(split), str(scene)))
    return sorted(set(mappings))


def enabled_tracker_names(config: Dict[str, Any], requested: Optional[List[str]] = None) -> List[str]:
    """Return enabled canonical tracker names."""
    settings = config.get("trackers", {})
    names = []
    for alias, canonical in TRACKER_ALIASES.items():
        if not bool(settings.get("run_%s" % canonical, True)):
            continue
        if requested is not None and alias not in requested and canonical not in requested:
            continue
        names.append(canonical)
    return names


def _pipeline_subset(split: str) -> str:
    return {"train": "internal_holdout", "val": "official_val", "test": "test"}.get(split, split)
