"""Plan YOLO all-class train/holdout/validation scene splits."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from deep_oc_sort_3d.detection2d.yolo_class_audit import (
    DEFAULT_CLASS_MAPPING,
    count_gt_visible_boxes_by_class,
    summarize_class_distribution,
)


def audit_scene_class_coverage(
    root: Union[str, Path],
    split: str,
    scenes: List[str],
    camera_id: Optional[str] = None,
    frame_stride: int = 1,
    max_frames_per_scene: Optional[int] = None,
) -> Dict[str, Any]:
    """Audit visible GT class coverage for a scene list."""
    return count_gt_visible_boxes_by_class(
        root=root,
        split=split,
        scenes=scenes,
        camera_id=camera_id,
        max_frames_per_scene=max_frames_per_scene,
        frame_stride=frame_stride,
    )


def suggest_train_holdout_split(
    coverage: Dict[str, Any],
    holdout_scenes: Optional[List[str]] = None,
    target_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Suggest train/holdout scene lists and report missing-class warnings."""
    if target_classes is None:
        target_classes = list(DEFAULT_CLASS_MAPPING.keys())
    if holdout_scenes is None:
        holdout_scenes = []
    per_scene = coverage.get("per_scene", {})
    all_scenes = sorted(per_scene.keys())
    holdout_set = set(holdout_scenes)
    train_scenes = [scene for scene in all_scenes if scene not in holdout_set]
    train_counts = _sum_scene_counts(per_scene, train_scenes)
    holdout_counts = _sum_scene_counts(per_scene, holdout_scenes)
    warnings = []
    for class_name in target_classes:
        if int(train_counts.get(class_name, 0)) == 0:
            warnings.append("missing in train: %s" % class_name)
        if holdout_scenes and int(holdout_counts.get(class_name, 0)) == 0:
            warnings.append("missing in holdout: %s" % class_name)
    return {
        "train_scenes": train_scenes,
        "holdout_scenes": list(holdout_scenes),
        "warnings": warnings,
        "coverage_summary": {
            "train": train_counts,
            "holdout": holdout_counts,
        },
    }


def validate_no_scene_overlap(
    train_scenes: List[str],
    holdout_scenes: List[str],
    val_scenes: List[str],
) -> bool:
    """Return True when train, holdout, and official val scenes are disjoint."""
    train_set = set(train_scenes)
    holdout_set = set(holdout_scenes)
    val_set = set(val_scenes)
    if train_set.intersection(holdout_set):
        return False
    if train_set.intersection(val_set):
        return False
    if holdout_set.intersection(val_set):
        return False
    return True


def summarize_split_coverage(split_coverages: Dict[str, Dict[str, Any]]) -> str:
    """Render class coverage for multiple named splits."""
    lines = []
    for split_name in sorted(split_coverages.keys()):
        lines.append("[%s]" % split_name)
        lines.append(summarize_class_distribution(split_coverages[split_name]))
    return "\n\n".join(lines)


def _sum_scene_counts(per_scene: Dict[str, Any], scenes: List[str]) -> Dict[str, int]:
    counts = {}
    for class_name in DEFAULT_CLASS_MAPPING.keys():
        counts[class_name] = 0
    for scene_name in scenes:
        scene_stats = per_scene.get(scene_name, {})
        scene_counts = scene_stats.get("class_counts", {})
        for class_name, value in scene_counts.items():
            counts[class_name] = counts.get(class_name, 0) + int(value)
    return counts


def build_split_plan_dict(
    train_scenes: List[str],
    holdout_scenes: List[str],
    official_val_scenes: List[str],
    train_coverage: Dict[str, Any],
    holdout_coverage: Dict[str, Any],
    official_val_coverage: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a serializable split-plan dictionary."""
    return {
        "train_scenes": list(train_scenes),
        "internal_holdout_scenes": list(holdout_scenes),
        "official_val_scenes": list(official_val_scenes),
        "no_scene_overlap": validate_no_scene_overlap(train_scenes, holdout_scenes, official_val_scenes),
        "coverage": {
            "train": train_coverage,
            "internal_holdout": holdout_coverage,
            "official_val": official_val_coverage,
        },
    }

