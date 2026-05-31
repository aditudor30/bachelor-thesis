"""Summary helpers for YOLO curriculum exports."""

import json
from pathlib import Path
from typing import Any, Dict, Union


def print_curriculum_summary(summary: Dict[str, Any]) -> None:
    """Print a compact curriculum summary."""
    print("curriculum: %s" % summary.get("curriculum"))
    print("total_images: %s" % summary.get("total_images"))
    print("total_labels: %s" % summary.get("total_labels"))
    print("total_objects: %s" % summary.get("total_objects"))
    print("per_class_counts: %s" % summary.get("per_class_counts", {}))
    print("per_class_images: %s" % summary.get("per_class_images", {}))
    print("per_difficulty_counts: %s" % summary.get("per_difficulty_counts", {}))
    print("per_scene_counts: %s" % summary.get("per_scene_counts", {}))
    print("per_camera_counts: %s" % summary.get("per_camera_counts", {}))
    print("missing_classes: %s" % summary.get("missing_classes", []))
    print("person_only_frames: %s" % summary.get("person_only_frames"))
    print("rare_class_frames: %s" % summary.get("rare_class_frames"))


def save_curriculum_summary(summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save curriculum summary JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def compare_curriculum_summaries(summary_a: Dict[str, Any], summary_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two curriculum summaries."""
    class_names = sorted(
        set(summary_a.get("per_class_counts", {}).keys()).union(set(summary_b.get("per_class_counts", {}).keys()))
    )
    per_class_delta = {}
    for class_name in class_names:
        per_class_delta[class_name] = int(summary_b.get("per_class_counts", {}).get(class_name, 0)) - int(
            summary_a.get("per_class_counts", {}).get(class_name, 0)
        )
    return {
        "curriculum_a": summary_a.get("curriculum"),
        "curriculum_b": summary_b.get("curriculum"),
        "delta_images": int(summary_b.get("total_images", 0)) - int(summary_a.get("total_images", 0)),
        "delta_objects": int(summary_b.get("total_objects", 0)) - int(summary_a.get("total_objects", 0)),
        "per_class_delta": per_class_delta,
    }

