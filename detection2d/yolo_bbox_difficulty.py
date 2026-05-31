"""Scale-based bbox difficulty helpers for YOLO dataset audits.

The difficulty label is only a proxy based on bbox size. It does not measure
occlusion, truncation, blur, pose, or semantic ambiguity.
"""

from typing import Any, Dict, List, Optional


def default_difficulty_config() -> Dict[str, Any]:
    """Return default scale thresholds for easy/medium/hard bins."""
    return {
        "easy_area_norm": 0.015,
        "medium_area_norm": 0.004,
        "easy_min_side": 48.0,
        "medium_min_side": 24.0,
    }


def classify_bbox_difficulty(
    width_px: float,
    height_px: float,
    area_norm: float,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify bbox scale as easy, medium, or hard."""
    cfg = default_difficulty_config()
    if config is not None:
        cfg.update(config)
    min_side = min(float(width_px), float(height_px))
    if float(area_norm) >= float(cfg["easy_area_norm"]) and min_side >= float(cfg["easy_min_side"]):
        return "easy"
    if float(area_norm) >= float(cfg["medium_area_norm"]) and min_side >= float(cfg["medium_min_side"]):
        return "medium"
    return "hard"


def difficulty_summary(records: List[Any]) -> Dict[str, Any]:
    """Summarize difficulty counts overall and per class."""
    counts = {"easy": 0, "medium": 0, "hard": 0}
    per_class = {}
    for record in records:
        difficulty = str(getattr(record, "difficulty", "hard"))
        class_name = str(getattr(record, "class_name", "unknown"))
        if difficulty not in counts:
            counts[difficulty] = 0
        counts[difficulty] += 1
        if class_name not in per_class:
            per_class[class_name] = {"easy": 0, "medium": 0, "hard": 0}
        if difficulty not in per_class[class_name]:
            per_class[class_name][difficulty] = 0
        per_class[class_name][difficulty] += 1
    return {
        "count_per_difficulty": counts,
        "per_class": per_class,
    }

