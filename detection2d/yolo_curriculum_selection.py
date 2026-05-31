"""Select frame-level samples for YOLO curriculum exports."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from deep_oc_sort_3d.detection2d.yolo_bbox_audit import load_bbox_audit_csv


def load_audit_csv(path: Union[str, Path]) -> Any:
    """Load bbox audit CSV as pandas DataFrame when available, else rows."""
    rows = load_bbox_audit_csv(Path(path))
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)


def load_class_rich_frames(path: Union[str, Path]) -> Any:
    """Load class-rich frame CSV as pandas DataFrame when available, else rows."""
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(_parse_class_rich_row(row))
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)


def filter_audit_by_curriculum(
    audit_records: Any,
    curriculum: str,
    allowed_difficulties: List[str],
    min_area_norm: Optional[float],
    target_classes: Optional[List[str]],
    allowed_scenes: Optional[List[str]],
    allowed_cameras: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Filter audit records by curriculum constraints."""
    rows = _to_rows(audit_records)
    difficulty_set = set(allowed_difficulties)
    class_set = None if target_classes is None else set(target_classes)
    scene_set = None if allowed_scenes is None else set(allowed_scenes)
    camera_set = None if allowed_cameras is None else set(allowed_cameras)
    filtered = []
    for row in rows:
        if row.get("difficulty") not in difficulty_set:
            continue
        if min_area_norm is not None and float(row.get("area_norm", 0.0)) < float(min_area_norm):
            continue
        if class_set is not None and row.get("class_name") not in class_set:
            continue
        if scene_set is not None and row.get("scene_name") not in scene_set:
            continue
        if camera_set is not None and row.get("camera_id") not in camera_set:
            continue
        filtered.append(row)
    return filtered


def score_frame_for_curriculum(
    frame_records: List[Dict[str, Any]],
    target_classes: List[str],
    class_priority: Dict[str, float],
    difficulty_weights: Dict[str, float],
    area_weight: float = 1.0,
    rare_class_bonus: float = 2.0,
) -> float:
    """Score one frame using class priority, difficulty, and bbox area."""
    target_set = set(target_classes)
    score = 0.0
    for row in frame_records:
        class_name = str(row.get("class_name", ""))
        difficulty = str(row.get("difficulty", "hard"))
        area_norm = float(row.get("area_norm", 0.0))
        class_weight = float(class_priority.get(class_name, 1.0))
        difficulty_weight = float(difficulty_weights.get(difficulty, 0.5))
        score += class_weight * difficulty_weight * (1.0 + float(area_weight) * area_norm)
        if class_name in target_set and class_name != "Person":
            score += float(rare_class_bonus)
    return float(score)


def select_curriculum_frames(
    audit_csv: Union[str, Path],
    class_rich_frames_csv: Optional[Union[str, Path]],
    curriculum: str,
    target_classes: List[str],
    allowed_difficulties: List[str],
    class_priority: Dict[str, float],
    scene_priority: Optional[Dict[str, List[str]]] = None,
    camera_priority: Optional[Dict[str, List[str]]] = None,
    max_frames_total: Optional[int] = None,
    max_frames_per_class: Optional[int] = None,
    max_person_only_frames: int = 0,
    min_area_norm: Optional[float] = None,
    exclude_scenes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Select ranked frame records for a curriculum export."""
    audit_rows = _to_rows(load_audit_csv(audit_csv))
    exclude_set = set(exclude_scenes or [])
    allowed_rows = []
    for row in audit_rows:
        if row.get("scene_name") in exclude_set:
            continue
        if row.get("difficulty") not in set(allowed_difficulties):
            continue
        if min_area_norm is not None and float(row.get("area_norm", 0.0)) < float(min_area_norm):
            continue
        allowed_rows.append(row)

    grouped = _group_by_frame(allowed_rows)
    class_rich_keys = _class_rich_keys(class_rich_frames_csv, exclude_set, curriculum)
    difficulty_weights = _difficulty_weights(curriculum)
    candidates = []
    for key, frame_records in grouped.items():
        class_counts = _class_counts(frame_records)
        selected_targets = sorted([name for name in class_counts.keys() if name in set(target_classes)])
        contains_person_only = set(class_counts.keys()) == set(["Person"])
        source = "class_rich" if key in class_rich_keys else "audit"
        score = score_frame_for_curriculum(
            frame_records,
            target_classes,
            class_priority,
            difficulty_weights,
        )
        score += _priority_bonus(frame_records, scene_priority, camera_priority)
        if source == "class_rich":
            score += 10.0
        candidates.append(_frame_candidate(key, frame_records, class_counts, selected_targets, score, source, contains_person_only))

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = []
    class_frame_counts = {}
    person_only_count = 0
    for candidate in candidates:
        if max_frames_total is not None and len(selected) >= int(max_frames_total):
            break
        if candidate["contains_person_only"]:
            if person_only_count >= int(max_person_only_frames):
                continue
        if max_frames_per_class is not None:
            frame_classes = set(json.loads(candidate["class_counts_json"]).keys())
            target_frame_classes = frame_classes.intersection(set(target_classes))
            if target_frame_classes:
                if all(class_frame_counts.get(name, 0) >= int(max_frames_per_class) for name in target_frame_classes):
                    continue
        selected.append(candidate)
        if candidate["contains_person_only"]:
            person_only_count += 1
        for class_name in json.loads(candidate["class_counts_json"]).keys():
            if class_name in set(target_classes):
                class_frame_counts[class_name] = class_frame_counts.get(class_name, 0) + 1

    _print_selection_warnings(selected, target_classes)
    return selected


def _to_rows(data: Any) -> List[Dict[str, Any]]:
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return list(data)


def _parse_class_rich_row(row: Dict[str, str]) -> Dict[str, Any]:
    parsed = dict(row)
    for field in ["frame_id", "num_target_objects"]:
        if field in parsed and parsed[field] != "":
            parsed[field] = int(float(parsed[field]))
    for field in ["max_area_norm", "mean_area_norm"]:
        if field in parsed and parsed[field] != "":
            parsed[field] = float(parsed[field])
    for field in ["recommended_for_easy_export", "recommended_for_medium_export"]:
        if field in parsed:
            parsed[field] = str(parsed[field]).lower() in ("true", "1", "yes")
    return parsed


def _frame_key(row: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (str(row["split"]), str(row["scene_name"]), str(row["camera_id"]), int(row["frame_id"]))


def _group_by_frame(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]:
    grouped = {}
    for row in rows:
        key = _frame_key(row)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    return grouped


def _class_rich_keys(path: Optional[Union[str, Path]], exclude_scenes: Set[str], curriculum: str) -> Set[Tuple[str, str, str, int]]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    rows = _to_rows(load_class_rich_frames(p))
    keys = set()
    flag_name = "recommended_for_easy_export"
    if curriculum != "easy_allclass":
        flag_name = "recommended_for_medium_export"
    for row in rows:
        if row.get("scene_name") in exclude_scenes:
            continue
        if flag_name in row and not bool(row.get(flag_name)):
            continue
        keys.add((str(row["split"]), str(row["scene_name"]), str(row["camera_id"]), int(row["frame_id"])))
    return keys


def _difficulty_weights(curriculum: str) -> Dict[str, float]:
    if curriculum == "easy_allclass":
        return {"easy": 3.0, "medium": 1.0, "hard": 0.2}
    return {"easy": 2.0, "medium": 1.5, "hard": 0.2}


def _priority_bonus(
    frame_records: List[Dict[str, Any]],
    scene_priority: Optional[Dict[str, List[str]]],
    camera_priority: Optional[Dict[str, List[str]]],
) -> float:
    score = 0.0
    for row in frame_records:
        class_name = str(row.get("class_name"))
        scene_name = str(row.get("scene_name"))
        camera_id = str(row.get("camera_id"))
        if scene_priority is not None and scene_name in scene_priority.get(class_name, []):
            score += 3.0
        if camera_priority is not None and camera_id in camera_priority.get(class_name, []):
            score += 3.0
    return score


def _class_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        class_name = str(row["class_name"])
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def _difficulty_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        difficulty = str(row["difficulty"])
        counts[difficulty] = counts.get(difficulty, 0) + 1
    return counts


def _frame_candidate(
    key: Tuple[str, str, str, int],
    frame_records: List[Dict[str, Any]],
    class_counts: Dict[str, int],
    selected_targets: List[str],
    score: float,
    source: str,
    contains_person_only: bool,
) -> Dict[str, Any]:
    areas = [float(row.get("area_norm", 0.0)) for row in frame_records]
    return {
        "curriculum": "",
        "split": key[0],
        "scene_name": key[1],
        "camera_id": key[2],
        "frame_id": key[3],
        "class_counts_json": json.dumps(class_counts, sort_keys=True),
        "selected_target_classes_json": json.dumps(selected_targets, sort_keys=True),
        "max_area_norm": float(max(areas)) if areas else 0.0,
        "mean_area_norm": float(np.mean(np.asarray(areas, dtype=float))) if areas else 0.0,
        "difficulties_json": json.dumps(_difficulty_counts(frame_records), sort_keys=True),
        "score": float(score),
        "source": source,
        "contains_person_only": bool(contains_person_only),
        "contains_rare_class": any(name != "Person" for name in class_counts.keys()),
    }


def _print_selection_warnings(selected: List[Dict[str, Any]], target_classes: List[str]) -> None:
    counts = {}
    for item in selected:
        class_counts = json.loads(item["class_counts_json"])
        for class_name in class_counts.keys():
            counts[class_name] = counts.get(class_name, 0) + 1
    for class_name in target_classes:
        if counts.get(class_name, 0) == 0:
            print("warning: no selected frames for class %s" % class_name)
