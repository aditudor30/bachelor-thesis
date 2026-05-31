"""Manifest helpers for YOLO curriculum exports."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


CURRICULUM_MANIFEST_FIELDS = [
    "curriculum",
    "split",
    "scene_name",
    "scene_id",
    "camera_id",
    "frame_id",
    "image_path",
    "label_path",
    "class_counts_json",
    "selected_target_classes_json",
    "max_area_norm",
    "mean_area_norm",
    "difficulties_json",
    "score",
    "source",
    "contains_person_only",
    "contains_rare_class",
]


def write_curriculum_manifest(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Write curriculum manifest CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURRICULUM_MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_manifest_row(record))


def read_curriculum_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read curriculum manifest CSV."""
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(_parse_manifest_row(row))
    return rows


def summarize_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Summarize manifest rows."""
    rows = read_curriculum_manifest(path)
    per_class = {}
    per_scene = {}
    per_camera = {}
    per_difficulty = {}
    person_only = 0
    rare = 0
    for row in rows:
        per_scene[row["scene_name"]] = per_scene.get(row["scene_name"], 0) + 1
        per_camera[row["camera_id"]] = per_camera.get(row["camera_id"], 0) + 1
        if row["contains_person_only"]:
            person_only += 1
        if row["contains_rare_class"]:
            rare += 1
        for class_name, count in row["class_counts"].items():
            per_class[class_name] = per_class.get(class_name, 0) + int(count)
        for difficulty, count in row["difficulties"].items():
            per_difficulty[difficulty] = per_difficulty.get(difficulty, 0) + int(count)
    return {
        "total_images": len(rows),
        "per_class_counts": per_class,
        "per_scene_counts": per_scene,
        "per_camera_counts": per_camera,
        "per_difficulty_counts": per_difficulty,
        "person_only_frames": person_only,
        "rare_class_frames": rare,
    }


def check_manifest_for_duplicates(path: Union[str, Path]) -> Dict[str, Any]:
    """Check duplicate frame keys in a manifest."""
    rows = read_curriculum_manifest(path)
    counts = {}
    duplicates = []
    for row in rows:
        key = (row["split"], row["scene_name"], row["camera_id"], int(row["frame_id"]))
        counts[key] = counts.get(key, 0) + 1
    for key, count in counts.items():
        if count > 1:
            duplicates.append({"key": list(key), "count": int(count)})
    return {
        "num_rows": len(rows),
        "num_duplicates": len(duplicates),
        "duplicates": duplicates,
    }


def _manifest_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for field in CURRICULUM_MANIFEST_FIELDS:
        row[field] = record.get(field)
    return row


def _parse_manifest_row(row: Dict[str, str]) -> Dict[str, Any]:
    parsed = dict(row)
    for field in ["scene_id", "frame_id"]:
        if parsed.get(field, "") != "":
            parsed[field] = int(float(parsed[field]))
    for field in ["max_area_norm", "mean_area_norm", "score"]:
        if parsed.get(field, "") != "":
            parsed[field] = float(parsed[field])
    for field in ["contains_person_only", "contains_rare_class"]:
        parsed[field] = str(parsed.get(field, "")).lower() in ("true", "1", "yes")
    parsed["class_counts"] = _loads_json_dict(parsed.get("class_counts_json"))
    parsed["selected_target_classes"] = _loads_json_list(parsed.get("selected_target_classes_json"))
    parsed["difficulties"] = _loads_json_dict(parsed.get("difficulties_json"))
    return parsed


def _loads_json_dict(value: Any) -> Dict[str, Any]:
    try:
        data = json.loads(value)
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _loads_json_list(value: Any) -> List[Any]:
    try:
        data = json.loads(value)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return []

