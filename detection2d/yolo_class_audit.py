"""Class-distribution audit helpers for SmartSpaces YOLO data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.detection2d.yolo_label_utils import read_yolo_label_file


DEFAULT_CLASS_MAPPING = {
    "Person": 0,
    "Forklift": 1,
    "PalletTruck": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "NovaCarter": 6,
}


def count_gt_visible_boxes_by_class(
    root: Union[str, Path],
    split: str,
    scenes: List[str],
    camera_id: Optional[str] = None,
    max_frames_per_scene: Optional[int] = None,
    frame_stride: int = 1,
) -> Dict[str, Any]:
    """Count visible GT boxes by class, scene, and camera."""
    root_path = Path(root)
    stride = max(int(frame_stride), 1)
    result = {
        "split": split,
        "camera_id": camera_id,
        "class_counts": _zero_class_counts(),
        "per_scene": {},
        "per_camera": {},
        "total_visible_boxes": 0,
        "total_frames_with_visible_boxes": 0,
        "missing_classes": [],
    }
    for scene_name in scenes:
        objects = _load_scene_gt(root_path, split, scene_name)
        scene_counts = _zero_class_counts()
        frames_with_boxes = set()
        for obj in objects:
            if not _frame_is_selected(obj.frame_id, max_frames_per_scene, stride):
                continue
            cameras = _visible_cameras_for_object(obj, camera_id)
            if not cameras:
                continue
            for cam in cameras:
                _increment_count(result["class_counts"], obj.object_type)
                _increment_count(scene_counts, obj.object_type)
                if cam not in result["per_camera"]:
                    result["per_camera"][cam] = _zero_class_counts()
                _increment_count(result["per_camera"][cam], obj.object_type)
                result["total_visible_boxes"] += 1
                frames_with_boxes.add((obj.frame_id, cam))
        result["per_scene"][scene_name] = {
            "class_counts": scene_counts,
            "frames_with_visible_boxes": len(frames_with_boxes),
        }
        result["total_frames_with_visible_boxes"] += len(frames_with_boxes)
    result["missing_classes"] = _missing_classes(result["class_counts"])
    return result


def count_yolo_labels_by_class(
    yolo_dataset_dir: Union[str, Path],
    split: str,
) -> Dict[str, Any]:
    """Count exported YOLO labels by class for one split."""
    dataset_dir = Path(yolo_dataset_dir)
    names = _load_yolo_names(dataset_dir / "data.yaml")
    labels_dir = dataset_dir / "labels" / split
    images_dir = dataset_dir / "images" / split
    class_counts = _zero_class_counts(names)
    image_class_counts = {}
    images_with_labels = 0
    empty_images = 0
    missing_label_files = 0
    total_labels = 0

    for image_path in sorted(images_dir.glob("*.*")):
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            missing_label_files += 1
            empty_images += 1
            image_class_counts[str(image_path)] = {}
            continue
        labels = read_yolo_label_file(label_path)
        if labels:
            images_with_labels += 1
        else:
            empty_images += 1
        counts_for_image = {}
        for label in labels:
            class_name = names.get(int(label.class_id), str(label.class_id))
            _increment_count(class_counts, class_name)
            _increment_count(counts_for_image, class_name)
            total_labels += 1
        image_class_counts[str(image_path)] = counts_for_image

    return {
        "dataset": str(dataset_dir),
        "split": split,
        "names": names,
        "class_counts": class_counts,
        "total_labels": total_labels,
        "num_images": len(list(images_dir.glob("*.*"))),
        "images_with_labels": images_with_labels,
        "empty_images": empty_images,
        "missing_label_files": missing_label_files,
        "missing_classes": _missing_classes(class_counts),
        "image_class_counts": image_class_counts,
    }


def find_frames_with_classes(
    root: Union[str, Path],
    split: str,
    scenes: List[str],
    target_classes: List[str],
    camera_id: Optional[str] = None,
    max_frames_per_scene: Optional[int] = None,
    frame_stride: int = 1,
) -> List[Dict[str, Any]]:
    """Find frame/camera pairs that contain at least one target class."""
    root_path = Path(root)
    target_set = set(str(name) for name in target_classes)
    stride = max(int(frame_stride), 1)
    frames = {}
    for scene_name in scenes:
        objects = _load_scene_gt(root_path, split, scene_name)
        for obj in objects:
            if not _frame_is_selected(obj.frame_id, max_frames_per_scene, stride):
                continue
            cameras = _visible_cameras_for_object(obj, camera_id)
            if not cameras:
                continue
            for cam in cameras:
                key = (scene_name, obj.frame_id, cam)
                if key not in frames:
                    frames[key] = {
                        "split": split,
                        "scene_name": scene_name,
                        "frame_id": int(obj.frame_id),
                        "camera_id": cam,
                        "class_counts": {},
                        "target_class_counts": {},
                    }
                _increment_count(frames[key]["class_counts"], obj.object_type)
                if obj.object_type in target_set:
                    _increment_count(frames[key]["target_class_counts"], obj.object_type)
    selected = [item for item in frames.values() if item["target_class_counts"]]
    return sorted(selected, key=lambda item: (item["scene_name"], item["camera_id"], item["frame_id"]))


def summarize_class_distribution(counts: Dict[str, Any]) -> str:
    """Return a readable class-distribution summary."""
    class_counts = counts.get("class_counts", {})
    total = float(sum(int(value) for value in class_counts.values()))
    lines = []
    lines.append("total labels/boxes: %d" % int(total))
    for class_name in sorted(class_counts.keys()):
        value = int(class_counts[class_name])
        pct = 0.0 if total <= 0.0 else float(value) / total * 100.0
        lines.append("  %s: %d (%.2f%%)" % (class_name, value, pct))
    missing = _missing_classes(class_counts)
    present_values = [int(value) for value in class_counts.values() if int(value) > 0]
    imbalance = None
    if present_values:
        imbalance = float(max(present_values)) / float(max(min(present_values), 1))
    lines.append("missing classes: %s" % (", ".join(missing) if missing else "none"))
    lines.append("imbalance ratio: %s" % ("None" if imbalance is None else "%.3f" % imbalance))
    return "\n".join(lines)


def top_images_with_classes(
    yolo_counts: Dict[str, Any],
    target_classes: List[str],
    limit: int = 10,
) -> List[Tuple[str, Dict[str, int]]]:
    """Return image paths that contain the requested classes."""
    target_set = set(target_classes)
    rows = []
    image_counts = yolo_counts.get("image_class_counts", {})
    for image_path, counts in image_counts.items():
        score = sum(int(counts.get(class_name, 0)) for class_name in target_set)
        if score <= 0:
            continue
        rows.append((image_path, counts, score))
    rows = sorted(rows, key=lambda item: item[2], reverse=True)
    return [(row[0], row[1]) for row in rows[: int(limit)]]


def _load_scene_gt(root: Path, split: str, scene_name: str) -> List[GroundTruthObject]:
    scene_paths = get_scene_paths(root, split, scene_name)
    path = scene_paths.ground_truth_path
    if path is None or not path.exists():
        return []
    return load_ground_truth_json(path)


def _frame_is_selected(frame_id: int, max_frames_per_scene: Optional[int], frame_stride: int) -> bool:
    if max_frames_per_scene is not None and int(frame_id) >= int(max_frames_per_scene):
        return False
    return int(frame_id) % max(int(frame_stride), 1) == 0


def _visible_cameras_for_object(obj: GroundTruthObject, camera_id: Optional[str]) -> List[str]:
    if camera_id is not None:
        if camera_id in obj.visible_bboxes_2d:
            return [camera_id]
        return []
    return sorted(obj.visible_bboxes_2d.keys())


def _zero_class_counts(names: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    if names is None:
        names = {value: key for key, value in DEFAULT_CLASS_MAPPING.items()}
    counts = {}
    for _class_id, class_name in names.items():
        counts[str(class_name)] = 0
    return counts


def _increment_count(counts: Dict[str, int], class_name: str) -> None:
    key = str(class_name)
    if key not in counts:
        counts[key] = 0
    counts[key] += 1


def _missing_classes(class_counts: Dict[str, int]) -> List[str]:
    missing = []
    for class_name in DEFAULT_CLASS_MAPPING.keys():
        if int(class_counts.get(class_name, 0)) == 0:
            missing.append(class_name)
    return missing


def _load_yolo_names(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {value: key for key, value in DEFAULT_CLASS_MAPPING.items()}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names_raw = data.get("names", {}) if isinstance(data, dict) else {}
    names = {}
    if isinstance(names_raw, list):
        for idx, value in enumerate(names_raw):
            names[int(idx)] = str(value)
    elif isinstance(names_raw, dict):
        for key, value in names_raw.items():
            names[int(key)] = str(value)
    if not names:
        names = {value: key for key, value in DEFAULT_CLASS_MAPPING.items()}
    return names


def counts_to_csv_rows(counts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert audit counts to flat CSV rows."""
    rows = []
    per_scene = counts.get("per_scene", {})
    for scene_name, scene_stats in per_scene.items():
        class_counts = scene_stats.get("class_counts", {})
        for class_name, value in class_counts.items():
            rows.append(
                {
                    "scope": "scene",
                    "name": scene_name,
                    "class_name": class_name,
                    "count": int(value),
                }
            )
    per_camera = counts.get("per_camera", {})
    for camera_name, class_counts in per_camera.items():
        for class_name, value in class_counts.items():
            rows.append(
                {
                    "scope": "camera",
                    "name": camera_name,
                    "class_name": class_name,
                    "count": int(value),
                }
            )
    for class_name, value in counts.get("class_counts", {}).items():
        rows.append({"scope": "total", "name": "all", "class_name": class_name, "count": int(value)})
    return rows


def counts_to_json_text(counts: Dict[str, int]) -> str:
    """Serialize class counts with deterministic key order."""
    return json.dumps(counts, sort_keys=True)

