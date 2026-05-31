"""Export YOLO curriculum datasets from bbox audit CSV selections."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import yaml

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths, scene_name_to_id
from deep_oc_sort_3d.data.frame_io import infer_camera_id_from_video_path, list_video_files, safe_read_video_frame
from deep_oc_sort_3d.detection2d.yolo_curriculum_manifest import write_curriculum_manifest
from deep_oc_sort_3d.detection2d.yolo_curriculum_selection import load_audit_csv, select_curriculum_frames
from deep_oc_sort_3d.detection2d.yolo_curriculum_summary import save_curriculum_summary
from deep_oc_sort_3d.detection2d.yolo_dataset_exporter import DEFAULT_CLASS_MAPPING
from deep_oc_sort_3d.detection2d.yolo_label_utils import write_yolo_label_file, xyxy_to_yolo_norm
from deep_oc_sort_3d.detection2d.yolo_types import YoloExportRecord, YoloLabel


DEFAULT_TARGET_CLASSES = [
    "Forklift",
    "PalletTruck",
    "Transporter",
    "FourierGR1T2",
    "AgilityDigit",
    "NovaCarter",
    "Person",
]


class YoloCurriculumExporter:
    """Export selected curriculum frames to a YOLO train dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        output_dir: Union[str, Path],
        audit_csv: Union[str, Path],
        class_rich_frames_csv: Optional[Union[str, Path]] = None,
        curriculum: str = "easy_allclass",
        class_mapping: Optional[Dict[str, int]] = None,
        allowed_difficulties: Optional[List[str]] = None,
        target_classes: Optional[List[str]] = None,
        class_priority: Optional[Dict[str, float]] = None,
        scene_priority: Optional[Dict[str, List[str]]] = None,
        camera_priority: Optional[Dict[str, List[str]]] = None,
        exclude_scenes: Optional[List[str]] = None,
        max_frames_total: Optional[int] = None,
        max_frames_per_class: Optional[int] = None,
        max_person_only_frames: int = 500,
        min_area_norm: Optional[float] = None,
        include_empty_frames: bool = False,
        jpeg_quality: int = 95,
        include_all_visible_objects: bool = True,
    ):
        self.root = Path(root)
        self.output_dir = Path(output_dir)
        self.audit_csv = Path(audit_csv)
        self.class_rich_frames_csv = None if class_rich_frames_csv is None else Path(class_rich_frames_csv)
        self.curriculum = curriculum
        self.class_mapping = dict(class_mapping) if class_mapping is not None else dict(DEFAULT_CLASS_MAPPING)
        self.allowed_difficulties = allowed_difficulties if allowed_difficulties is not None else _default_difficulties(curriculum)
        self.target_classes = target_classes if target_classes is not None else list(DEFAULT_TARGET_CLASSES)
        self.class_priority = class_priority if class_priority is not None else _default_class_priority()
        self.scene_priority = scene_priority
        self.camera_priority = camera_priority
        self.exclude_scenes = exclude_scenes
        self.max_frames_total = max_frames_total
        self.max_frames_per_class = max_frames_per_class
        self.max_person_only_frames = max_person_only_frames
        self.min_area_norm = min_area_norm if min_area_norm is not None else _default_min_area(curriculum)
        self.include_empty_frames = include_empty_frames
        self.jpeg_quality = int(jpeg_quality)
        self.include_all_visible_objects = include_all_visible_objects
        self.audit_rows = _rows_from_audit(load_audit_csv(self.audit_csv))
        self.audit_by_frame = _group_audit_by_frame(self.audit_rows)
        self.video_path_cache = {}
        self.manifest_rows = []
        self.records = []

    def export(self) -> List[YoloExportRecord]:
        """Export selected frames to images/train and labels/train."""
        selected_frames = select_curriculum_frames(
            audit_csv=self.audit_csv,
            class_rich_frames_csv=self.class_rich_frames_csv,
            curriculum=self.curriculum,
            target_classes=self.target_classes,
            allowed_difficulties=self.allowed_difficulties,
            class_priority=self.class_priority,
            scene_priority=self.scene_priority,
            camera_priority=self.camera_priority,
            max_frames_total=self.max_frames_total,
            max_frames_per_class=self.max_frames_per_class,
            max_person_only_frames=self.max_person_only_frames,
            min_area_norm=self.min_area_norm,
            exclude_scenes=self.exclude_scenes,
        )
        self.records = []
        self.manifest_rows = []
        for frame in selected_frames:
            record = self._export_frame(frame)
            if record is not None:
                self.records.append(record)
        self.write_data_yaml()
        self.write_manifest(self.records)
        summary = self.summarize(self.records)
        save_curriculum_summary(summary, self.output_dir / "curriculum_summary.json")
        return list(self.records)

    def write_data_yaml(self) -> None:
        """Write YOLO data.yaml."""
        names = {}
        for class_name, class_id in self.class_mapping.items():
            names[int(class_id)] = class_name
        data = {
            "path": str(self.output_dir),
            "train": "images/train",
            "val": "images/train",
            "names": names,
        }
        path = self.output_dir / "data.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")

    def write_manifest(self, records: List[YoloExportRecord]) -> None:
        """Write curriculum manifest CSV."""
        write_curriculum_manifest(self.manifest_rows, self.output_dir / "curriculum_manifest.csv")

    def summarize(self, records: List[YoloExportRecord]) -> Dict[str, Any]:
        """Summarize exported records and manifest rows."""
        per_class_counts = {}
        per_class_images = {}
        per_difficulty_counts = {}
        per_scene_counts = {}
        per_camera_counts = {}
        total_objects = 0
        person_only = 0
        rare = 0
        for row in self.manifest_rows:
            class_counts = json.loads(row["class_counts_json"])
            difficulties = json.loads(row["difficulties_json"])
            per_scene_counts[row["scene_name"]] = per_scene_counts.get(row["scene_name"], 0) + 1
            per_camera_counts[row["camera_id"]] = per_camera_counts.get(row["camera_id"], 0) + 1
            if row["contains_person_only"]:
                person_only += 1
            if row["contains_rare_class"]:
                rare += 1
            for class_name, count in class_counts.items():
                per_class_counts[class_name] = per_class_counts.get(class_name, 0) + int(count)
                per_class_images[class_name] = per_class_images.get(class_name, 0) + 1
                total_objects += int(count)
            for difficulty, count in difficulties.items():
                per_difficulty_counts[difficulty] = per_difficulty_counts.get(difficulty, 0) + int(count)
        missing = []
        for class_name in self.class_mapping.keys():
            if per_class_counts.get(class_name, 0) == 0:
                missing.append(class_name)
        return {
            "curriculum": self.curriculum,
            "total_images": len(records),
            "total_labels": len(records),
            "total_objects": total_objects,
            "per_class_counts": per_class_counts,
            "per_class_images": per_class_images,
            "per_difficulty_counts": per_difficulty_counts,
            "per_scene_counts": per_scene_counts,
            "per_camera_counts": per_camera_counts,
            "missing_classes": missing,
            "person_only_frames": person_only,
            "rare_class_frames": rare,
        }

    def _export_frame(self, frame: Dict[str, Any]) -> Optional[YoloExportRecord]:
        key = (frame["split"], frame["scene_name"], frame["camera_id"], int(frame["frame_id"]))
        rows = self.audit_by_frame.get(key, [])
        if not rows:
            return None
        label_rows = rows if self.include_all_visible_objects else self._filter_label_rows(rows)
        labels = self._labels_from_rows(label_rows)
        if not labels and not self.include_empty_frames:
            return None
        video_path = self._video_path(frame["split"], frame["scene_name"], frame["camera_id"])
        if video_path is None:
            print("warning: missing video for %s %s %s" % (frame["split"], frame["scene_name"], frame["camera_id"]))
            return None
        image_rgb = safe_read_video_frame(video_path, int(frame["frame_id"]))
        if image_rgb is None:
            print("warning: could not read frame %d from %s" % (int(frame["frame_id"]), video_path))
            return None
        image_path = self._image_path(frame)
        label_path = self._label_path(frame)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        write_yolo_label_file(labels, label_path)
        scene_id = scene_name_to_id(str(frame["scene_name"]))
        if scene_id is None:
            scene_id = -1
        manifest_row = dict(frame)
        manifest_row["curriculum"] = self.curriculum
        manifest_row["scene_id"] = scene_id
        manifest_row["image_path"] = str(image_path)
        manifest_row["label_path"] = str(label_path)
        manifest_row["class_counts_json"] = json.dumps(_class_counts(label_rows), sort_keys=True)
        manifest_row["difficulties_json"] = json.dumps(_difficulty_counts(label_rows), sort_keys=True)
        self.manifest_rows.append(manifest_row)
        return YoloExportRecord(
            image_path=image_path,
            label_path=label_path,
            scene_name=str(frame["scene_name"]),
            split=str(frame["split"]),
            camera_id=str(frame["camera_id"]),
            frame_id=int(frame["frame_id"]),
            num_objects=len(labels),
        )

    def _filter_label_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for row in rows:
            if row["difficulty"] in set(self.allowed_difficulties):
                filtered.append(row)
        return filtered

    def _labels_from_rows(self, rows: List[Dict[str, Any]]) -> List[YoloLabel]:
        labels = []
        for row in rows:
            class_id = int(row["class_id"])
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            yolo_box = xyxy_to_yolo_norm(box, int(row["image_width"]), int(row["image_height"]))
            if yolo_box is None:
                continue
            labels.append(YoloLabel(class_id, yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3]))
        return labels

    def _video_path(self, split: str, scene_name: str, camera_id: str) -> Optional[Path]:
        key = (split, scene_name, camera_id)
        if key in self.video_path_cache:
            return self.video_path_cache[key]
        scene_paths = get_scene_paths(self.root, split, scene_name)
        path = None
        if scene_paths.videos_dir is not None:
            for video_path in list_video_files(scene_paths.videos_dir):
                if infer_camera_id_from_video_path(video_path) == camera_id:
                    path = video_path
                    break
        self.video_path_cache[key] = path
        return path

    def _image_path(self, frame: Dict[str, Any]) -> Path:
        name = "%s_%s_%06d.jpg" % (frame["scene_name"], frame["camera_id"], int(frame["frame_id"]))
        return self.output_dir / "images" / "train" / name

    def _label_path(self, frame: Dict[str, Any]) -> Path:
        name = "%s_%s_%06d.txt" % (frame["scene_name"], frame["camera_id"], int(frame["frame_id"]))
        return self.output_dir / "labels" / "train" / name


def _rows_from_audit(data: Any) -> List[Dict[str, Any]]:
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return list(data)


def _group_audit_by_frame(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]:
    grouped = {}
    for row in rows:
        key = (str(row["split"]), str(row["scene_name"]), str(row["camera_id"]), int(row["frame_id"]))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    return grouped


def _default_difficulties(curriculum: str) -> List[str]:
    if curriculum == "easy_allclass":
        return ["easy"]
    return ["easy", "medium"]


def _default_min_area(curriculum: str) -> float:
    if curriculum == "easy_allclass":
        return 0.004
    return 0.001


def _default_class_priority() -> Dict[str, float]:
    return {
        "PalletTruck": 5.0,
        "Forklift": 4.0,
        "Transporter": 4.0,
        "FourierGR1T2": 4.0,
        "AgilityDigit": 4.0,
        "NovaCarter": 4.0,
        "Person": 1.0,
    }


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

