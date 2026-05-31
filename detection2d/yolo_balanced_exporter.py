"""Balanced YOLO dataset export for rare SmartSpaces classes."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import yaml

from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths, list_scenes
from deep_oc_sort_3d.data.frame_io import (
    get_video_resolution,
    infer_camera_id_from_video_path,
    list_video_files,
    safe_read_video_frame,
)
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.detection2d.yolo_class_audit import counts_to_json_text
from deep_oc_sort_3d.detection2d.yolo_dataset_exporter import DEFAULT_CLASS_MAPPING
from deep_oc_sort_3d.detection2d.yolo_label_utils import write_yolo_label_file, xyxy_to_yolo_norm
from deep_oc_sort_3d.detection2d.yolo_types import YoloExportRecord, YoloLabel


MANIFEST_FIELDS = [
    "split",
    "scene_name",
    "camera_id",
    "frame_id",
    "image_path",
    "label_path",
    "class_counts_json",
    "is_rare_class_frame",
]


class BalancedYoloDatasetExporter:
    """Export YOLO images with extra coverage for rare classes."""

    def __init__(
        self,
        root: Union[str, Path],
        output_dir: Union[str, Path],
        class_mapping: Optional[Dict[str, int]] = None,
        image_ext: str = ".jpg",
        frame_stride: int = 1,
        max_frames_per_scene: Optional[int] = None,
        camera_id: Optional[str] = None,
        include_empty_frames: bool = False,
        jpeg_quality: int = 95,
        target_classes: Optional[List[str]] = None,
        rare_class_boost: int = 3,
        max_images_per_class: Optional[int] = None,
        min_objects_per_exported_frame: int = 1,
    ):
        self.root = Path(root)
        self.output_dir = Path(output_dir)
        self.class_mapping = dict(class_mapping) if class_mapping is not None else dict(DEFAULT_CLASS_MAPPING)
        self.image_ext = image_ext
        self.frame_stride = max(int(frame_stride), 1)
        self.max_frames_per_scene = max_frames_per_scene
        self.camera_id = camera_id
        self.include_empty_frames = include_empty_frames
        self.jpeg_quality = int(jpeg_quality)
        self.target_classes = [str(item) for item in target_classes] if target_classes is not None else None
        self.rare_class_boost = max(int(rare_class_boost), 1)
        self.max_images_per_class = max_images_per_class
        self.min_objects_per_exported_frame = max(int(min_objects_per_exported_frame), 0)
        self.manifest_rows = []
        self.ignored_empty_frames = 0
        self.saved_empty_frames = 0

    def export_split(self, split: str, scenes: Optional[List[str]] = None) -> List[YoloExportRecord]:
        """Export one train/val split."""
        if split not in ("train", "val"):
            raise ValueError("Balanced YOLO export only supports train/val splits.")
        scene_names = scenes if scenes is not None else list_scenes(self.root, split)
        records = []
        class_image_counts = {}
        for scene_name in scene_names:
            records.extend(self.export_scene(split, scene_name, class_image_counts))
        return records

    def export_scene(
        self,
        split: str,
        scene_name: str,
        class_image_counts: Optional[Dict[str, int]] = None,
    ) -> List[YoloExportRecord]:
        """Export selected frame/camera pairs for one scene."""
        if class_image_counts is None:
            class_image_counts = {}
        scene_paths = get_scene_paths(self.root, split, scene_name)
        if scene_paths.ground_truth_path is None or not scene_paths.ground_truth_path.exists():
            print("warning: missing GT for %s %s" % (split, scene_name))
            return []
        gt_objects = load_ground_truth_json(scene_paths.ground_truth_path)
        gt_by_frame = _group_gt_by_frame(gt_objects)
        video_paths = self._video_paths(scene_paths)
        calibrations = self._calibrations(scene_paths)
        camera_ids = self._camera_ids(gt_objects, video_paths)
        records = []

        for camera_id in camera_ids:
            video_path = video_paths.get(camera_id)
            if video_path is None:
                print("warning: missing video for %s %s %s" % (split, scene_name, camera_id))
                continue
            frame_ids = self._candidate_frame_ids(gt_by_frame, camera_id)
            for frame_id in frame_ids:
                image_width, image_height = self._frame_size(video_path, calibrations.get(camera_id))
                labels = self._labels_for_frame(gt_by_frame.get(frame_id, []), camera_id, image_width, image_height)
                class_counts = self._label_class_counts(labels)
                is_rare = self._is_rare_class_frame(class_counts)
                if len(labels) < self.min_objects_per_exported_frame:
                    if not self.include_empty_frames:
                        self.ignored_empty_frames += 1
                        continue
                if not labels and not self.include_empty_frames:
                    self.ignored_empty_frames += 1
                    continue
                if self._class_limits_reached(class_counts, class_image_counts):
                    continue

                frame_rgb = safe_read_video_frame(video_path, frame_id)
                if frame_rgb is None:
                    print("warning: could not read frame %d from %s" % (frame_id, video_path))
                    continue

                image_path = self._image_path(split, scene_name, camera_id, frame_id)
                label_path = self._label_path(split, scene_name, camera_id, frame_id)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                write_yolo_label_file(labels, label_path)
                if not labels:
                    self.saved_empty_frames += 1
                self._update_class_image_counts(class_counts, class_image_counts)
                records.append(
                    YoloExportRecord(
                        image_path=image_path,
                        label_path=label_path,
                        scene_name=scene_name,
                        split=split,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        num_objects=len(labels),
                    )
                )
                self.manifest_rows.append(
                    {
                        "split": split,
                        "scene_name": scene_name,
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "class_counts_json": counts_to_json_text(class_counts),
                        "is_rare_class_frame": is_rare,
                    }
                )
        return records

    def write_data_yaml(self, yaml_path: Path) -> None:
        """Write YOLO data.yaml for the balanced export."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        names = {}
        for class_name, class_id in self.class_mapping.items():
            names[int(class_id)] = class_name
        data = {
            "path": str(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")

    def write_manifest(self, path: Path) -> None:
        """Write export_manifest.csv."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in self.manifest_rows:
                writer.writerow(row)

    def class_distribution(self, records: List[YoloExportRecord]) -> Dict[str, Any]:
        """Count labels in exported records."""
        id_to_name = {int(value): key for key, value in self.class_mapping.items()}
        class_counts = {}
        image_counts = {}
        total_objects = 0
        for record in records:
            if not record.label_path.exists():
                continue
            present_classes = set()
            for line in record.label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                class_name = id_to_name.get(class_id, str(class_id))
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                present_classes.add(class_name)
                total_objects += 1
            for class_name in present_classes:
                image_counts[class_name] = image_counts.get(class_name, 0) + 1
        return {
            "num_records": len(records),
            "total_objects": total_objects,
            "class_counts": class_counts,
            "image_counts_by_class": image_counts,
            "ignored_empty_frames": self.ignored_empty_frames,
            "saved_empty_frames": self.saved_empty_frames,
        }

    def _candidate_frame_ids(
        self,
        gt_by_frame: Dict[int, List[GroundTruthObject]],
        camera_id: str,
    ) -> List[int]:
        max_frame = max(gt_by_frame.keys()) if gt_by_frame else -1
        if self.max_frames_per_scene is not None:
            max_frame = min(max_frame, int(self.max_frames_per_scene) - 1)
        if max_frame < 0:
            return []
        rare_stride = max(1, self.frame_stride // self.rare_class_boost)
        frame_ids = []
        for frame_id in range(0, max_frame + 1):
            labels_classes = self._visible_class_names(gt_by_frame.get(frame_id, []), camera_id)
            is_rare = any(self._is_target_class(class_name) for class_name in labels_classes)
            if is_rare and frame_id % rare_stride == 0:
                frame_ids.append(frame_id)
            elif frame_id % self.frame_stride == 0:
                frame_ids.append(frame_id)
        return sorted(set(frame_ids))

    def _labels_for_frame(
        self,
        gt_objects: List[GroundTruthObject],
        camera_id: str,
        image_width: int,
        image_height: int,
    ) -> List[YoloLabel]:
        labels = []
        for obj in gt_objects:
            bbox = obj.visible_bboxes_2d.get(camera_id)
            if bbox is None:
                continue
            class_id = self._class_id_for_name(obj.object_type)
            if class_id is None:
                continue
            yolo_box = xyxy_to_yolo_norm(bbox, image_width, image_height)
            if yolo_box is None:
                continue
            labels.append(YoloLabel(class_id, yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3]))
        return labels

    def _visible_class_names(self, gt_objects: List[GroundTruthObject], camera_id: str) -> List[str]:
        names = []
        for obj in gt_objects:
            if camera_id in obj.visible_bboxes_2d:
                names.append(str(obj.object_type))
        return names

    def _is_rare_class_frame(self, class_counts: Dict[str, int]) -> bool:
        return any(self._is_target_class(class_name) for class_name in class_counts.keys())

    def _is_target_class(self, class_name: str) -> bool:
        if self.target_classes is not None:
            return str(class_name) in set(self.target_classes)
        return str(class_name) != "Person"

    def _class_limits_reached(self, class_counts: Dict[str, int], class_image_counts: Dict[str, int]) -> bool:
        if self.max_images_per_class is None or not class_counts:
            return False
        for class_name in class_counts.keys():
            if class_image_counts.get(class_name, 0) < int(self.max_images_per_class):
                return False
        return True

    def _update_class_image_counts(self, class_counts: Dict[str, int], class_image_counts: Dict[str, int]) -> None:
        for class_name in class_counts.keys():
            class_image_counts[class_name] = class_image_counts.get(class_name, 0) + 1

    def _label_class_counts(self, labels: List[YoloLabel]) -> Dict[str, int]:
        id_to_name = {int(value): key for key, value in self.class_mapping.items()}
        counts = {}
        for label in labels:
            class_name = id_to_name.get(int(label.class_id), str(label.class_id))
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def _class_id_for_name(self, class_name: str) -> Optional[int]:
        if class_name in self.class_mapping:
            return int(self.class_mapping[class_name])
        lower = {}
        for name, class_id in self.class_mapping.items():
            lower[name.lower()] = int(class_id)
        return lower.get(str(class_name).lower())

    def _camera_ids(
        self,
        gt_objects: List[GroundTruthObject],
        video_paths: Dict[str, Path],
    ) -> List[str]:
        if self.camera_id is not None:
            return [self.camera_id]
        camera_ids = set(video_paths.keys())
        for obj in gt_objects:
            camera_ids.update(obj.visible_bboxes_2d.keys())
        return sorted(camera_ids)

    def _video_paths(self, scene_paths: Any) -> Dict[str, Path]:
        if scene_paths.videos_dir is None:
            return {}
        paths = {}
        for path in list_video_files(scene_paths.videos_dir):
            paths[infer_camera_id_from_video_path(path)] = path
        return paths

    def _calibrations(self, scene_paths: Any) -> Dict[str, Any]:
        if scene_paths.calibration_path is None or not scene_paths.calibration_path.exists():
            return {}
        return load_calibration_json(scene_paths.calibration_path)

    def _frame_size(self, video_path: Path, calibration: Any) -> Tuple[int, int]:
        if calibration is not None and calibration.frame_width is not None and calibration.frame_height is not None:
            return (int(calibration.frame_width), int(calibration.frame_height))
        resolution = get_video_resolution(video_path)
        if resolution is not None:
            return resolution
        return (1920, 1080)

    def _image_path(self, split: str, scene_name: str, camera_id: str, frame_id: int) -> Path:
        name = "%s_%s_%06d%s" % (scene_name, camera_id, frame_id, self.image_ext)
        return self.output_dir / "images" / split / name

    def _label_path(self, split: str, scene_name: str, camera_id: str, frame_id: int) -> Path:
        name = "%s_%s_%06d.txt" % (scene_name, camera_id, frame_id)
        return self.output_dir / "labels" / split / name


def _group_gt_by_frame(objects: List[GroundTruthObject]) -> Dict[int, List[GroundTruthObject]]:
    grouped = {}
    for obj in objects:
        frame_id = int(obj.frame_id)
        if frame_id not in grouped:
            grouped[frame_id] = []
        grouped[frame_id].append(obj)
    return grouped

