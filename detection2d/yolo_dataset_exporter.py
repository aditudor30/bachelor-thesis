"""Export SmartSpaces GT boxes as a YOLO detection dataset."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import yaml

from deep_oc_sort_3d.data.dataset_structure import list_scenes
from deep_oc_sort_3d.data.frame_io import get_video_resolution, safe_read_video_frame
from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.detection2d.yolo_label_utils import write_yolo_label_file, xyxy_to_yolo_norm
from deep_oc_sort_3d.detection2d.yolo_types import YoloExportRecord, YoloLabel


DEFAULT_CLASS_MAPPING = {
    "Person": 0,
    "Forklift": 1,
    "PalletTruck": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "NovaCarter": 6,
}


class YoloDatasetExporter:
    """Export selected frames and GT labels to YOLO format."""

    def __init__(
        self,
        root: Union[str, Path],
        output_dir: Union[str, Path],
        class_mapping: Optional[Dict[str, int]] = None,
        image_ext: str = ".jpg",
        frame_stride: int = 1,
        max_frames_per_scene: Optional[int] = None,
        max_scenes: Optional[int] = None,
        camera_id: Optional[str] = None,
        include_empty_frames: bool = False,
        jpeg_quality: int = 95,
    ):
        self.root = Path(root)
        self.output_dir = Path(output_dir)
        self.class_mapping = dict(class_mapping) if class_mapping is not None else dict(DEFAULT_CLASS_MAPPING)
        self.image_ext = image_ext
        self.frame_stride = max(int(frame_stride), 1)
        self.max_frames_per_scene = max_frames_per_scene
        self.max_scenes = max_scenes
        self.camera_id = camera_id
        self.include_empty_frames = include_empty_frames
        self.jpeg_quality = int(jpeg_quality)
        self.ignored_empty_frames = 0
        self.saved_empty_frames = 0

    def export_split(self, split: str, scenes: Optional[List[str]] = None) -> List[YoloExportRecord]:
        """Export one split, usually train or val."""
        if split not in ("train", "val"):
            raise ValueError("YOLO dataset export only supports train/val splits.")
        scene_names = scenes if scenes is not None else list_scenes(self.root, split)
        if self.max_scenes is not None:
            scene_names = scene_names[: self.max_scenes]
        records = []
        for scene_name in scene_names:
            records.extend(self.export_scene(split, scene_name))
        return records

    def export_scene(self, split: str, scene_name: str) -> List[YoloExportRecord]:
        """Export one scene to YOLO image/label files."""
        records = []
        camera_ids = self._camera_ids_for_scene(split, scene_name)
        for camera_id in camera_ids:
            frame_dataset = SmartSpacesFrameDataset(
                root=self.root,
                split=split,
                scene_name=scene_name,
                max_frames=self.max_frames_per_scene,
                camera_id=camera_id,
                load_rgb=False,
                load_depth=False,
                load_gt=True,
            )
            video_path = frame_dataset.video_paths_by_camera.get(camera_id)
            if video_path is None:
                print("warning: missing video for %s %s %s" % (split, scene_name, camera_id))
                continue
            frame_limit = len(frame_dataset)
            if self.max_frames_per_scene is not None:
                frame_limit = min(frame_limit, int(self.max_frames_per_scene))
            for frame_id in range(0, frame_limit, self.frame_stride):
                sample = frame_dataset[frame_id]
                image_width, image_height = self._frame_size(sample, video_path)
                labels = self._labels_for_sample(sample, image_width, image_height)
                if not labels and not self.include_empty_frames:
                    self.ignored_empty_frames += 1
                    continue

                frame_rgb = safe_read_video_frame(video_path, frame_id)
                if frame_rgb is None:
                    print("warning: could not read frame %d from %s" % (frame_id, video_path))
                    continue

                image_path = self._image_path(split, scene_name, camera_id, frame_id)
                label_path = self._label_path(split, scene_name, camera_id, frame_id)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                write_yolo_label_file(labels, label_path)
                if not labels:
                    self.saved_empty_frames += 1
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
        return records

    def write_data_yaml(self, yaml_path: Path) -> None:
        """Write YOLO data.yaml."""
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

    def class_distribution(self, records: List[YoloExportRecord]) -> Dict[str, Any]:
        """Count exported objects per class id by reading generated labels."""
        counts = {}
        total_objects = 0
        for record in records:
            if not record.label_path.exists():
                continue
            for line in record.label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                counts[class_id] = counts.get(class_id, 0) + 1
                total_objects += 1
        return {
            "num_records": len(records),
            "total_objects": total_objects,
            "class_counts": counts,
            "ignored_empty_frames": self.ignored_empty_frames,
            "saved_empty_frames": self.saved_empty_frames,
        }

    def _camera_ids_for_scene(self, split: str, scene_name: str) -> List[str]:
        frame_dataset = SmartSpacesFrameDataset(
            root=self.root,
            split=split,
            scene_name=scene_name,
            max_frames=1,
            camera_id=self.camera_id,
            load_rgb=False,
            load_depth=False,
            load_gt=True,
        )
        if self.camera_id is not None:
            return [self.camera_id]
        return sorted(frame_dataset.video_paths_by_camera.keys())

    def _labels_for_sample(self, sample: Dict[str, Any], image_width: int, image_height: int) -> List[YoloLabel]:
        labels = []
        gt_objects = sample.get("gt_objects")
        if gt_objects is None:
            return labels
        camera_id = str(sample.get("camera_id"))
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

    def _class_id_for_name(self, class_name: str) -> Optional[int]:
        if class_name in self.class_mapping:
            return int(self.class_mapping[class_name])
        lower = {}
        for name, class_id in self.class_mapping.items():
            lower[name.lower()] = int(class_id)
        return lower.get(str(class_name).lower())

    def _frame_size(self, sample: Dict[str, Any], video_path: Path) -> Any:
        calibration = sample.get("calibration")
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

