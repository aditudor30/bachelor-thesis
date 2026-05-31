"""All-class YOLO export wrappers for train/val and internal holdout."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deep_oc_sort_3d.detection2d.yolo_balanced_exporter import BalancedYoloDatasetExporter
from deep_oc_sort_3d.detection2d.yolo_dataset_exporter import DEFAULT_CLASS_MAPPING
from deep_oc_sort_3d.detection2d.yolo_types import YoloExportRecord


ALL_RARE_TARGET_CLASSES = [
    "Forklift",
    "PalletTruck",
    "Transporter",
    "FourierGR1T2",
    "AgilityDigit",
    "NovaCarter",
]


MANIFEST_FIELDS = [
    "split",
    "scene_name",
    "camera_id",
    "frame_id",
    "image_path",
    "label_path",
    "class_counts_json",
    "contains_rare_class",
]


class AllClassYoloExporter:
    """Export all-class balanced YOLO datasets with explicit scene splits."""

    def __init__(
        self,
        root: Union[str, Path],
        output_dir: Union[str, Path],
        train_scenes: List[str],
        val_scenes: List[str],
        class_mapping: Optional[Dict[str, int]] = None,
        camera_id: Optional[str] = None,
        frame_stride: int = 5,
        max_frames_per_scene: Optional[int] = 1000,
        target_classes: Optional[List[str]] = None,
        rare_class_boost: int = 4,
        include_empty_frames: bool = False,
        jpeg_quality: int = 95,
    ):
        self.root = Path(root)
        self.output_dir = Path(output_dir)
        self.train_scenes = list(train_scenes)
        self.val_scenes = list(val_scenes)
        self.class_mapping = dict(class_mapping) if class_mapping is not None else dict(DEFAULT_CLASS_MAPPING)
        self.camera_id = camera_id
        self.frame_stride = int(frame_stride)
        self.max_frames_per_scene = max_frames_per_scene
        self.target_classes = list(target_classes) if target_classes is not None else list(ALL_RARE_TARGET_CLASSES)
        self.rare_class_boost = int(rare_class_boost)
        self.include_empty_frames = include_empty_frames
        self.jpeg_quality = int(jpeg_quality)
        self._exporter = self._make_exporter(self.output_dir)
        self.records = []

    def export_train_val(self) -> List[YoloExportRecord]:
        """Export configured train and val scene lists."""
        train_records = self._exporter.export_split("train", self.train_scenes)
        val_records = self._exporter.export_split("val", self.val_scenes)
        self.records = train_records + val_records
        self.write_data_yaml()
        self.write_manifest()
        return list(self.records)

    def export_internal_holdout(
        self,
        holdout_scenes: List[str],
        holdout_output_dir: Union[str, Path],
    ) -> List[YoloExportRecord]:
        """Export a diagnostic holdout dataset from train scenes."""
        holdout_exporter = self._make_exporter(Path(holdout_output_dir))
        records = holdout_exporter.export_split("train", holdout_scenes)
        holdout_exporter.write_data_yaml(Path(holdout_output_dir) / "data.yaml")
        self._write_manifest_rows(holdout_exporter.manifest_rows, Path(holdout_output_dir) / "export_manifest.csv")
        return records

    def write_data_yaml(self) -> None:
        """Write data.yaml for the train/val export."""
        self._exporter.write_data_yaml(self.output_dir / "data.yaml")

    def write_manifest(self) -> None:
        """Write export_manifest.csv for the train/val export."""
        self._write_manifest_rows(self._exporter.manifest_rows, self.output_dir / "export_manifest.csv")

    def summarize_export(self) -> Dict[str, Any]:
        """Return class-distribution summary for the last export."""
        return self._exporter.class_distribution(self.records)

    def _make_exporter(self, output_dir: Path) -> BalancedYoloDatasetExporter:
        return BalancedYoloDatasetExporter(
            root=self.root,
            output_dir=output_dir,
            class_mapping=self.class_mapping,
            frame_stride=self.frame_stride,
            max_frames_per_scene=self.max_frames_per_scene,
            camera_id=self.camera_id,
            include_empty_frames=self.include_empty_frames,
            jpeg_quality=self.jpeg_quality,
            target_classes=self.target_classes,
            rare_class_boost=self.rare_class_boost,
        )

    @staticmethod
    def _write_manifest_rows(rows: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "split": row.get("split"),
                        "scene_name": row.get("scene_name"),
                        "camera_id": row.get("camera_id"),
                        "frame_id": row.get("frame_id"),
                        "image_path": row.get("image_path"),
                        "label_path": row.get("label_path"),
                        "class_counts_json": row.get("class_counts_json"),
                        "contains_rare_class": row.get("is_rare_class_frame"),
                    }
                )
