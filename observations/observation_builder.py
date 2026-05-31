"""Build standardized Observation3D records from YOLO detections."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D
from deep_oc_sort_3d.geometry.depth_sampling import sample_depth_robust
from deep_oc_sort_3d.geometry.projection_3d import bbox_bottom_center
from deep_oc_sort_3d.observations.detection_gt_matching import match_detections_to_gt
from deep_oc_sort_3d.observations.observation_types import Observation3D


DEFAULT_CLASS_DIMENSIONS = {
    "Person": [0.6, 0.6, 1.75],
    "Forklift": [1.2, 2.5, 2.0],
    "PalletTruck": [0.8, 1.2, 1.0],
    "Transporter": [0.8, 1.2, 1.0],
    "FourierGR1T2": [0.7, 0.7, 1.7],
    "AgilityDigit": [0.7, 0.7, 1.7],
    "NovaCarter": [0.8, 1.0, 1.2],
}


class Observation3DBuilder:
    """Construct Observation3D JSONL-ready records from YOLO detections."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        scene_name: str,
        yolo_detections_csv: Union[str, Path],
        camera_id: Optional[str] = None,
        depth_sampling_method: str = "center_median",
        iou_threshold: float = 0.3,
        class_must_match: bool = True,
        use_depth_if_available: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.scene_name = scene_name
        self.camera_id = camera_id
        self.depth_sampling_method = depth_sampling_method
        self.iou_threshold = iou_threshold
        self.class_must_match = class_must_match
        self.use_depth_if_available = use_depth_if_available
        self.detections = self._load_detections(Path(yolo_detections_csv))
        self.detections_by_key = self._group_detections(self.detections)
        self.frame_datasets = {}

    def build_for_frame(self, frame_id: int, camera_id: str) -> List[Observation3D]:
        """Build observations for one frame and camera."""
        key = (int(frame_id), camera_id)
        indexed_detections = self.detections_by_key.get(key, [])
        if not indexed_detections:
            return []
        detections = [item[1] for item in indexed_detections]
        try:
            sample = self._frame_dataset(camera_id)[frame_id]
        except Exception:
            sample = {
                "gt_objects": None,
                "depth": None,
                "calibration": None,
            }
        gt_objects = sample.get("gt_objects")
        depth = sample.get("depth")
        calibration = sample.get("calibration")

        matched_gt = {}
        matched_iou = {}
        if self.split in ("train", "val") and gt_objects is not None:
            matched_gt, matched_iou = match_detections_to_gt(
                detections=detections,
                gt_objects=gt_objects,
                camera_id=camera_id,
                iou_threshold=self.iou_threshold,
                class_must_match=self.class_must_match,
            )

        observations = []
        for local_idx, det in enumerate(detections):
            detection_id = indexed_detections[local_idx][0]
            gt = matched_gt.get(local_idx)
            iou = matched_iou.get(local_idx)
            depth_value = self._sample_depth(depth, det.bbox_xyxy)
            if gt is not None:
                center_3d = np.asarray(gt.location_3d, dtype=float)
                dimensions_3d = np.asarray(gt.bbox3d_scale, dtype=float)
                yaw = float(gt.bbox3d_rotation[-1])
                object_id = int(gt.object_id)
                matched = True
            else:
                dimensions_3d = self._default_dimensions(det.class_name)
                center_3d = self._baseline_center(det, dimensions_3d, calibration)
                yaw = 0.0
                object_id = None
                matched = False
            observations.append(
                Observation3D(
                    scene_id=det.scene_id,
                    scene_name=det.scene_name,
                    split=det.split,
                    camera_id=det.camera_id,
                    frame_id=det.frame_id,
                    detection_id=detection_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=det.bbox_xyxy,
                    bbox_xywh=det.bbox_xywh,
                    center_3d=center_3d,
                    dimensions_3d=dimensions_3d,
                    yaw=yaw,
                    object_id=object_id,
                    matched_gt=matched,
                    matched_iou=iou,
                    depth_value=depth_value,
                    depth_sampling_method=self.depth_sampling_method if depth_value is not None else None,
                    source=det.source,
                )
            )
        return observations

    def build(self, max_frames: Optional[int] = None) -> List[Observation3D]:
        """Build observations for all selected detections."""
        observations = []
        keys = sorted(self.detections_by_key.keys())
        for frame_id, camera_id in keys:
            if max_frames is not None and frame_id >= int(max_frames):
                continue
            if self.camera_id is not None and camera_id != self.camera_id:
                continue
            observations.extend(self.build_for_frame(frame_id, camera_id))
        return observations

    def summary(self, observations: List[Observation3D]) -> Dict[str, Any]:
        """Summarize built observations."""
        matched = [obs for obs in observations if obs.matched_gt]
        ious = [obs.matched_iou for obs in matched if obs.matched_iou is not None]
        depth_valid = [obs for obs in observations if obs.depth_value is not None]
        center_valid = [obs for obs in observations if obs.center_3d is not None]
        per_class = {}
        for obs in observations:
            if obs.class_name not in per_class:
                per_class[obs.class_name] = 0
            per_class[obs.class_name] += 1
        return {
            "num_detections": len(self.detections),
            "num_observations": len(observations),
            "matched_gt": len(matched),
            "unmatched": len(observations) - len(matched),
            "mean_iou": None if not ious else float(np.mean(ious)),
            "per_class_counts": per_class,
            "depth_valid": len(depth_valid),
            "center_3d_available": len(center_valid),
        }

    def _load_detections(self, path: Path) -> List[Detection2D]:
        detections = []
        for det in read_detections_csv(path):
            if det.split != self.split or det.scene_name != self.scene_name:
                continue
            if self.camera_id is not None and det.camera_id != self.camera_id:
                continue
            detections.append(det)
        return detections

    def _group_detections(self, detections: List[Detection2D]) -> Dict[Tuple[int, str], List[Tuple[int, Detection2D]]]:
        grouped = {}
        for detection_id, det in enumerate(detections):
            key = (det.frame_id, det.camera_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((detection_id, det))
        return grouped

    def _frame_dataset(self, camera_id: str) -> SmartSpacesFrameDataset:
        if camera_id not in self.frame_datasets:
            self.frame_datasets[camera_id] = SmartSpacesFrameDataset(
                root=self.root,
                split=self.split,
                scene_name=self.scene_name,
                camera_id=camera_id,
                load_rgb=False,
                load_depth=self.use_depth_if_available and self.split in ("train", "val"),
                load_gt=self.split in ("train", "val"),
            )
        return self.frame_datasets[camera_id]

    def _sample_depth(
        self,
        depth: Any,
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> Optional[float]:
        if depth is None:
            return None
        value = sample_depth_robust(depth, bbox_xyxy, method=self.depth_sampling_method)
        if value is None:
            return None
        return float(value) / 1000.0

    def _default_dimensions(self, class_name: str) -> np.ndarray:
        values = DEFAULT_CLASS_DIMENSIONS.get(class_name)
        if values is None:
            values = [0.8, 0.8, 1.5]
        return np.asarray(values, dtype=float)

    def _baseline_center(
        self,
        det: Detection2D,
        dimensions_3d: np.ndarray,
        calibration: Any,
    ) -> Optional[np.ndarray]:
        if calibration is None or calibration.homography is None:
            return None
        try:
            u, v = bbox_bottom_center(det.bbox_xyxy)
            point = np.asarray([u, v, 1.0], dtype=float)
            mapped = calibration.homography.dot(point)
            if abs(mapped[2]) < 1e-12:
                return None
            mapped = mapped / mapped[2]
            return np.asarray([mapped[0], mapped[1], float(dimensions_3d[2]) * 0.5], dtype=float)
        except Exception:
            return None
