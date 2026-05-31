"""Build 3D training targets from SmartSpaces frame samples."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.geometry.camera_geometry import camera_to_world, ensure_matrix, pixel_depth_to_camera_point
from deep_oc_sort_3d.geometry.depth_quality import guess_depth_unit, summarize_depth_array
from deep_oc_sort_3d.geometry.depth_sampling import sample_depth_robust
from deep_oc_sort_3d.geometry.projection_3d import bbox_bottom_center, bbox_center
from deep_oc_sort_3d.training.target_types import FrameTrainingTargets, TrainingObjectTarget


DEFAULT_CLASS_MAPPING = {
    "Person": 0,
    "Forklift": 1,
    "PalletTruck": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "NovaCarter": 6,
}


class TrainingTargetBuilder:
    """Build future model targets from parsed ground truth and depth."""

    def __init__(
        self,
        class_mapping: Optional[Dict[str, int]] = None,
        use_depth_backprojection: bool = True,
        use_bottom_center: bool = True,
        depth_sampling_method: str = "center_median",
        auto_convert_depth_units: bool = True,
    ):
        if class_mapping is None:
            self.class_mapping = dict(DEFAULT_CLASS_MAPPING)
        else:
            self.class_mapping = dict(class_mapping)
        self.use_depth_backprojection = use_depth_backprojection
        self.use_bottom_center = use_bottom_center
        self.depth_sampling_method = depth_sampling_method
        self.auto_convert_depth_units = auto_convert_depth_units

    def build_targets_from_sample(self, sample: Dict[str, Any]) -> FrameTrainingTargets:
        """Build target dataclasses from one SmartSpacesFrameDataset sample."""
        scene_id = int(sample.get("scene_id", -1))
        scene_name = str(sample.get("scene_name", ""))
        frame_id = int(sample.get("frame_id", -1))
        camera_id = str(sample.get("camera_id", ""))
        gt_objects = sample.get("gt_objects")

        if gt_objects is None:
            return FrameTrainingTargets(
                scene_id=scene_id,
                scene_name=scene_name,
                frame_id=frame_id,
                camera_id=camera_id,
                targets=[],
            )

        depth = sample.get("depth")
        calibration = sample.get("calibration")
        depth_unit = self._guess_depth_unit(depth)
        targets = []
        for obj in gt_objects:
            target = self._build_target_for_object(
                obj=obj,
                sample=sample,
                scene_id=scene_id,
                scene_name=scene_name,
                frame_id=frame_id,
                camera_id=camera_id,
                depth=depth,
                depth_unit=depth_unit,
                calibration=calibration,
            )
            targets.append(target)

        return FrameTrainingTargets(
            scene_id=scene_id,
            scene_name=scene_name,
            frame_id=frame_id,
            camera_id=camera_id,
            targets=targets,
        )

    def _build_target_for_object(
        self,
        obj: GroundTruthObject,
        sample: Dict[str, Any],
        scene_id: int,
        scene_name: str,
        frame_id: int,
        camera_id: str,
        depth: Any,
        depth_unit: Optional[str],
        calibration: Any,
    ) -> TrainingObjectTarget:
        class_name = str(obj.object_type)
        class_id = self._class_id_for_name(class_name)
        center_3d = np.asarray(obj.location_3d, dtype=float).reshape(3)
        dimensions_3d = np.asarray(obj.bbox3d_scale, dtype=float).reshape(3)
        rotation_3d = np.asarray(obj.bbox3d_rotation, dtype=float).reshape(3)
        yaw = float(rotation_3d[-1])
        bbox_xyxy = self._bbox_for_camera(obj, camera_id)

        depth_value = None
        backprojected = None
        error = None
        if bbox_xyxy is not None and depth is not None:
            depth_value_raw = sample_depth_robust(depth, bbox_xyxy, method=self.depth_sampling_method)
            depth_value = self._convert_scalar_depth(depth_value_raw, depth_unit)
            if self.use_depth_backprojection and calibration is not None:
                backprojected = self._backproject_bbox_with_sampled_depth(
                    bbox_xyxy=bbox_xyxy,
                    depth_value=depth_value,
                    calibration=calibration,
                )
                if backprojected is not None:
                    error = float(np.linalg.norm(backprojected - center_3d))

        return TrainingObjectTarget(
            scene_id=scene_id,
            scene_name=scene_name,
            frame_id=frame_id,
            camera_id=camera_id,
            class_name=class_name,
            class_id=class_id,
            object_id=int(obj.object_id),
            bbox_xyxy=bbox_xyxy,
            center_3d=center_3d,
            dimensions_3d=dimensions_3d,
            rotation_3d=rotation_3d,
            yaw=yaw,
            depth_value=depth_value,
            backprojected_center_3d=backprojected,
            backprojection_error=error,
        )

    def _class_id_for_name(self, class_name: str) -> int:
        if class_name in self.class_mapping:
            return int(self.class_mapping[class_name])
        lower_mapping = {}
        for key, value in self.class_mapping.items():
            lower_mapping[key.lower()] = value
        return int(lower_mapping.get(class_name.lower(), -1))

    @staticmethod
    def _bbox_for_camera(
        obj: GroundTruthObject,
        camera_id: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        bbox = obj.visible_bboxes_2d.get(camera_id)
        if bbox is None:
            return None
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    def _guess_depth_unit(self, depth: Any) -> Optional[str]:
        if depth is None or not self.auto_convert_depth_units:
            return None
        try:
            return guess_depth_unit(summarize_depth_array(depth))
        except Exception:
            return None

    def _convert_scalar_depth(self, depth_value: Optional[float], depth_unit: Optional[str]) -> Optional[float]:
        if depth_value is None:
            return None
        if self.auto_convert_depth_units and depth_unit == "millimeters_likely":
            return float(depth_value) / 1000.0
        return float(depth_value)

    def _backproject_bbox_with_sampled_depth(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        depth_value: Optional[float],
        calibration: Any,
    ) -> Optional[np.ndarray]:
        if depth_value is None or calibration is None:
            return None
        intrinsic = ensure_matrix(calibration.intrinsic_matrix, (3, 3))
        extrinsic = ensure_matrix(calibration.extrinsic_matrix)
        if intrinsic is None or extrinsic is None:
            return None

        if self.use_bottom_center:
            u, v = bbox_bottom_center(bbox_xyxy)
        else:
            u, v = bbox_center(bbox_xyxy)
        try:
            point_camera = pixel_depth_to_camera_point(u, v, depth_value, intrinsic)
            return camera_to_world(point_camera, extrinsic)
        except Exception:
            return None
