"""Isolated pseudo-3D estimator for Step 15C."""

from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.pseudo3d.bbox_depth_estimator import (
    bbox_projection_point,
    estimate_depth_from_bbox_height,
)
from deep_oc_sort_3d.pseudo3d.camera_model_adapter import backproject_pixel_with_depth, camera_model_from_calibration
from deep_oc_sort_3d.pseudo3d.ground_approx_estimator import estimate_ground_contact_point
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import Pseudo3DPriorTable
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DInput, Pseudo3DOutput
from deep_oc_sort_3d.pseudo3d.smoothing import smooth_track_outputs
from deep_oc_sort_3d.pseudo3d.yaw_estimator import estimate_default_yaw, estimate_yaw_from_motion


class Pseudo3DEstimator:
    """Pseudo-3D estimator using bbox height, calibration, and class priors."""

    def __init__(self, priors: Pseudo3DPriorTable, config: Dict[str, Any]) -> None:
        self.priors = priors
        self.config = config
        self.version = str(config.get("pseudo3d", {}).get("version", "0.1"))

    def estimate(self, item: Pseudo3DInput) -> Pseudo3DOutput:
        """Estimate pseudo-3D fields for one bbox without GT/depth input."""
        prior = self.priors.get(item.class_id)
        if prior is None:
            return self._failure(item, "missing_class_prior")

        camera_model, camera_error = camera_model_from_calibration(item.calibration)
        if camera_model is None:
            return self._failure(item, camera_error or "missing_intrinsics", prior=prior)

        method = str(self.config.get("method", {}).get("primary", "bbox_height_depth"))
        center = None
        coordinate_frame = "unknown"
        depth = None
        failure_reason = None
        center_source = "unknown"
        depth_source = "unknown"

        if method == "bottom_center_ground_approx":
            center, ground_error = estimate_ground_contact_point(
                item.bbox_xyxy,
                camera_model,
                self.config.get("ground_approx", {}),
            )
            if center is not None:
                coordinate_frame = "world"
                center_source = "pseudo3d_ground_plane"
                depth_source = "ground_plane_intersection"
            else:
                failure_reason = ground_error

        if center is None:
            depth, depth_error = estimate_depth_from_bbox_height(
                item.bbox_xyxy,
                prior.height,
                camera_model,
                self.config.get("bbox_height_depth", {}),
            )
            if depth is None:
                return self._failure(item, depth_error or "depth_estimation_failed", prior=prior)
            u, v = bbox_projection_point(
                item.bbox_xyxy,
                str(self.config.get("bbox_height_depth", {}).get("projection_point", "bottom_center")),
            )
            center, coordinate_frame, projection_error = backproject_pixel_with_depth(
                u,
                v,
                depth,
                camera_model,
                require_world_coordinates=bool(self.config.get("pseudo3d", {}).get("require_world_coordinates", False)),
            )
            if center is None:
                return self._failure(item, projection_error or "backprojection_failed", prior=prior, depth=depth)
            center_source = "pseudo3d_bbox_height"
            depth_source = "bbox_height_prior"
            failure_reason = projection_error

        yaw = estimate_default_yaw(self.config.get("yaw", {}))
        return Pseudo3DOutput(
            center_3d=center,
            dimensions_3d=np.asarray([prior.width, prior.length, prior.height], dtype=float),
            yaw=yaw,
            depth=depth,
            confidence_3d=self._confidence(prior, failure_reason),
            center_3d_source=center_source,
            dimensions_3d_source="class_prior",
            yaw_source="class_default",
            depth_source=depth_source,
            is_gt_derived=False,
            is_estimated_for_test=bool(self.config.get("metadata", {}).get("mark_estimated_for_test", True)),
            pseudo3d_method=method if center_source != "pseudo3d_bbox_height" else "bbox_height_depth",
            pseudo3d_version=self.version,
            subset=item.subset,
            split=item.split,
            scene_name=item.scene_name,
            camera_id=item.camera_id,
            frame_id=item.frame_id,
            class_id=item.class_id,
            class_name=item.class_name,
            local_track_id=item.local_track_id if item.local_track_id is not None else item.track_id,
            global_track_id=item.global_track_id,
            candidate_id=item.candidate_id,
            bbox_xyxy=item.bbox_xyxy,
            confidence_2d=item.confidence,
            coordinate_frame=coordinate_frame,
            projection_valid=None,
            projection_error_reason=failure_reason,
            failure_reason=failure_reason,
            source_notes="dimensions from final class prior; estimator did not use GT/depth",
        )

    def estimate_batch(self, inputs: List[Pseudo3DInput]) -> List[Pseudo3DOutput]:
        """Estimate pseudo-3D outputs for independent records."""
        return [self.estimate(item) for item in inputs]

    def estimate_track(self, inputs_for_same_track: List[Pseudo3DInput]) -> List[Pseudo3DOutput]:
        """Estimate and optionally smooth/yaw-refine one track."""
        inputs = sorted(inputs_for_same_track, key=lambda item: int(item.frame_id))
        outputs = self.estimate_batch(inputs)
        yaw = estimate_yaw_from_motion(outputs, self.config.get("yaw", {}))
        if yaw is not None:
            for output in outputs:
                if output.center_3d is not None:
                    output.yaw = yaw
                    output.yaw_source = "motion_direction"
        return smooth_track_outputs(outputs, self.config.get("smoothing", {}))

    def _failure(
        self,
        item: Pseudo3DInput,
        reason: str,
        prior: Optional[Any] = None,
        depth: Optional[float] = None,
    ) -> Pseudo3DOutput:
        dimensions = None
        if prior is not None:
            dimensions = np.asarray([prior.width, prior.length, prior.height], dtype=float)
        return Pseudo3DOutput(
            center_3d=None,
            dimensions_3d=dimensions,
            yaw=estimate_default_yaw(self.config.get("yaw", {})),
            depth=depth,
            confidence_3d=0.0,
            center_3d_source="unknown",
            dimensions_3d_source="class_prior" if prior is not None else "unknown",
            yaw_source="class_default",
            depth_source="unknown",
            is_gt_derived=False,
            is_estimated_for_test=bool(self.config.get("metadata", {}).get("mark_estimated_for_test", True)),
            pseudo3d_method=str(self.config.get("method", {}).get("primary", "bbox_height_depth")),
            pseudo3d_version=self.version,
            subset=item.subset,
            split=item.split,
            scene_name=item.scene_name,
            camera_id=item.camera_id,
            frame_id=item.frame_id,
            class_id=item.class_id,
            class_name=item.class_name,
            local_track_id=item.local_track_id if item.local_track_id is not None else item.track_id,
            global_track_id=item.global_track_id,
            candidate_id=item.candidate_id,
            bbox_xyxy=item.bbox_xyxy,
            confidence_2d=item.confidence,
            coordinate_frame="unknown",
            projection_valid=False,
            projection_error_reason=reason,
            failure_reason=reason,
            source_notes="pseudo-3D failure; no GT/depth used",
        )

    def _confidence(self, prior: Any, failure_reason: Optional[str]) -> float:
        if failure_reason in (None, ""):
            base = 0.75
        else:
            base = 0.5
        if prior.confidence_level == "high":
            return base
        if prior.confidence_level == "medium":
            return min(base, 0.5)
        return min(base, 0.25)

