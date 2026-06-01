"""Batch Observation3D construction from per-camera Detection2D CSV files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.observations.observation_builder import Observation3DBuilder
from deep_oc_sort_3d.observations.observation_io import write_observations_jsonl
from deep_oc_sort_3d.pipeline.pipeline_paths import (
    check_overwrite,
    ensure_pipeline_dirs,
    get_detection_csv_path,
    get_observation_jsonl_path,
    make_run_root,
)
from deep_oc_sort_3d.pipeline.run_config import PipelineRunConfig


class BatchObservationBuilder:
    """Build Observation3D JSONL files over configured subsets, scenes, and cameras."""

    def __init__(self, config: PipelineRunConfig, overwrite: bool = False):
        self.config = config
        self.overwrite = overwrite
        self.run_root = make_run_root(config.output_root, config.run_name)
        ensure_pipeline_dirs(self.run_root)

    def run(self) -> List[Dict[str, Any]]:
        """Run all configured subsets."""
        rows = []
        for subset_name in self.config.scenes_by_subset.keys():
            rows.extend(self.run_subset(subset_name))
        return rows

    def run_subset(self, subset_name: str) -> List[Dict[str, Any]]:
        """Run one subset."""
        split = self.config.split_by_subset.get(subset_name)
        if split is None:
            return [_error_row(subset_name, "", "", "", "missing split mapping")]
        rows = []
        for scene_name in self.config.scenes_by_subset.get(subset_name, []):
            rows.extend(self.run_scene(subset_name, split, scene_name))
        return rows

    def run_scene(self, subset_name: str, split: str, scene_name: str) -> List[Dict[str, Any]]:
        """Run all selected cameras for one scene."""
        camera_ids = self._camera_ids_for_scene(subset_name, scene_name)
        if not camera_ids:
            return [_error_row(subset_name, split, scene_name, "", "no detection CSV files found")]
        rows = []
        for camera_id in camera_ids:
            rows.append(self.run_camera(subset_name, split, scene_name, camera_id))
        return rows

    def run_camera(self, subset_name: str, split: str, scene_name: str, camera_id: str) -> Dict[str, Any]:
        """Build observations for one per-camera Detection2D CSV."""
        detections_csv = get_detection_csv_path(self.run_root, subset_name, scene_name, camera_id)
        observations_jsonl = get_observation_jsonl_path(self.run_root, subset_name, scene_name, camera_id)
        if not detections_csv.exists():
            return _row(
                subset_name,
                split,
                scene_name,
                camera_id,
                detections_csv,
                observations_jsonl,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                {},
                "error",
                "missing detections CSV",
            )
        if observations_jsonl.exists() and not self.overwrite:
            return self._summary_from_existing(subset_name, split, scene_name, camera_id, detections_csv, observations_jsonl)
        if not check_overwrite(observations_jsonl, self.overwrite):
            return self._summary_from_existing(subset_name, split, scene_name, camera_id, detections_csv, observations_jsonl)
        try:
            builder = Observation3DBuilder(
                root=self.config.root,
                split=split,
                scene_name=scene_name,
                yolo_detections_csv=detections_csv,
                camera_id=camera_id,
                depth_sampling_method=self.config.depth_sampling_method,
                iou_threshold=self.config.iou_threshold,
                class_must_match=self.config.class_must_match,
                use_depth_if_available=split in ("train", "val"),
            )
            observations = builder.build(max_frames=self.config.max_frames)
            write_observations_jsonl(observations, observations_jsonl)
            summary = builder.summary(observations)
            return _row_from_summary(
                subset_name,
                split,
                scene_name,
                camera_id,
                detections_csv,
                observations_jsonl,
                summary,
                "ok",
                "",
            )
        except Exception as exc:
            return _row(
                subset_name,
                split,
                scene_name,
                camera_id,
                detections_csv,
                observations_jsonl,
                len(read_detections_csv(detections_csv)),
                0,
                0,
                0,
                None,
                0,
                0,
                {},
                "error",
                str(exc),
            )

    def _camera_ids_for_scene(self, subset_name: str, scene_name: str) -> List[str]:
        base = self.run_root / "detections2d" / subset_name / scene_name
        if self.config.camera_ids is not None:
            return list(self.config.camera_ids)
        if not base.exists():
            return []
        return sorted(path.stem for path in base.glob("*.csv"))

    def _summary_from_existing(
        self,
        subset_name: str,
        split: str,
        scene_name: str,
        camera_id: str,
        detections_csv: Path,
        observations_jsonl: Path,
    ) -> Dict[str, Any]:
        from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl

        observations = read_observations_jsonl(observations_jsonl)
        matched = [obs for obs in observations if obs.matched_gt]
        ious = [obs.matched_iou for obs in matched if obs.matched_iou is not None]
        per_class = {}
        for obs in observations:
            per_class[obs.class_name] = per_class.get(obs.class_name, 0) + 1
        mean_iou = None if not ious else float(sum(ious)) / float(len(ious))
        return _row(
            subset_name,
            split,
            scene_name,
            camera_id,
            detections_csv,
            observations_jsonl,
            len(read_detections_csv(detections_csv)),
            len(observations),
            len(matched),
            len(observations) - len(matched),
            mean_iou,
            len([obs for obs in observations if obs.depth_value is not None]),
            len([obs for obs in observations if obs.center_3d is not None]),
            per_class,
            "skipped_existing",
            "",
        )


def _row_from_summary(
    subset: str,
    split: str,
    scene_name: str,
    camera_id: str,
    detections_csv: Path,
    observations_jsonl: Path,
    summary: Dict[str, Any],
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return _row(
        subset,
        split,
        scene_name,
        camera_id,
        detections_csv,
        observations_jsonl,
        int(summary.get("num_detections", 0)),
        int(summary.get("num_observations", 0)),
        int(summary.get("matched_gt", 0)),
        int(summary.get("unmatched", 0)),
        summary.get("mean_iou"),
        int(summary.get("depth_valid", 0)),
        int(summary.get("center_3d_available", 0)),
        summary.get("per_class_counts", {}),
        status,
        error_message,
    )


def _row(
    subset: str,
    split: str,
    scene_name: str,
    camera_id: str,
    detections_csv: Path,
    observations_jsonl: Path,
    num_detections: int,
    num_observations: int,
    matched_gt: int,
    unmatched: int,
    mean_iou: Optional[float],
    depth_valid: int,
    center_3d_available: int,
    per_class_counts: Dict[str, int],
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "subset": subset,
        "split": split,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "detections_csv": str(detections_csv),
        "observations_jsonl": str(observations_jsonl),
        "num_detections": int(num_detections),
        "num_observations": int(num_observations),
        "matched_gt": int(matched_gt),
        "unmatched": int(unmatched),
        "mean_iou": mean_iou,
        "depth_valid": int(depth_valid),
        "center_3d_available": int(center_3d_available),
        "per_class_counts_json": json.dumps(per_class_counts, sort_keys=True),
        "status": status,
        "error_message": error_message,
    }


def _error_row(subset: str, split: str, scene_name: str, camera_id: str, message: str) -> Dict[str, Any]:
    return _row(subset, split, scene_name, camera_id, Path(""), Path(""), 0, 0, 0, 0, None, 0, 0, {}, "error", message)
