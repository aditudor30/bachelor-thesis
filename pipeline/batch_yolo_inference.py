"""Batch YOLO inference orchestration over SmartSpaces scenes and cameras."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths, scene_name_to_id
from deep_oc_sort_3d.data.frame_io import get_video_frame_count, infer_camera_id_from_video_path, list_video_files
from deep_oc_sort_3d.detection2d.yolo_detection_io import (
    read_detections_csv,
    write_detections_csv,
    write_mot_like_detections,
)
from deep_oc_sort_3d.detection2d.yolo_inference import load_yolo_model, run_yolo_on_video
from deep_oc_sort_3d.pipeline.pipeline_paths import (
    check_overwrite,
    ensure_pipeline_dirs,
    get_detection_csv_path,
    get_mot_like_path,
    make_run_root,
)
from deep_oc_sort_3d.pipeline.run_config import PipelineRunConfig


class BatchYoloInferenceRunner:
    """Run YOLO inference over configured subsets, scenes, and cameras."""

    def __init__(self, config: PipelineRunConfig, overwrite: bool = False):
        self.config = config
        self.overwrite = overwrite
        self.run_root = make_run_root(config.output_root, config.run_name)
        ensure_pipeline_dirs(self.run_root)
        self.model = None

    def run(self) -> List[Dict[str, Any]]:
        """Run all configured subsets."""
        rows = []
        self._ensure_model()
        for subset_name in self.config.scenes_by_subset.keys():
            rows.extend(self.run_subset(subset_name))
        return rows

    def run_subset(self, subset_name: str) -> List[Dict[str, Any]]:
        """Run one configured subset."""
        split = self.config.split_by_subset.get(subset_name)
        if split is None:
            return [_error_row(subset_name, "", "", "", "missing split mapping")]
        rows = []
        for scene_name in self.config.scenes_by_subset.get(subset_name, []):
            rows.extend(self.run_scene(subset_name, split, scene_name))
        return rows

    def run_scene(self, subset_name: str, split: str, scene_name: str) -> List[Dict[str, Any]]:
        """Run all selected cameras for one scene."""
        videos = self._video_paths_for_scene(split, scene_name)
        rows = []
        if not videos:
            return [_error_row(subset_name, split, scene_name, "", "no videos found")]
        for camera_id, video_path in videos.items():
            rows.append(self.run_camera(subset_name, split, scene_name, camera_id, video_path))
        return rows

    def run_camera(
        self,
        subset_name: str,
        split: str,
        scene_name: str,
        camera_id: str,
        video_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run YOLO for one camera video and write per-camera outputs."""
        if video_path is None:
            video_path = self._video_paths_for_scene(split, scene_name).get(camera_id)
        detections_csv = get_detection_csv_path(self.run_root, subset_name, scene_name, camera_id)
        mot_like_path = get_mot_like_path(self.run_root, subset_name, scene_name, camera_id)
        if video_path is None:
            return _row(subset_name, split, scene_name, camera_id, 0, 0, detections_csv, mot_like_path, "error", "missing video")
        if detections_csv.exists() and not self.overwrite:
            detections = read_detections_csv(detections_csv)
            return _row(
                subset_name,
                split,
                scene_name,
                camera_id,
                self._num_frames_processed(video_path),
                len(detections),
                detections_csv,
                mot_like_path if mot_like_path.exists() else None,
                "skipped_existing",
                "",
            )
        if not check_overwrite(detections_csv, self.overwrite):
            return _row(subset_name, split, scene_name, camera_id, 0, 0, detections_csv, mot_like_path, "skipped_existing", "")
        try:
            scene_id = scene_name_to_id(scene_name)
            if scene_id is None:
                scene_id = -1
            detections = run_yolo_on_video(
                model=self._ensure_model(),
                video_path=video_path,
                scene_id=scene_id,
                scene_name=scene_name,
                split=split,
                camera_id=camera_id,
                conf_threshold=self.config.conf_threshold,
                max_frames=self.config.max_frames,
                frame_stride=self.config.frame_stride,
                imgsz=self.config.imgsz,
            )
            write_detections_csv(detections, detections_csv)
            mot_path = None
            if self.config.export_mot_like:
                if check_overwrite(mot_like_path, self.overwrite):
                    write_mot_like_detections(detections, mot_like_path)
                    mot_path = mot_like_path
            return _row(
                subset_name,
                split,
                scene_name,
                camera_id,
                self._num_frames_processed(video_path),
                len(detections),
                detections_csv,
                mot_path,
                "ok",
                "",
            )
        except Exception as exc:
            return _row(
                subset_name,
                split,
                scene_name,
                camera_id,
                0,
                0,
                detections_csv,
                mot_like_path,
                "error",
                str(exc),
            )

    def _ensure_model(self) -> Any:
        if self.model is None:
            self.model = load_yolo_model(self.config.detector_model)
            self._try_set_device()
        return self.model

    def _try_set_device(self) -> None:
        try:
            if hasattr(self.model, "to"):
                self.model.to(self.config.device)
        except Exception as exc:
            print("warning: could not move YOLO model to device %s: %s" % (self.config.device, exc))

    def _video_paths_for_scene(self, split: str, scene_name: str) -> Dict[str, Path]:
        scene_paths = get_scene_paths(self.config.root, split, scene_name)
        videos = {}
        if scene_paths.videos_dir is None:
            return videos
        allowed = None if self.config.camera_ids is None else set(self.config.camera_ids)
        for video_path in list_video_files(scene_paths.videos_dir):
            camera_id = infer_camera_id_from_video_path(video_path)
            if allowed is not None and camera_id not in allowed:
                continue
            videos[camera_id] = video_path
        return videos

    def _num_frames_processed(self, video_path: Path) -> int:
        frame_count = get_video_frame_count(video_path)
        if self.config.max_frames is not None:
            if frame_count is None:
                frame_count = int(self.config.max_frames)
            else:
                frame_count = min(int(frame_count), int(self.config.max_frames))
        if frame_count is None:
            return 0
        stride = max(int(self.config.frame_stride), 1)
        return int((int(frame_count) + stride - 1) / stride)


def _row(
    subset: str,
    split: str,
    scene_name: str,
    camera_id: str,
    num_frames_processed: int,
    num_detections: int,
    detections_csv: Path,
    mot_like_path: Optional[Path],
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "subset": subset,
        "split": split,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "num_frames_processed": int(num_frames_processed),
        "num_detections": int(num_detections),
        "detections_csv": str(detections_csv),
        "mot_like_path": "" if mot_like_path is None else str(mot_like_path),
        "status": status,
        "error_message": error_message,
    }


def _error_row(subset: str, split: str, scene_name: str, camera_id: str, message: str) -> Dict[str, Any]:
    return _row(subset, split, scene_name, camera_id, 0, 0, Path(""), None, "error", message)
