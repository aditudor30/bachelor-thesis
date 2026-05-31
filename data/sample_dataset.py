"""Frame-level SmartSpaces dataset loader.

This module returns real RGB/depth/annotation samples while keeping all heavy
inputs lazy. It does not define or run a model.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deep_oc_sort_3d.data.calibration import CameraCalibration, load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import ScenePaths, get_scene_paths
from deep_oc_sort_3d.data.depth_io import (
    infer_camera_id_from_depth_path,
    inspect_h5_depth_file,
    list_depth_files,
    safe_read_depth_frame_h5,
)
from deep_oc_sort_3d.data.frame_io import (
    get_video_frame_count,
    infer_camera_id_from_video_path,
    list_video_files,
    safe_read_video_frame,
)
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json


class SmartSpacesFrameDataset:
    """Minimal frame-level loader for one scene and one camera."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        scene_name: str,
        max_frames: Optional[int] = None,
        camera_id: Optional[str] = None,
        load_rgb: bool = True,
        load_depth: bool = True,
        load_gt: bool = True,
        depth_dataset_name: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.scene_name = scene_name
        self.max_frames = max_frames
        self.requested_camera_id = camera_id
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self.load_gt = load_gt
        self.depth_dataset_name = depth_dataset_name

        self.scene_paths = get_scene_paths(self.root, split, scene_name)
        self.video_paths_by_camera = self._build_video_index(self.scene_paths)
        self.depth_paths_by_camera = self._build_depth_index(self.scene_paths)
        self.calibrations = self._load_calibrations()
        self.gt_by_frame = self._load_ground_truth_by_frame()
        self.camera_id = self._select_camera_id(camera_id)
        self.video_frame_count = self._get_selected_video_frame_count()
        self.depth_frame_count = self._get_selected_depth_frame_count()
        self.gt_frame_count = self._get_gt_frame_count()
        self.frame_count = self._resolve_frame_count()

    def __len__(self) -> int:
        """Return the number of indexable frames for the selected camera."""
        return self.frame_count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample dictionary for a 0-based frame index."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index %d out of range for dataset length %d" % (idx, len(self)))

        frame_id = idx
        warnings = []
        video_path = self.video_paths_by_camera.get(self.camera_id)
        depth_path = self.depth_paths_by_camera.get(self.camera_id)

        rgb = None
        if self.load_rgb:
            if video_path is None:
                warnings.append("RGB video path is missing for camera %s" % self.camera_id)
            else:
                rgb = safe_read_video_frame(video_path, frame_id)
                if rgb is None:
                    warnings.append("RGB frame %d could not be read for camera %s" % (frame_id, self.camera_id))

        depth = None
        if self.split != "test" and self.load_depth:
            if depth_path is None:
                warnings.append("Depth path is missing for camera %s" % self.camera_id)
            else:
                depth = safe_read_depth_frame_h5(depth_path, frame_id, self.depth_dataset_name)
                if depth is None:
                    warnings.append("Depth frame %d could not be read for camera %s" % (frame_id, self.camera_id))

        gt_objects = self.get_gt_for_frame(frame_id)
        calibration = self.get_camera_calibration(self.camera_id)
        scene_id = self.scene_paths.scene_id if self.scene_paths.scene_id is not None else -1

        return {
            "split": self.split,
            "scene_name": self.scene_name,
            "scene_id": scene_id,
            "camera_id": self.camera_id,
            "frame_id": frame_id,
            "rgb": rgb,
            "depth": depth,
            "gt_objects": gt_objects,
            "calibration": calibration,
            "map_path": self.scene_paths.map_path
            if self.scene_paths.map_path is not None and self.scene_paths.map_path.exists()
            else None,
            "rgb_path": video_path,
            "depth_path": depth_path if self.split != "test" else None,
            "warnings": warnings,
        }

    def list_cameras(self) -> List[str]:
        """List camera ids known from videos, depth files, or calibration."""
        camera_ids = set()
        camera_ids.update(self.video_paths_by_camera.keys())
        camera_ids.update(self.depth_paths_by_camera.keys())
        camera_ids.update(self.calibrations.keys())
        return sorted(camera_ids)

    def get_available_frame_ids(self) -> List[int]:
        """Return available 0-based frame ids for this dataset view."""
        return list(range(len(self)))

    def get_gt_for_frame(self, frame_id: int) -> Optional[List[GroundTruthObject]]:
        """Return ground-truth objects for a frame, or None when unavailable."""
        if self.split == "test" or not self.load_gt or self.gt_by_frame is None:
            return None
        return list(self.gt_by_frame.get(frame_id, []))

    def get_camera_calibration(self, camera_id: str) -> Optional[CameraCalibration]:
        """Return calibration for a camera id when available."""
        return self.calibrations.get(camera_id)

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight loader summary."""
        return {
            "root": str(self.root),
            "split": self.split,
            "scene_name": self.scene_name,
            "scene_id": self.scene_paths.scene_id,
            "selected_camera_id": self.camera_id,
            "requested_camera_id": self.requested_camera_id,
            "max_frames": self.max_frames,
            "num_cameras": len(self.list_cameras()),
            "cameras": self.list_cameras(),
            "num_video_files": len(self.video_paths_by_camera),
            "num_depth_files": len(self.depth_paths_by_camera),
            "video_frame_count_selected": self.video_frame_count,
            "depth_frame_count_selected": self.depth_frame_count,
            "gt_frame_count": self.gt_frame_count,
            "resolved_frame_count": self.frame_count,
            "has_ground_truth": self.gt_by_frame is not None,
            "has_calibration_for_selected_camera": self.get_camera_calibration(self.camera_id) is not None,
            "map_path": str(self.scene_paths.map_path)
            if self.scene_paths.map_path is not None and self.scene_paths.map_path.exists()
            else None,
            "test_missing_depth_and_gt_is_expected": self.split == "test",
        }

    def _build_video_index(self, scene_paths: ScenePaths) -> Dict[str, Path]:
        if scene_paths.videos_dir is None:
            return {}
        paths = list_video_files(scene_paths.videos_dir)
        index = {}
        for path in paths:
            index[infer_camera_id_from_video_path(path)] = path
        return index

    def _build_depth_index(self, scene_paths: ScenePaths) -> Dict[str, Path]:
        if self.split == "test" or scene_paths.depth_maps_dir is None:
            return {}
        paths = list_depth_files(scene_paths.depth_maps_dir)
        index = {}
        for path in paths:
            index[infer_camera_id_from_depth_path(path)] = path
        return index

    def _load_calibrations(self) -> Dict[str, CameraCalibration]:
        path = self.scene_paths.calibration_path
        if path is None or not path.exists():
            return {}
        return load_calibration_json(path)

    def _load_ground_truth_by_frame(self) -> Optional[Dict[int, List[GroundTruthObject]]]:
        if self.split == "test" or not self.load_gt:
            return None
        path = self.scene_paths.ground_truth_path
        if path is None or not path.exists():
            return None
        objects = load_ground_truth_json(path)
        by_frame = {}
        for obj in objects:
            if obj.frame_id not in by_frame:
                by_frame[obj.frame_id] = []
            by_frame[obj.frame_id].append(obj)
        return by_frame

    def _select_camera_id(self, camera_id: Optional[str]) -> str:
        if camera_id is not None:
            return camera_id
        cameras = self.list_cameras()
        if cameras:
            return cameras[0]
        return ""

    def _get_selected_video_frame_count(self) -> Optional[int]:
        path = self.video_paths_by_camera.get(self.camera_id)
        if path is None:
            return None
        return get_video_frame_count(path)

    def _get_selected_depth_frame_count(self) -> Optional[int]:
        if self.split == "test":
            return None
        path = self.depth_paths_by_camera.get(self.camera_id)
        if path is None:
            return None
        report = inspect_h5_depth_file(path)
        value = report.get("num_frames")
        if value is None:
            return None
        return int(value)

    def _get_gt_frame_count(self) -> Optional[int]:
        if self.gt_by_frame is None or not self.gt_by_frame:
            return None
        return max(self.gt_by_frame.keys()) + 1

    def _resolve_frame_count(self) -> int:
        candidates = []
        if self.video_frame_count is not None:
            candidates.append(self.video_frame_count)
        if self.depth_frame_count is not None:
            candidates.append(self.depth_frame_count)
        if self.gt_frame_count is not None:
            candidates.append(self.gt_frame_count)

        if candidates:
            inferred = max(candidates)
        elif self.max_frames is not None:
            inferred = self.max_frames
        else:
            inferred = 0

        if self.max_frames is not None:
            return min(inferred, self.max_frames)
        return inferred
