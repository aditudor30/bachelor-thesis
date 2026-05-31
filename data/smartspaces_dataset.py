"""Minimal SmartSpaces dataset loader.

This loader intentionally stops at structure inspection and metadata parsing.
It does not decode videos, consume full depth-map directories, or start any
model/training pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from deep_oc_sort_3d.data.calibration import CameraCalibration, load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import (
    ScenePaths,
    get_scene_paths,
    list_scenes,
    validate_scene_structure,
)
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
DEPTH_EXTENSIONS = (".h5", ".hdf5", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".exr", ".tif", ".tiff")


class SmartSpacesDataset:
    """A minimal split/scene loader for SmartSpaces 2026."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        scene_name: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.scene_name = scene_name
        self.max_frames = max_frames

        if scene_name is None:
            scenes = list_scenes(self.root, split)
            self.scene_names = scenes
        else:
            self.scene_names = [scene_name]

    def get_scene_paths(self) -> ScenePaths:
        """Return paths for the selected scene."""
        if self.scene_name is None:
            if not self.scene_names:
                return get_scene_paths(self.root, self.split, "")
            return get_scene_paths(self.root, self.split, self.scene_names[0])
        return get_scene_paths(self.root, self.split, self.scene_name)

    def load_calibrations(self) -> Dict[str, CameraCalibration]:
        """Load camera calibrations for the selected scene when available."""
        scene_paths = self.get_scene_paths()
        if scene_paths.calibration_path is None or not scene_paths.calibration_path.exists():
            return {}
        return load_calibration_json(scene_paths.calibration_path)

    def load_ground_truth(self) -> Optional[List[GroundTruthObject]]:
        """Load ground truth for train/val; return None for test or missing files."""
        if self.split == "test":
            return None
        scene_paths = self.get_scene_paths()
        if scene_paths.ground_truth_path is None or not scene_paths.ground_truth_path.exists():
            return None
        objects = load_ground_truth_json(scene_paths.ground_truth_path)
        if self.max_frames is None:
            return objects
        return [obj for obj in objects if obj.frame_id < self.max_frames]

    def list_video_files(self) -> List[Path]:
        """List video files for the selected scene."""
        scene_paths = self.get_scene_paths()
        if scene_paths.videos_dir is None or not scene_paths.videos_dir.exists():
            return []
        return self._list_files(scene_paths.videos_dir, VIDEO_EXTENSIONS, None)

    def list_depth_files(self) -> Optional[List[Path]]:
        """List depth map files for train/val; return None for test or missing dirs."""
        if self.split == "test":
            return None
        scene_paths = self.get_scene_paths()
        if scene_paths.depth_maps_dir is None or not scene_paths.depth_maps_dir.exists():
            return None
        files = self._list_files(scene_paths.depth_maps_dir, DEPTH_EXTENSIONS, self.max_frames)
        if files:
            return files
        return self._list_files(scene_paths.depth_maps_dir, None, self.max_frames)

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary of the selected scene."""
        scene_paths = self.get_scene_paths()
        structure = validate_scene_structure(scene_paths, self.split)
        video_files = self.list_video_files()
        depth_files = self.list_depth_files()
        return {
            "root": str(self.root),
            "split": self.split,
            "scene_name": scene_paths.scene_name,
            "scene_id": scene_paths.scene_id,
            "max_frames": self.max_frames,
            "structure": structure,
            "num_video_files": len(video_files),
            "num_depth_files": None if depth_files is None else len(depth_files),
            "has_ground_truth": (
                scene_paths.ground_truth_path is not None
                and scene_paths.ground_truth_path.exists()
                and self.split != "test"
            ),
            "has_calibration": scene_paths.calibration_path is not None and scene_paths.calibration_path.exists(),
            "has_map": scene_paths.map_path is not None and scene_paths.map_path.exists(),
            "test_missing_depth_and_gt_is_expected": self.split == "test",
        }

    @staticmethod
    def _list_files(
        directory: Path,
        extensions: Optional[Tuple[str, ...]],
        limit: Optional[int],
    ) -> List[Path]:
        files = []
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if extensions is not None and path.suffix.lower() not in extensions:
                continue
            files.append(path)
            if limit is not None and len(files) >= limit:
                break
        return sorted(files)
