"""Dataset structure helpers for SmartSpaces MTMC scenes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


KNOWN_SPLITS = ("train", "val", "test")


@dataclass
class ScenePaths:
    """Resolved paths for one dataset scene."""

    split: str
    scene_name: str
    scene_id: Optional[int]
    root: Path
    videos_dir: Optional[Path]
    depth_maps_dir: Optional[Path]
    ground_truth_path: Optional[Path]
    calibration_path: Optional[Path]
    map_path: Optional[Path]


def expected_scene_names(split: str) -> List[str]:
    """Return the expected scene names for a known split."""
    if split == "train":
        return ["Warehouse_%03d" % index for index in range(0, 20)]
    if split == "val":
        return ["Warehouse_%03d" % index for index in range(20, 23)]
    if split == "test":
        return ["Warehouse_%03d" % index for index in range(23, 26)]
    return []


def scene_name_to_id(scene_name: str) -> Optional[int]:
    """Parse a Warehouse_XXX scene id, returning None when it is not parseable."""
    prefix = "Warehouse_"
    if not scene_name.startswith(prefix):
        return None
    suffix = scene_name[len(prefix) :]
    try:
        return int(suffix)
    except (TypeError, ValueError):
        return None


def list_splits(root: Path) -> List[str]:
    """List split directories found under root without failing on a missing root."""
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []
    splits = []
    for split in KNOWN_SPLITS:
        split_path = root_path / split
        if split_path.exists() and split_path.is_dir():
            splits.append(split)
    return splits


def list_scenes(root: Path, split: str) -> List[str]:
    """List scene directory names for a split without assuming the split exists."""
    split_path = Path(root) / split
    if not split_path.exists() or not split_path.is_dir():
        return []
    scene_names = []
    for child in split_path.iterdir():
        if child.is_dir():
            scene_names.append(child.name)
    return sorted(scene_names)


def get_scene_paths(root: Path, split: str, scene_name: str) -> ScenePaths:
    """Build the conventional paths for one scene.

    Paths are returned even when files do not exist so callers can report exactly
    what is present and what is missing.
    """
    root_path = Path(root)
    scene_root = resolve_scene_root(root_path, split, scene_name)
    return ScenePaths(
        split=split,
        scene_name=scene_name,
        scene_id=scene_name_to_id(scene_name),
        root=scene_root,
        videos_dir=scene_root / "videos",
        depth_maps_dir=scene_root / "depth_maps",
        ground_truth_path=scene_root / "ground_truth.json",
        calibration_path=scene_root / "calibration.json",
        map_path=scene_root / "map.png",
    )


def resolve_scene_root(root: Path, split: str, scene_name: str) -> Path:
    """Resolve a scene directory from dataset, split, or scene roots.

    The canonical input is the dataset root, so the first candidate is
    ``root/split/scene_name``. For debug use, this also accepts ``root`` as a
    split directory or as the scene directory itself.
    """
    root_path = Path(root)
    conventional = root_path / split / scene_name

    candidates = [conventional]
    split_path = _child_path(root_path, split)
    if split_path is not None:
        scene_under_split = _child_path(split_path, scene_name)
        if scene_under_split is not None:
            candidates.append(scene_under_split)

    scene_under_root = _child_path(root_path, scene_name)
    if scene_under_root is not None:
        candidates.append(scene_under_root)

    if _looks_like_scene_dir(root_path):
        candidates.append(root_path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return conventional


def _child_path(parent: Path, child_name: str) -> Optional[Path]:
    if not parent.exists() or not parent.is_dir() or not child_name:
        return None
    exact = parent / child_name
    if exact.exists():
        return exact
    child_name_lower = child_name.lower()
    for child in parent.iterdir():
        if child.name.lower() == child_name_lower:
            return child
    return None


def _looks_like_scene_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    scene_markers = ["videos", "calibration.json", "map.png", "depth_maps", "ground_truth.json"]
    for marker in scene_markers:
        if (path / marker).exists():
            return True
    return False


def _path_exists(path: Optional[Path], expect_dir: bool) -> bool:
    if path is None:
        return False
    if expect_dir:
        return path.exists() and path.is_dir()
    return path.exists() and path.is_file()


def validate_scene_structure(scene_paths: ScenePaths, split: str) -> Dict[str, Any]:
    """Validate required scene files for a split and report optional absences.

    The function never raises for missing dataset files. It returns a structured
    report suitable for inspection CLIs and smoke tests.
    """
    checks = {
        "root": _path_exists(scene_paths.root, True),
        "videos_dir": _path_exists(scene_paths.videos_dir, True),
        "depth_maps_dir": _path_exists(scene_paths.depth_maps_dir, True),
        "ground_truth_path": _path_exists(scene_paths.ground_truth_path, False),
        "calibration_path": _path_exists(scene_paths.calibration_path, False),
        "map_path": _path_exists(scene_paths.map_path, False),
    }

    if split == "test":
        required = ["root", "videos_dir", "calibration_path", "map_path"]
        optional = ["depth_maps_dir", "ground_truth_path"]
    elif split in ("train", "val"):
        required = [
            "root",
            "videos_dir",
            "depth_maps_dir",
            "ground_truth_path",
            "calibration_path",
            "map_path",
        ]
        optional = []
    else:
        required = ["root"]
        optional = ["videos_dir", "depth_maps_dir", "ground_truth_path", "calibration_path", "map_path"]

    expected_scenes = expected_scene_names(split)
    if expected_scenes:
        scene_expected_for_split = scene_paths.scene_name in expected_scenes
    else:
        scene_expected_for_split = None

    missing_required = []
    for name in required:
        if not checks.get(name, False):
            missing_required.append(name)

    missing_optional = []
    for name in optional:
        if not checks.get(name, False):
            missing_optional.append(name)

    notes = []
    if scene_expected_for_split is False:
        notes.append(
            "Scene %s is not expected for split %s. Expected scenes: %s"
            % (scene_paths.scene_name, split, ", ".join(expected_scenes))
        )

    return {
        "split": split,
        "scene_name": scene_paths.scene_name,
        "scene_id": scene_paths.scene_id,
        "root": str(scene_paths.root),
        "scene_expected_for_split": scene_expected_for_split,
        "expected_scenes": expected_scenes,
        "exists": checks,
        "required": required,
        "optional": optional,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "notes": notes,
        "is_valid": len(missing_required) == 0,
    }
