"""Path helpers for batch detection-to-observation pipeline runs."""

from pathlib import Path
from typing import Union


def make_run_root(output_root: Union[str, Path], run_name: str) -> Path:
    """Return the root directory for one pipeline run."""
    return Path(output_root) / str(run_name)


def get_detection_csv_path(run_root: Union[str, Path], subset: str, scene_name: str, camera_id: str) -> Path:
    """Return the per-camera Detection2D CSV path."""
    return Path(run_root) / "detections2d" / subset / scene_name / ("%s.csv" % camera_id)


def get_mot_like_path(run_root: Union[str, Path], subset: str, scene_name: str, camera_id: str) -> Path:
    """Return the per-camera MOT-like detection path."""
    return Path(run_root) / "mot_like" / subset / scene_name / ("%s.txt" % camera_id)


def get_observation_jsonl_path(run_root: Union[str, Path], subset: str, scene_name: str, camera_id: str) -> Path:
    """Return the per-camera Observation3D JSONL path."""
    return Path(run_root) / "observations3d" / subset / scene_name / ("%s.jsonl" % camera_id)


def get_summaries_dir(run_root: Union[str, Path]) -> Path:
    """Return the summaries directory for one run."""
    return Path(run_root) / "summaries"


def get_visualizations_dir(run_root: Union[str, Path]) -> Path:
    """Return the visualizations directory for one run."""
    return Path(run_root) / "visualizations"


def ensure_pipeline_dirs(run_root: Union[str, Path]) -> None:
    """Create the top-level pipeline output directories."""
    root = Path(run_root)
    for name in ["detections2d", "mot_like", "observations3d", "summaries", "visualizations"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def check_overwrite(path: Path, overwrite: bool) -> bool:
    """Return True when a path may be written, printing a warning when skipped."""
    if path.exists() and not overwrite:
        print("warning: output exists and overwrite is false, skipping: %s" % path)
        return False
    return True
