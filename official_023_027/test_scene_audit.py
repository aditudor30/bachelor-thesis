"""Dataset-structure audit for official test scenes 023 through 027."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.official_023_027.official_config import dataset_root, output_root, scene_names
from deep_oc_sort_3d.official_023_027.official_track1_io import write_json


def audit_test_scenes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Audit required test folders without reading videos."""
    root = dataset_root(config)
    rows = []
    for scene_name in scene_names(config, "all"):
        scene_root = root / "test" / scene_name
        videos = scene_root / "videos"
        calibration = scene_root / "calibration.json"
        map_path = scene_root / "map.png"
        video_files = sorted(
            [path for path in videos.iterdir() if path.is_file() and path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
        ) if videos.is_dir() else []
        missing = []
        if not scene_root.exists():
            missing.append("scene_root")
        if not videos.is_dir():
            missing.append("videos")
        elif not video_files:
            missing.append("video_files")
        if not calibration.is_file():
            missing.append("calibration.json")
        if not map_path.is_file():
            missing.append("map.png")
        rows.append(
            {
                "scene_name": scene_name,
                "scene_root": str(scene_root),
                "exists": scene_root.exists(),
                "is_symlink": scene_root.is_symlink(),
                "videos_dir": str(videos),
                "video_files": len(video_files),
                "calibration_exists": calibration.is_file(),
                "map_exists": map_path.is_file(),
                "depth_expected": False,
                "ground_truth_expected": False,
                "missing_required": missing,
                "status": "ok" if not missing else "error",
            }
        )
    summary = {
        "dataset_root": str(root),
        "required_scenes": scene_names(config, "all"),
        "scene_count": len(rows),
        "ok_scenes": sum(1 for row in rows if row.get("status") == "ok"),
        "missing_scenes": [row.get("scene_name") for row in rows if row.get("status") != "ok"],
        "status": "ok" if rows and all(row.get("status") == "ok" for row in rows) else "error",
        "scenes": rows,
    }
    write_json(output_root(config) / "audit" / "test_scene_audit.json", summary)
    return summary
