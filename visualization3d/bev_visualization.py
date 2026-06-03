"""Bird's-eye-view visualization for generic MVP exports."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deep_oc_sort_3d.visualization3d.visualization_io import load_global_frame_records_csv


def load_scene_map(root: Union[str, Path], split: str, scene_name: str) -> Optional[np.ndarray]:
    """Load ``map.png`` for a scene when it exists."""
    path = Path(root) / split / scene_name / "map.png"
    if not path.exists():
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def plot_bev_tracks(
    tracks_or_records: List[Dict[str, Any]],
    output_path: Union[str, Path],
    map_image: Optional[np.ndarray] = None,
    max_tracks: int = 100,
    class_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Plot x-y trajectories grouped by global_track_id.

    If a scene map is supplied, it is currently not used as a background because
    the world-to-map transform is not known. The plot is therefore explicitly a
    coordinate-space BEV, not a map-aligned visualization.
    """
    records = _filter_records(tracks_or_records, class_name)
    groups = _group_by_track(records)
    selected_ids = sorted(groups.keys())[: int(max_tracks)]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for track_id in selected_ids:
        points = _xy_points(groups[track_id])
        if points.shape[0] == 0:
            continue
        ax.plot(points[:, 0], points[:, 1], linewidth=1.2, alpha=0.75)
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.75)
    title = "Coordinate-space BEV global trajectories"
    if map_image is not None:
        title += " (map not aligned)"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(str(output), dpi=150)
    plt.close(fig)
    return {
        "output_path": str(output),
        "input_records": len(records),
        "tracks_plotted": len(selected_ids),
        "map_used": False,
    }


def plot_bev_from_generic_export(
    generic_csv_path: Union[str, Path],
    output_path: Union[str, Path],
    map_path: Optional[Path] = None,
    max_tracks: int = 100,
) -> Dict[str, Any]:
    """Load a generic export CSV and save a coordinate-space BEV PNG."""
    records = load_global_frame_records_csv(generic_csv_path)
    map_image = None
    if map_path is not None and Path(map_path).exists():
        try:
            map_image = np.asarray(Image.open(map_path).convert("RGB"))
        except Exception:
            map_image = None
    return plot_bev_tracks(records, output_path, map_image=map_image, max_tracks=max_tracks)


def _filter_records(records: List[Dict[str, Any]], class_name: Optional[str]) -> List[Dict[str, Any]]:
    if class_name is None:
        return list(records)
    return [record for record in records if str(record.get("class_name", "")) == str(class_name)]


def _group_by_track(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups = {}
    for record in records:
        track_id = _optional_int(record.get("global_track_id"))
        if track_id is None:
            continue
        groups.setdefault(track_id, []).append(record)
    return groups


def _xy_points(records: List[Dict[str, Any]]) -> np.ndarray:
    rows = []
    ordered = sorted(records, key=lambda item: _optional_int(item.get("frame_id")) or -1)
    for record in ordered:
        x = _optional_float(record.get("center_x"))
        y = _optional_float(record.get("center_y"))
        if x is None or y is None:
            continue
        rows.append([x, y])
    if not rows:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(rows, dtype=float)


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

