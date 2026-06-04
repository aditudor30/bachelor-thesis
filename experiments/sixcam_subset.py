"""Definition and discovery helpers for the pseudo-3D 6-camera subset."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SixCamItem:
    """One camera included in the pseudo-3D 6-camera experiment."""

    subset: str
    split: str
    scene_name: str
    camera_id: str
    pseudo3d_predictions_path: str
    frame_min: Optional[int] = None
    frame_max: Optional[int] = None
    unique_frames: Optional[int] = None
    num_predictions: Optional[int] = None
    num_complete: Optional[int] = None
    completion_rate: Optional[float] = None


DEFAULT_SIXCAM = [
    ("official_val", "val", "Warehouse_020", "Camera_0000"),
    ("official_val", "val", "Warehouse_021", "Camera_0000"),
    ("official_val", "val", "Warehouse_022", "Camera_0000"),
    ("internal_holdout", "train", "Warehouse_014", "Camera_0000"),
    ("internal_holdout", "train", "Warehouse_015", "Camera_0000"),
    ("internal_holdout", "train", "Warehouse_016", "Camera_0000"),
]


def get_default_sixcam_subset() -> List[SixCamItem]:
    """Return the expected fixed 6-camera subset."""
    return [SixCamItem(subset, split, scene, camera, "") for subset, split, scene, camera in DEFAULT_SIXCAM]


def discover_sixcam_subset(predictions_root: Union[str, Path]) -> List[SixCamItem]:
    """Discover pseudo-3D prediction stats for the fixed 6-camera subset."""
    root = Path(predictions_root)
    items = []
    for item in get_default_sixcam_subset():
        path = root / item.subset / item.scene_name / ("%s_pseudo3d_stabilized.jsonl" % item.camera_id)
        stats = _prediction_stats(path)
        items.append(
            SixCamItem(
                subset=item.subset,
                split=item.split,
                scene_name=item.scene_name,
                camera_id=item.camera_id,
                pseudo3d_predictions_path=str(path),
                frame_min=stats.get("frame_min"),
                frame_max=stats.get("frame_max"),
                unique_frames=stats.get("unique_frames"),
                num_predictions=stats.get("num_predictions"),
                num_complete=stats.get("num_complete"),
                completion_rate=stats.get("completion_rate"),
            )
        )
    return items


def write_sixcam_subset_json(items: List[SixCamItem], path: Union[str, Path]) -> None:
    """Write subset definition as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(item) for item in items], indent=2, sort_keys=True), encoding="utf-8")


def write_sixcam_subset_csv(items: List[SixCamItem], path: Union[str, Path]) -> None:
    """Write subset definition as CSV."""
    rows = [asdict(item) for item in items]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else list(asdict(SixCamItem("", "", "", "", "")).keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_sixcam_subset_json(path: Union[str, Path]) -> List[SixCamItem]:
    """Read subset definition JSON."""
    input_path = Path(path)
    if not input_path.exists():
        return []
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return [SixCamItem(**item) for item in data if isinstance(item, dict)]


def sixcam_items_from_config(config: Dict[str, Any]) -> List[SixCamItem]:
    """Return subset items from config, falling back to defaults."""
    rows = config.get("sixcam_subset")
    if not rows:
        return get_default_sixcam_subset()
    items = []
    for row in rows:
        items.append(
            SixCamItem(
                subset=str(row.get("subset", "")),
                split=str(row.get("split", "")),
                scene_name=str(row.get("scene_name", "")),
                camera_id=str(row.get("camera_id", "")),
                pseudo3d_predictions_path=str(row.get("pseudo3d_predictions_path", "")),
            )
        )
    return items


def write_frame_coverage_csv(items: List[SixCamItem], path: Union[str, Path]) -> None:
    """Write compact frame coverage CSV for the subset."""
    write_sixcam_subset_csv(items, path)


def _prediction_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "frame_min": None,
            "frame_max": None,
            "unique_frames": 0,
            "num_predictions": 0,
            "num_complete": 0,
            "completion_rate": None,
        }
    frames = []
    total = 0
    complete = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        frame_id = _optional_int(row.get("frame_id"))
        if frame_id is not None:
            frames.append(frame_id)
        if _is_complete_prediction(row):
            complete += 1
    return {
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "unique_frames": len(set(frames)),
        "num_predictions": total,
        "num_complete": complete,
        "completion_rate": float(complete) / float(total) if total else None,
    }


def _is_complete_prediction(row: Dict[str, Any]) -> bool:
    return (
        row.get("center_3d") not in (None, "", [])
        and row.get("dimensions_3d") not in (None, "", [])
        and row.get("yaw") is not None
        and row.get("depth") is not None
    )


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

