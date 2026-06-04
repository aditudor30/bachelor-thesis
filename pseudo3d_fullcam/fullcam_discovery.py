"""Discovery helpers for full-camera pseudo-3D generation."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json


@dataclass
class FullCamItem:
    """One camera file required by the Step 15G full-camera pseudo-3D run."""

    subset: str
    split: str
    scene_name: str
    camera_id: str
    input_records_path: str
    raw_prediction_path: str
    stabilized_prediction_path: str
    frame_min: Optional[int] = None
    frame_max: Optional[int] = None
    num_records: Optional[int] = None
    raw_exists: bool = False
    stabilized_exists: bool = False
    status: str = "unknown"


def discover_required_camera_files(config: Dict[str, Any]) -> List[FullCamItem]:
    """Discover required camera record files from the configured input root."""
    source = _input_source(config)
    input_root = _input_root(config, source)
    raw_root = _raw_output_root(config)
    stabilized_root = _stabilized_output_root(config)
    items = []
    for subset in _configured_subsets(config):
        split = _split_for_subset(subset)
        for scene_name in _configured_scenes(config, subset, input_root):
            for path, camera_id in _input_camera_paths(input_root, subset, scene_name, source, config):
                stats = _record_stats(path)
                raw_path = raw_root / subset / scene_name / ("%s_pseudo3d_predictions.jsonl" % camera_id)
                stabilized_path = stabilized_root / subset / scene_name / ("%s_pseudo3d_stabilized.jsonl" % camera_id)
                status = "ok" if path.exists() else "missing_input"
                items.append(
                    FullCamItem(
                        subset=subset,
                        split=split,
                        scene_name=scene_name,
                        camera_id=camera_id,
                        input_records_path=str(path),
                        raw_prediction_path=str(raw_path),
                        stabilized_prediction_path=str(stabilized_path),
                        frame_min=stats.get("frame_min"),
                        frame_max=stats.get("frame_max"),
                        num_records=stats.get("num_records"),
                        raw_exists=_raw_exists(config, raw_path, subset, scene_name, camera_id),
                        stabilized_exists=_stabilized_exists(config, stabilized_path, subset, scene_name, camera_id),
                        status=status,
                    )
                )
    return sorted(items, key=lambda item: (item.subset, item.scene_name, item.camera_id))


def audit_existing_predictions(items: List[FullCamItem]) -> Dict[str, Any]:
    """Summarize whether required cameras already have raw/stabilized outputs."""
    rows = [fullcam_item_to_dict(item) for item in items]
    raw_existing = [row for row in rows if row.get("raw_exists")]
    stabilized_existing = [row for row in rows if row.get("stabilized_exists")]
    missing_inputs = [row for row in rows if row.get("status") != "ok"]
    return {
        "required_camera_files": len(items),
        "raw_prediction_files_existing": len(raw_existing),
        "stabilized_prediction_files_existing": len(stabilized_existing),
        "raw_prediction_file_coverage": _rate(len(raw_existing), len(items)),
        "stabilized_prediction_file_coverage": _rate(len(stabilized_existing), len(items)),
        "missing_input_files": len(missing_inputs),
        "missing_raw_prediction_files": len(items) - len(raw_existing),
        "missing_stabilized_prediction_files": len(items) - len(stabilized_existing),
        "per_subset": _count(rows, "subset"),
        "per_scene": _count(rows, "scene_name"),
        "per_camera": rows,
    }


def write_fullcam_items_json(items: List[FullCamItem], path: Union[str, Path]) -> None:
    """Write discovered camera files as JSON."""
    write_json({"items": [fullcam_item_to_dict(item) for item in items]}, path)


def write_fullcam_items_csv(items: List[FullCamItem], path: Union[str, Path]) -> None:
    """Write discovered camera files as CSV."""
    write_csv([fullcam_item_to_dict(item) for item in items], path)


def read_fullcam_items_json(path: Union[str, Path]) -> List[FullCamItem]:
    """Read discovered camera items from JSON."""
    input_path = Path(path)
    if not input_path.exists():
        return []
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return [FullCamItem(**row) for row in rows if isinstance(row, dict)]


def fullcam_item_to_dict(item: FullCamItem) -> Dict[str, Any]:
    """Convert a FullCamItem to a JSON/CSV-safe dictionary."""
    return asdict(item)


def filter_fullcam_items(
    items: List[FullCamItem],
    subsets: Optional[Sequence[str]] = None,
    scenes: Optional[Sequence[str]] = None,
    camera_ids: Optional[Sequence[str]] = None,
    max_cameras: Optional[int] = None,
) -> List[FullCamItem]:
    """Filter discovered items for debug or partial reruns."""
    subset_set = _string_set(subsets)
    scene_set = _string_set(scenes)
    camera_set = _string_set(camera_ids)
    out = []
    for item in items:
        if subset_set and item.subset not in subset_set:
            continue
        if scene_set and item.scene_name not in scene_set:
            continue
        if camera_set and item.camera_id not in camera_set:
            continue
        out.append(item)
        if max_cameras is not None and len(out) >= int(max_cameras):
            break
    return out


def missing_prediction_rows(items: List[FullCamItem]) -> List[Dict[str, Any]]:
    """Return compact rows for cameras still missing raw or stabilized outputs."""
    rows = []
    for item in items:
        if item.raw_exists and item.stabilized_exists and item.status == "ok":
            continue
        rows.append(
            {
                "subset": item.subset,
                "scene_name": item.scene_name,
                "camera_id": item.camera_id,
                "status": item.status,
                "raw_exists": item.raw_exists,
                "stabilized_exists": item.stabilized_exists,
                "input_records_path": item.input_records_path,
                "raw_prediction_path": item.raw_prediction_path,
                "stabilized_prediction_path": item.stabilized_prediction_path,
            }
        )
    return rows


def _input_source(config: Dict[str, Any]) -> str:
    return str(config.get("input_selection", {}).get("source", "frame_global_records"))


def _input_root(config: Dict[str, Any], source: str) -> Path:
    paths = config.get("paths", {})
    if source == "observations3d":
        return Path(paths.get("observations_root", "output/pipeline_runs/yolo11m_medium_curriculum_conf001/observations3d"))
    return Path(paths.get("frame_records_root", "output/final_mvp_exports/yolo11m_medium_conf001_transition/frame_global_records"))


def _raw_output_root(config: Dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    default = _output_root(config) / "predictions_raw"
    return Path(paths.get("raw_output_root", default))


def _stabilized_output_root(config: Dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    default = _output_root(config) / "predictions_stabilized"
    return Path(paths.get("stabilized_output_root", default))


def _output_root(config: Dict[str, Any]) -> Path:
    section = config.get("step15g", config.get("experiment", {}))
    return Path(section.get("output_root", "output/pseudo3d/baseline_v2_pseudo3d_fullcam"))


def _configured_subsets(config: Dict[str, Any]) -> List[str]:
    selection = config.get("input_selection", {})
    values = selection.get("subsets")
    if values:
        return [str(value) for value in values]
    scenes = selection.get("scenes", {})
    if isinstance(scenes, dict) and scenes:
        return [str(key) for key in scenes.keys()]
    return ["official_val", "internal_holdout", "test"]


def _configured_scenes(config: Dict[str, Any], subset: str, input_root: Path) -> List[str]:
    scenes = config.get("input_selection", {}).get("scenes", {})
    if isinstance(scenes, dict) and scenes.get(subset):
        return [str(value) for value in scenes.get(subset, [])]
    subset_root = input_root / subset
    if not subset_root.exists():
        return []
    return sorted([path.name for path in subset_root.iterdir() if path.is_dir()])


def _input_camera_paths(
    input_root: Path,
    subset: str,
    scene_name: str,
    source: str,
    config: Dict[str, Any],
) -> List[Any]:
    camera_ids = config.get("input_selection", {}).get("camera_ids")
    scene_root = input_root / subset / scene_name
    if camera_ids:
        return [(_path_for_camera(scene_root, source, str(camera_id)), str(camera_id)) for camera_id in camera_ids]
    if not scene_root.exists():
        return []
    if source == "observations3d":
        paths = [path for path in scene_root.iterdir() if path.is_file() and path.suffix.lower() in (".jsonl", ".csv")]
        return sorted([(path, path.stem) for path in paths], key=lambda pair: str(pair[1]))
    paths = sorted(scene_root.glob("*_global_records.csv"))
    return [(path, path.name.replace("_global_records.csv", "")) for path in paths]


def _path_for_camera(scene_root: Path, source: str, camera_id: str) -> Path:
    if source == "observations3d":
        jsonl = scene_root / ("%s.jsonl" % camera_id)
        return jsonl if jsonl.exists() else scene_root / ("%s.csv" % camera_id)
    return scene_root / ("%s_global_records.csv" % camera_id)


def _record_stats(path: Path) -> Dict[str, Optional[int]]:
    if not path.exists():
        return {"frame_min": None, "frame_max": None, "num_records": 0}
    if path.suffix.lower() == ".jsonl":
        return _jsonl_record_stats(path)
    return _csv_record_stats(path)


def _csv_record_stats(path: Path) -> Dict[str, Optional[int]]:
    frames = []
    count = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count += 1
            frame_id = _optional_int(row.get("frame_id"))
            if frame_id is not None:
                frames.append(frame_id)
    return {"frame_min": min(frames) if frames else None, "frame_max": max(frames) if frames else None, "num_records": count}


def _jsonl_record_stats(path: Path) -> Dict[str, Optional[int]]:
    frames = []
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        count += 1
        try:
            row = json.loads(line)
        except ValueError:
            continue
        frame_id = _optional_int(row.get("frame_id")) if isinstance(row, dict) else None
        if frame_id is not None:
            frames.append(frame_id)
    return {"frame_min": min(frames) if frames else None, "frame_max": max(frames) if frames else None, "num_records": count}


def _raw_exists(config: Dict[str, Any], target: Path, subset: str, scene_name: str, camera_id: str) -> bool:
    return target.exists() or _legacy_prediction_path(config, "existing_raw_root", subset, scene_name, camera_id, "predictions").exists()


def _stabilized_exists(config: Dict[str, Any], target: Path, subset: str, scene_name: str, camera_id: str) -> bool:
    return target.exists() or _legacy_prediction_path(config, "existing_stabilized_root", subset, scene_name, camera_id, "stabilized").exists()


def _legacy_prediction_path(config: Dict[str, Any], key: str, subset: str, scene_name: str, camera_id: str, kind: str) -> Path:
    root = config.get("paths", {}).get(key)
    if not root:
        return Path("__missing__")
    suffix = "pseudo3d_predictions" if kind == "predictions" else "pseudo3d_stabilized"
    return Path(root) / subset / scene_name / ("%s_%s.jsonl" % (camera_id, suffix))


def _split_for_subset(subset: str) -> str:
    if subset == "official_val":
        return "val"
    if subset == "internal_holdout":
        return "train"
    return "test"


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _count(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _string_set(values: Optional[Iterable[str]]) -> Optional[set]:
    if not values:
        return None
    return set(str(value) for value in values)
