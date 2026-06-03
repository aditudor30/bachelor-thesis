"""Candidate selection for MVP paper/demo figures."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.visualization3d.cuboid_projection import is_projected_cuboid_visible, project_cuboid_to_image
from deep_oc_sort_3d.visualization3d.visualization_io import parse_center_dimensions_yaw_from_record


@dataclass
class FigureCandidate:
    """One recommended frame candidate for a visualization figure."""

    subset: str
    split: str
    scene_name: str
    camera_id: str
    frame_id: int
    records_path: str
    figure_type: str
    num_records: int
    num_assigned: int
    num_classes: int
    class_counts: Dict[str, int]
    num_projectable_3d: int
    projection_success_rate: Optional[float]
    score: float
    notes: str


def scan_frame_records_for_candidates(
    records_root: Union[str, Path],
    root: Optional[Union[str, Path]] = None,
    subsets: Optional[List[str]] = None,
    scenes: Optional[List[str]] = None,
    camera_ids: Optional[List[str]] = None,
    frame_stride: int = 50,
    max_frames_per_camera: Optional[int] = 100,
    figure_type: str = "tracking_2d",
    show_progress: bool = True,
) -> List[FigureCandidate]:
    """Scan frame-level global record CSV files and return figure candidates."""
    record_files = _find_record_files(Path(records_root), subsets, scenes, camera_ids)
    candidates = []
    calibration_cache = {}
    for subset, scene_name, camera_id, path in _progress_iter(record_files, show_progress, "scan frame records", "file"):
        split = subset_to_split(subset)
        frame_groups = _load_sampled_frame_groups(path, frame_stride, max_frames_per_camera)
        calibration = None
        if figure_type == "cuboid_3d" and root is not None:
            calibration = _load_camera_calibration(root, split, scene_name, camera_id, calibration_cache)
        for frame_id, records in sorted(frame_groups.items()):
            candidate = _build_candidate(
                subset=subset,
                split=split,
                scene_name=scene_name,
                camera_id=camera_id,
                frame_id=frame_id,
                records_path=path,
                records=records,
                figure_type=figure_type,
                calibration=calibration,
            )
            candidates.append(candidate)
    return candidates


def select_top_candidates(
    candidates: List[FigureCandidate],
    top_k: int = 20,
    min_records: int = 3,
    max_records: int = 30,
    min_projectable_3d: int = 0,
) -> List[FigureCandidate]:
    """Filter candidates by basic constraints and return the top scoring rows."""
    filtered = []
    for candidate in candidates:
        if candidate.num_records < int(min_records):
            continue
        if candidate.num_records > int(max_records):
            continue
        if candidate.num_projectable_3d < int(min_projectable_3d):
            continue
        filtered.append(candidate)
    return sorted(filtered, key=lambda item: item.score, reverse=True)[: int(top_k)]


def subset_to_split(subset: str) -> str:
    """Map final export subset names to dataset split names."""
    if subset == "official_val":
        return "val"
    if subset == "test":
        return "test"
    return "train"


def _build_candidate(
    subset: str,
    split: str,
    scene_name: str,
    camera_id: str,
    frame_id: int,
    records_path: Path,
    records: List[Dict[str, Any]],
    figure_type: str,
    calibration: Any,
) -> FigureCandidate:
    class_counts = _class_counts(records)
    assigned = [record for record in records if _has_global_id(record)]
    parsed_3d = [record for record in records if parse_center_dimensions_yaw_from_record(record) is not None]
    num_projectable, success_rate, notes = _projection_stats(parsed_3d, calibration)
    candidate = FigureCandidate(
        subset=subset,
        split=split,
        scene_name=scene_name,
        camera_id=camera_id,
        frame_id=int(frame_id),
        records_path=str(records_path),
        figure_type=figure_type,
        num_records=len(records),
        num_assigned=len(assigned),
        num_classes=len(class_counts),
        class_counts=class_counts,
        num_projectable_3d=num_projectable if figure_type == "cuboid_3d" else 0,
        projection_success_rate=success_rate if figure_type == "cuboid_3d" else None,
        score=0.0,
        notes=notes,
    )
    candidate.score = _score_candidate(candidate)
    return candidate


def _score_candidate(candidate: FigureCandidate) -> float:
    if candidate.figure_type == "cuboid_3d":
        from deep_oc_sort_3d.visualization3d.figure_quality_scoring import score_cuboid_3d_candidate

        return score_cuboid_3d_candidate(candidate)
    from deep_oc_sort_3d.visualization3d.figure_quality_scoring import score_tracking_2d_candidate

    return score_tracking_2d_candidate(candidate)


def _projection_stats(records: List[Dict[str, Any]], calibration: Any) -> Tuple[int, Optional[float], str]:
    if not records:
        return 0, None, "no_3d_fields"
    if calibration is None:
        return len(records), None, "projection_not_checked_no_calibration"
    success = 0
    for record in records:
        parsed = parse_center_dimensions_yaw_from_record(record)
        if parsed is None:
            continue
        center, dimensions, yaw = parsed
        result = project_cuboid_to_image(center, dimensions, yaw, calibration)
        if not result.get("success"):
            continue
        points = result.get("points_2d")
        width = getattr(calibration, "frame_width", None)
        height = getattr(calibration, "frame_height", None)
        if width is None or height is None or is_projected_cuboid_visible(points, int(width), int(height)):
            success += 1
    return success, float(success) / float(len(records)), "projection_checked"


def _load_sampled_frame_groups(
    path: Path,
    frame_stride: int,
    max_frames_per_camera: Optional[int],
) -> Dict[int, List[Dict[str, Any]]]:
    groups = {}
    max_frames = None if max_frames_per_camera is None else int(max_frames_per_camera)
    stride = max(1, int(frame_stride))
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = _optional_int(row.get("frame_id"))
            if frame_id is None:
                continue
            if frame_id % stride != 0:
                continue
            if frame_id not in groups:
                if max_frames is not None and len(groups) >= max_frames:
                    continue
                groups[frame_id] = []
            groups[frame_id].append(dict(row))
    return groups


def _find_record_files(
    records_root: Path,
    subsets: Optional[List[str]],
    scenes: Optional[List[str]],
    camera_ids: Optional[List[str]],
) -> List[Tuple[str, str, str, Path]]:
    subset_filter = None if subsets is None else set([str(item) for item in subsets])
    scene_filter = None if scenes is None else set([str(item) for item in scenes])
    camera_filter = None if camera_ids is None else set([str(item) for item in camera_ids])
    output = []
    if not records_root.exists():
        return output
    for subset_dir in sorted(records_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        subset = subset_dir.name
        if subset_filter is not None and subset not in subset_filter:
            continue
        for scene_dir in sorted(subset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name
            if scene_filter is not None and scene_name not in scene_filter:
                continue
            for path in sorted(scene_dir.glob("*_global_records.csv")):
                camera_id = path.name.replace("_global_records.csv", "")
                if camera_filter is not None and camera_id not in camera_filter:
                    continue
                output.append((subset, scene_name, camera_id, path))
    return output


def _load_camera_calibration(
    root: Union[str, Path],
    split: str,
    scene_name: str,
    camera_id: str,
    cache: Dict[str, Any],
) -> Any:
    key = "%s/%s/%s" % (split, scene_name, camera_id)
    if key in cache:
        return cache[key]
    scene_paths = get_scene_paths(Path(root), split, scene_name)
    calibration = None
    if scene_paths.calibration_path is not None and scene_paths.calibration_path.exists():
        calibration = load_calibration_json(scene_paths.calibration_path).get(camera_id)
    cache[key] = calibration
    return calibration


def _class_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for record in records:
        key = str(record.get("class_name", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _has_global_id(record: Dict[str, Any]) -> bool:
    return record.get("global_track_id") not in (None, "")


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value

