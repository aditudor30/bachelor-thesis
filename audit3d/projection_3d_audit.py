"""Projection quality audit for 3D cuboid fields."""

from pathlib import Path
from typing import Any, Dict, List, Union

from deep_oc_sort_3d.audit3d.audit3d_io import (
    finite_dimensions,
    finite_float,
    finite_xyz,
    iter_data_files,
    progress_iter,
    read_csv_dicts,
)
from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.geometry.camera_geometry import world_to_camera
from deep_oc_sort_3d.visualization3d.cuboid_projection import (
    is_projected_cuboid_visible,
    project_cuboid_to_image,
)


def audit_projection_for_records(
    root: Union[str, Path],
    records_csv: Union[str, Path],
    split: str,
    scene_name: str,
    camera_id: str,
    max_records: int = 500,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit whether records can project 3D cuboids through calibration."""
    records = _read_records_for_camera(records_csv, scene_name, camera_id)
    if max_records > 0:
        records = records[:max_records]
    calibration = _load_camera_calibration(root, split, scene_name, camera_id)
    failure_counts = {}
    examples = []
    success = 0
    checked = 0
    for row in progress_iter(records, show_progress, "audit cuboid projection", "record"):
        checked += 1
        reason = _projection_failure_reason(row, calibration)
        if reason == "":
            success += 1
        else:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
            if len(examples) < 20:
                examples.append(_failure_example(row, reason))
    failed = checked - success
    return {
        "split": split,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "records_csv": str(records_csv),
        "total_records_checked": checked,
        "projection_success": success,
        "projection_failed": failed,
        "success_rate": float(success) / float(checked) if checked else None,
        "failure_reasons": failure_counts,
        "examples": examples,
    }


def audit_projection_batch(
    dataset_root: Union[str, Path],
    generic_export_root: Union[str, Path],
    subsets: Dict[str, Any],
    projection_config: Dict[str, Any],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run projection audit over configured subsets and scenes."""
    max_records = int(projection_config.get("max_records_per_camera", 200))
    summaries = []
    files = iter_data_files(generic_export_root, [".csv"])
    for subset_name, subset_config in subsets.items():
        split = str(subset_config.get("split", subset_name))
        scenes = subset_config.get("scenes", [])
        for scene_name in scenes:
            scene_files = _find_scene_csv_files(files, subset_name, str(scene_name))
            for file_path in progress_iter(scene_files, show_progress, "projection files %s" % scene_name, "file"):
                cameras = _camera_ids_in_file(file_path)
                for camera_id in cameras:
                    summaries.append(
                        audit_projection_for_records(
                            dataset_root,
                            file_path,
                            split=split,
                            scene_name=str(scene_name),
                            camera_id=str(camera_id),
                            max_records=max_records,
                            show_progress=show_progress,
                        )
                    )
    return _projection_batch_summary(summaries)


def projection_failures_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten projection failure examples for CSV output."""
    rows = []
    for item in summary.get("items", []):
        for reason, count in item.get("failure_reasons", {}).items():
            rows.append(
                {
                    "split": item.get("split"),
                    "scene_name": item.get("scene_name"),
                    "camera_id": item.get("camera_id"),
                    "reason": reason,
                    "count": count,
                    "records_csv": item.get("records_csv"),
                }
            )
        for example in item.get("examples", []):
            row = dict(example)
            row["split"] = item.get("split")
            row["scene_name"] = item.get("scene_name")
            row["camera_id"] = item.get("camera_id")
            rows.append(row)
    return rows


def _read_records_for_camera(records_csv: Union[str, Path], scene_name: str, camera_id: str) -> List[Dict[str, Any]]:
    rows = []
    for row in read_csv_dicts(records_csv):
        if str(row.get("camera_id", "")) != str(camera_id):
            continue
        if row.get("scene_name") not in (None, "") and str(row.get("scene_name")) != str(scene_name):
            continue
        rows.append(_normalize_record(row))
    return rows


def _normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(row)
    for key in ["frame_id", "global_track_id", "class_id"]:
        if key in output and output[key] not in (None, ""):
            try:
                output[key] = int(float(output[key]))
            except (TypeError, ValueError):
                pass
    for key in ["center_x", "center_y", "center_z", "width_3d", "length_3d", "height_3d", "yaw", "x", "y", "z", "width", "length", "height"]:
        if key in output:
            output[key] = finite_float(output.get(key))
    return output


def _load_camera_calibration(root: Union[str, Path], split: str, scene_name: str, camera_id: str) -> Any:
    scene_paths = get_scene_paths(Path(root), split, scene_name)
    if scene_paths.calibration_path is None or not scene_paths.calibration_path.exists():
        return None
    return load_calibration_json(scene_paths.calibration_path).get(camera_id)


def _projection_failure_reason(row: Dict[str, Any], calibration: Any) -> str:
    center = finite_xyz(row, "center_x", "center_y", "center_z")
    if center is None:
        center = finite_xyz(row, "x", "y", "z")
    dims = finite_dimensions(row, "width_3d", "length_3d", "height_3d")
    if dims is None:
        dims = finite_dimensions(row, "width", "length", "height")
    yaw = finite_float(row.get("yaw"))
    if center is None or yaw is None:
        return "missing_3d_fields"
    if dims is None:
        return "invalid_dimensions"
    if calibration is None:
        return "missing_calibration"
    if _is_center_behind_camera(center, calibration):
        return "behind_camera"
    result = project_cuboid_to_image(center, dims, yaw, calibration)
    if not result.get("success"):
        error = str(result.get("error_message", "projection_error"))
        if error == "invalid_cuboid_inputs":
            return "invalid_dimensions"
        if error == "projection_failed":
            return "projection_error"
        return "projection_error"
    width = getattr(calibration, "frame_width", None)
    height = getattr(calibration, "frame_height", None)
    if width is not None and height is not None:
        if not is_projected_cuboid_visible(result.get("points_2d"), int(width), int(height)):
            return "out_of_image"
    return ""


def _is_center_behind_camera(center: Any, calibration: Any) -> bool:
    extrinsic = getattr(calibration, "extrinsic_matrix", None)
    if extrinsic is None:
        return False
    try:
        camera_point = world_to_camera(center, extrinsic)
    except Exception:
        return False
    if camera_point.shape[0] < 3:
        return False
    return float(camera_point[2]) <= 1e-12


def _failure_example(row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "reason": reason,
        "frame_id": row.get("frame_id"),
        "global_track_id": row.get("global_track_id"),
        "class_id": row.get("class_id"),
        "center_x": row.get("center_x", row.get("x")),
        "center_y": row.get("center_y", row.get("y")),
        "center_z": row.get("center_z", row.get("z")),
        "width": row.get("width_3d", row.get("width")),
        "length": row.get("length_3d", row.get("length")),
        "height": row.get("height_3d", row.get("height")),
        "yaw": row.get("yaw"),
    }


def _find_scene_csv_files(files: List[Path], subset_name: str, scene_name: str) -> List[Path]:
    output = []
    subset_lower = str(subset_name).lower()
    scene_lower = str(scene_name).lower()
    for path in files:
        text = str(path).lower()
        if subset_lower in text and scene_lower in text:
            output.append(path)
    return output


def _camera_ids_in_file(path: Path) -> List[str]:
    rows = read_csv_dicts(path)
    cameras = sorted(set([str(row.get("camera_id", "")) for row in rows if row.get("camera_id") not in (None, "")]))
    return cameras


def _projection_batch_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    checked = sum(int(item.get("total_records_checked", 0)) for item in items)
    success = sum(int(item.get("projection_success", 0)) for item in items)
    failed = sum(int(item.get("projection_failed", 0)) for item in items)
    reasons = {}
    for item in items:
        for reason, count in item.get("failure_reasons", {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return {
        "item_count": len(items),
        "total_records_checked": checked,
        "projection_success": success,
        "projection_failed": failed,
        "success_rate": float(success) / float(checked) if checked else None,
        "failure_reasons": reasons,
        "items": items,
    }
