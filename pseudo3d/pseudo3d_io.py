"""I/O helpers for isolated pseudo-3D predictions."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deep_oc_sort_3d.audit3d.audit3d_io import optional_float, optional_int, progress_iter
from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import (
    Pseudo3DInput,
    Pseudo3DOutput,
    pseudo3d_output_to_dict,
)


def read_frame_record_inputs(
    records_path: Union[str, Path],
    subset: str,
    split: str,
    scene_name: str,
    camera_id: str,
    calibration: Any,
    show_progress: bool = True,
) -> List[Pseudo3DInput]:
    """Read frame-level record CSV and convert rows to pseudo-3D inputs."""
    rows = _read_csv(records_path)
    inputs = []
    width = _calib_int(calibration, "frame_width", "frameWidth") or 0
    height = _calib_int(calibration, "frame_height", "frameHeight") or 0
    for row in progress_iter(rows, show_progress, "read pseudo3D frame records", "record"):
        bbox = (
            float(row.get("x1", 0.0)),
            float(row.get("y1", 0.0)),
            float(row.get("x2", 0.0)),
            float(row.get("y2", 0.0)),
        )
        inputs.append(
            Pseudo3DInput(
                scene_name=str(row.get("scene_name") or scene_name),
                camera_id=str(row.get("camera_id") or camera_id),
                frame_id=int(float(row.get("frame_id", 0))),
                class_id=int(float(row.get("class_id", -1))),
                class_name=str(row.get("class_name", "")),
                bbox_xyxy=bbox,
                confidence=float(row.get("confidence", 0.0)),
                image_width=width,
                image_height=height,
                calibration=calibration,
                track_id=optional_int(row.get("local_track_id")),
                subset=str(row.get("subset") or subset),
                split=str(row.get("split") or split),
                local_track_id=optional_int(row.get("local_track_id")),
                global_track_id=optional_int(row.get("global_track_id")),
                candidate_id=_optional_str(row.get("candidate_id")),
            )
        )
    return inputs


def load_scene_camera_calibration(root: Union[str, Path], split: str, scene_name: str, camera_id: str) -> Any:
    """Load calibration object for one scene/camera."""
    scene_paths = get_scene_paths(Path(root), split, scene_name)
    if scene_paths.calibration_path is None or not scene_paths.calibration_path.exists():
        return {}
    return load_calibration_json(scene_paths.calibration_path).get(camera_id, {})


def write_pseudo3d_predictions_jsonl(outputs: List[Pseudo3DOutput], path: Union[str, Path]) -> None:
    """Write pseudo-3D predictions as JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(pseudo3d_output_to_dict(output), sort_keys=True) for output in outputs]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_pseudo3d_predictions_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read pseudo-3D prediction JSONL as dictionaries."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_pseudo3d_predictions_csv(outputs: List[Pseudo3DOutput], path: Union[str, Path]) -> None:
    """Write pseudo-3D predictions as CSV."""
    rows = [flatten_prediction_dict(pseudo3d_output_to_dict(output)) for output in outputs]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_pseudo3d_predictions_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read pseudo-3D prediction CSV as dictionaries."""
    return _read_csv(path)


def flatten_prediction_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten array-valued prediction fields for CSV output."""
    out = dict(row)
    center = row.get("center_3d")
    dims = row.get("dimensions_3d")
    bbox = row.get("bbox_xyxy")
    if isinstance(center, list) and len(center) >= 3:
        out["center_x"] = center[0]
        out["center_y"] = center[1]
        out["center_z"] = center[2]
    if isinstance(dims, list) and len(dims) >= 3:
        out["width_3d"] = dims[0]
        out["length_3d"] = dims[1]
        out["height_3d"] = dims[2]
    if isinstance(bbox, list) and len(bbox) >= 4:
        out["x1"] = bbox[0]
        out["y1"] = bbox[1]
        out["x2"] = bbox[2]
        out["y2"] = bbox[3]
    out.pop("center_3d", None)
    out.pop("dimensions_3d", None)
    out.pop("bbox_xyxy", None)
    return out


def prediction_summary(outputs: List[Pseudo3DOutput]) -> Dict[str, Any]:
    """Build compact prediction summary."""
    failed = [output for output in outputs if output.failure_reason]
    return {
        "num_predictions": len(outputs),
        "num_success": len(outputs) - len(failed),
        "num_failed": len(failed),
        "success_rate": float(len(outputs) - len(failed)) / float(len(outputs)) if outputs else None,
        "failure_reasons": _count_values([output.failure_reason for output in failed]),
        "source_metadata_completeness": source_metadata_completeness(outputs),
        "per_class": _per_output_counts(outputs, "class_id"),
        "per_subset": _per_output_counts(outputs, "subset"),
    }


def source_metadata_completeness(outputs: List[Pseudo3DOutput]) -> Dict[str, Any]:
    """Measure source metadata completeness for generated outputs."""
    total = len(outputs)
    fields = [
        "center_3d_source",
        "dimensions_3d_source",
        "yaw_source",
        "depth_source",
        "pseudo3d_method",
    ]
    summary = {"total": total}
    for field in fields:
        count = sum(1 for output in outputs if getattr(output, field, None) not in (None, "", "unknown"))
        summary["%s_complete" % field] = count
        summary["%s_complete_rate" % field] = float(count) / float(total) if total else None
    estimated = sum(1 for output in outputs if output.is_estimated_for_test)
    summary["is_estimated_for_test_set"] = estimated
    summary["is_estimated_for_test_rate"] = float(estimated) / float(total) if total else None
    return summary


def _read_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _calib_int(calibration: Any, snake_key: str, json_key: str) -> Optional[int]:
    value = None
    if isinstance(calibration, dict):
        value = calibration.get(snake_key)
        if value is None:
            value = calibration.get(json_key)
    else:
        value = getattr(calibration, snake_key, None)
    return optional_int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _count_values(values: List[Any]) -> Dict[str, int]:
    counts = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _per_output_counts(outputs: List[Pseudo3DOutput], field: str) -> Dict[str, int]:
    counts = {}
    for output in outputs:
        key = str(getattr(output, field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    names = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names
