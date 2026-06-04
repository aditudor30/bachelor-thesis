"""I/O helpers for stabilized pseudo-3D predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from deep_oc_sort_3d.audit3d.audit3d_io import read_csv_dicts, read_json_if_exists, read_jsonl_dicts, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import write_pseudo3d_predictions_csv, write_pseudo3d_predictions_jsonl
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


def read_pseudo3d_outputs_jsonl(path: Union[str, Path]) -> List[Pseudo3DOutput]:
    """Read pseudo-3D JSONL predictions as dataclass outputs."""
    return [pseudo3d_output_from_dict(row) for row in read_jsonl_dicts(path)]


def read_pseudo3d_outputs_csv(path: Union[str, Path]) -> List[Pseudo3DOutput]:
    """Read pseudo-3D CSV predictions as dataclass outputs."""
    return [pseudo3d_output_from_dict(row) for row in read_csv_dicts(path)]


def read_pseudo3d_outputs(path: Union[str, Path]) -> List[Pseudo3DOutput]:
    """Read pseudo-3D predictions from JSONL or CSV."""
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        return read_pseudo3d_outputs_jsonl(input_path)
    return read_pseudo3d_outputs_csv(input_path)


def write_stabilized_outputs_jsonl(outputs: List[Pseudo3DOutput], path: Union[str, Path]) -> None:
    """Write stabilized outputs as JSONL."""
    write_pseudo3d_predictions_jsonl(outputs, path)


def write_stabilized_outputs_csv(outputs: List[Pseudo3DOutput], path: Union[str, Path]) -> None:
    """Write stabilized outputs as CSV."""
    write_pseudo3d_predictions_csv(outputs, path)


def write_smoothing_report_json(report: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write one smoothing report JSON."""
    write_json(report, path)


def write_smoothing_report_csv(rows: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Write smoothing report rows as CSV."""
    write_csv(rows, path)


def read_summary(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON summary if it exists."""
    return read_json_if_exists(path)


def write_summary(summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a JSON summary."""
    write_json(summary, path)


def pseudo3d_output_from_dict(row: Dict[str, Any]) -> Pseudo3DOutput:
    """Convert a JSONL/CSV prediction dictionary to Pseudo3DOutput."""
    return Pseudo3DOutput(
        center_3d=_array3(row.get("center_3d"), row.get("center_x"), row.get("center_y"), row.get("center_z")),
        dimensions_3d=_array3(row.get("dimensions_3d"), row.get("width_3d"), row.get("length_3d"), row.get("height_3d")),
        yaw=_optional_float(row.get("yaw")),
        depth=_optional_float(row.get("depth")),
        confidence_3d=_float_or(row.get("confidence_3d"), 0.0),
        center_3d_source=str(row.get("center_3d_source") or "unknown"),
        dimensions_3d_source=str(row.get("dimensions_3d_source") or "unknown"),
        yaw_source=str(row.get("yaw_source") or "unknown"),
        depth_source=str(row.get("depth_source") or "unknown"),
        is_gt_derived=_optional_bool(row.get("is_gt_derived"), False),
        is_estimated_for_test=_optional_bool(row.get("is_estimated_for_test"), True),
        pseudo3d_method=str(row.get("pseudo3d_method") or "bbox_height_depth"),
        pseudo3d_version=str(row.get("pseudo3d_version") or ""),
        subset=str(row.get("subset") or ""),
        split=str(row.get("split") or ""),
        scene_name=str(row.get("scene_name") or ""),
        camera_id=str(row.get("camera_id") or ""),
        frame_id=_int_or(row.get("frame_id"), -1),
        class_id=_int_or(row.get("class_id"), -1),
        class_name=str(row.get("class_name") or ""),
        local_track_id=_optional_int(row.get("local_track_id")),
        global_track_id=_optional_int(row.get("global_track_id")),
        candidate_id=_optional_str(row.get("candidate_id")),
        bbox_xyxy=_bbox(row),
        confidence_2d=_first_float_or([row.get("confidence_2d"), row.get("confidence")], 0.0),
        coordinate_frame=str(row.get("coordinate_frame") or "unknown"),
        projection_valid=_optional_bool_or_none(row.get("projection_valid")),
        projection_error_reason=_optional_str(row.get("projection_error_reason")),
        failure_reason=_optional_str(row.get("failure_reason")),
        source_notes=str(row.get("source_notes") or ""),
    )


def outputs_to_dicts(outputs: List[Pseudo3DOutput]) -> List[Dict[str, Any]]:
    """Convert outputs to dictionaries via JSON-safe serialization."""
    rows = []
    for output in outputs:
        text = json.dumps(_jsonable_output(output), sort_keys=True)
        rows.append(json.loads(text))
    return rows


def _jsonable_output(output: Pseudo3DOutput) -> Dict[str, Any]:
    from deep_oc_sort_3d.pseudo3d.pseudo3d_types import pseudo3d_output_to_dict

    return pseudo3d_output_to_dict(output)


def _array3(value: Any, x_value: Any = None, y_value: Any = None, z_value: Any = None) -> Optional[np.ndarray]:
    values = value
    if values in (None, ""):
        values = [x_value, y_value, z_value]
    if isinstance(values, str):
        values = _parse_sequence_string(values)
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3].astype(float)


def _bbox(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    value = row.get("bbox_xyxy")
    if isinstance(value, str):
        value = _parse_sequence_string(value)
    if isinstance(value, list) and len(value) >= 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return (
        float(_optional_float(row.get("x1")) or 0.0),
        float(_optional_float(row.get("y1")) or 0.0),
        float(_optional_float(row.get("x2")) or 0.0),
        float(_optional_float(row.get("y2")) or 0.0),
    )


def _parse_sequence_string(value: str) -> List[float]:
    text = value.strip().strip("[]()")
    if not text:
        return []
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    out = []
    for part in parts:
        if part:
            out.append(float(part))
    return out


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _optional_int(value: Any) -> Optional[int]:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _float_or(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return float(default) if parsed is None else float(parsed)


def _first_float_or(values: List[Any], default: float) -> float:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return float(parsed)
    return float(default)


def _int_or(value: Any, default: int) -> int:
    parsed = _optional_float(value)
    return int(default) if parsed is None else int(parsed)


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _optional_bool(value: Any, default: bool) -> bool:
    parsed = _optional_bool_or_none(value)
    return default if parsed is None else parsed


def _optional_bool_or_none(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None
