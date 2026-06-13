"""Build the V5 train/val-only pseudo3D calibration dataset."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import (
    calibration_scene_lookup,
    internal_to_official,
    observation_source_roots,
    output_root,
)
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import iter_jsonl, progress_iter, vector3, write_csv, write_json
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_metrics import axis_aligned_iou3d, summarize_match_rows
from deep_oc_sort_3d.v5_geometry_calibration.gt_prediction_matcher import match_prediction_to_gt


CLASS_NAME_TO_INTERNAL = {
    "Person": 0, "Forklift": 1, "PalletTruck": 2, "Transporter": 3,
    "FourierGR1T2": 4, "AgilityDigit": 5, "NovaCarter": 6,
}


def build_calibration_dataset(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Match available train/val pseudo3D observations to GT and write diagnostics."""
    root = output_root(config)
    scene_lookup = calibration_scene_lookup(config)
    files = _select_observation_files(config, scene_lookup)
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    gt_cache: Dict[Tuple[str, str], List[GroundTruthObject]] = {}
    match_rows: List[Dict[str, Any]] = []
    num_predictions = 0
    num_gt_visible = 0
    ambiguous = 0
    rejection_counts = defaultdict(int)
    gt_counted_files = set()
    for item in progress_iter(files, progress, "V5 calibration observation files"):
        phase, split, scene_name, camera_id, path = item
        cache_key = (split, scene_name)
        if cache_key not in gt_cache:
            gt_path = dataset_root / split / scene_name / "ground_truth.json"
            gt_cache[cache_key] = load_ground_truth_json(gt_path) if gt_path.is_file() else []
        gt_objects = gt_cache[cache_key]
        by_frame_class = _gt_index(gt_objects)
        file_key = (scene_name, camera_id)
        if file_key not in gt_counted_files:
            num_gt_visible += sum(1 for obj in gt_objects if camera_id in obj.visible_bboxes_2d)
            gt_counted_files.add(file_key)
        for prediction in iter_jsonl(path):
            num_predictions += 1
            frame_id = _int_value(prediction.get("frame_id"), -1)
            internal_class = _prediction_internal_class(prediction)
            if frame_id < 0 or internal_class is None:
                rejection_counts["invalid_frame_or_class"] += 1
                continue
            candidates = by_frame_class.get((frame_id, internal_class), [])
            gt, matched_iou, match_method = match_prediction_to_gt(prediction, candidates, camera_id, config)
            if gt is None:
                rejection_counts[match_method] += 1
                if match_method.startswith("ambiguous"):
                    ambiguous += 1
                continue
            row = _calibration_row(prediction, gt, phase, split, scene_name, camera_id, internal_class, matched_iou, match_method, config)
            if row is None:
                rejection_counts["missing_required_geometry"] += 1
                continue
            match_rows.append(row)
    summary = _dataset_summary(match_rows, num_predictions, num_gt_visible, ambiguous, rejection_counts, files)
    dataset_dir = root / "calibration_dataset"
    write_csv(dataset_dir / "calibration_matches.csv", match_rows)
    write_json(dataset_dir / "calibration_matches_summary.json", summary)
    write_csv(dataset_dir / "per_class_error_summary_before.csv", _group_summaries(match_rows, "official_class_id"))
    write_csv(dataset_dir / "per_camera_error_summary_before.csv", _group_summaries(match_rows, "camera_id"))
    write_json(dataset_dir / "match_rate_summary.json", {
        "num_predictions": num_predictions, "num_gt": num_gt_visible, "num_matches": len(match_rows),
        "match_rate": summary.get("match_rate"), "ambiguous_matches_rejected": ambiguous,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    })
    return summary


def _select_observation_files(config: Dict[str, Any], scene_lookup: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str, str, Path]]:
    selected: Dict[Tuple[str, str], Tuple[str, str, str, str, Path]] = {}
    for source_root in observation_source_roots(config):
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.jsonl")):
            scene_name = next((part for part in path.parts if part in scene_lookup), None)
            if scene_name is None:
                continue
            camera_id = path.stem
            key = (scene_name, camera_id)
            if key in selected:
                continue
            info = scene_lookup[scene_name]
            selected[key] = (info["phase"], info["split"], scene_name, camera_id, path)
    return sorted(selected.values(), key=lambda item: (item[0], item[2], item[3]))


def _gt_index(objects: List[GroundTruthObject]) -> Dict[Tuple[int, int], List[GroundTruthObject]]:
    output: Dict[Tuple[int, int], List[GroundTruthObject]] = defaultdict(list)
    for obj in objects:
        class_id = CLASS_NAME_TO_INTERNAL.get(obj.object_type)
        if class_id is not None:
            output[(int(obj.frame_id), int(class_id))].append(obj)
    return output


def _calibration_row(
    prediction: Dict[str, Any], gt: GroundTruthObject, phase: str, split: str, scene_name: str,
    camera_id: str, internal_class: int, matched_iou: Optional[float], match_method: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    pred_center = vector3(prediction.get("center_3d"))
    pred_dims = vector3(prediction.get("dimensions_3d"))
    pred_yaw = _float_value(prediction.get("yaw"))
    if pred_center is None or pred_dims is None or pred_yaw is None or np.any(pred_dims <= 0.0):
        return None
    gt_center = np.asarray(gt.location_3d, dtype=float)
    gt_dims = np.asarray(gt.bbox3d_scale, dtype=float)
    gt_yaw = float(gt.bbox3d_rotation[-1])
    official_class = internal_to_official(config, internal_class)
    if official_class is None or np.any(gt_dims <= 0.0):
        return None
    pred_distance = float(np.linalg.norm(pred_center))
    gt_distance = float(np.linalg.norm(gt_center))
    yaw_error = abs(float((gt_yaw - pred_yaw + np.pi) % (2.0 * np.pi) - np.pi))
    return {
        "phase": phase, "split": split, "scene_name": scene_name, "camera_id": camera_id,
        "coordinate_frame": str(prediction.get("coordinate_frame") or "unknown").lower(),
        "frame_id": int(gt.frame_id), "internal_class_id": internal_class, "official_class_id": official_class,
        "class_name": gt.object_type, "gt_object_id": int(gt.object_id), "match_method": match_method,
        "matched_iou": matched_iou, "confidence": prediction.get("confidence"),
        "bbox_xyxy": prediction.get("bbox_xyxy"), "depth_estimate": prediction.get("depth_value"),
        "pred_x": float(pred_center[0]), "pred_y": float(pred_center[1]), "pred_z": float(pred_center[2]),
        "gt_x": float(gt_center[0]), "gt_y": float(gt_center[1]), "gt_z": float(gt_center[2]),
        "pred_width": float(pred_dims[0]), "pred_length": float(pred_dims[1]), "pred_height": float(pred_dims[2]),
        "gt_width": float(gt_dims[0]), "gt_length": float(gt_dims[1]), "gt_height": float(gt_dims[2]),
        "pred_yaw": pred_yaw, "gt_yaw": gt_yaw,
        "center_error_before": float(np.linalg.norm(pred_center - gt_center)),
        "dimension_error_before": float(np.mean(np.abs(pred_dims - gt_dims))),
        "yaw_error_before": yaw_error, "pred_distance": pred_distance, "gt_distance": gt_distance,
        "depth_error_before": abs(pred_distance - gt_distance),
        "iou3d_proxy_before": axis_aligned_iou3d(pred_center, pred_dims, gt_center, gt_dims),
    }


def _dataset_summary(
    rows: List[Dict[str, Any]], num_predictions: int, num_gt: int, ambiguous: int,
    rejection_counts: Dict[str, int], files: List[Tuple[str, str, str, str, Path]],
) -> Dict[str, Any]:
    per_class = defaultdict(int)
    per_scene = defaultdict(int)
    per_camera = defaultdict(int)
    per_phase = defaultdict(int)
    per_coordinate_frame = defaultdict(int)
    for row in rows:
        per_class[str(row["official_class_id"])] += 1
        per_scene[str(row["scene_name"])] += 1
        per_camera[str(row["camera_id"])] += 1
        per_phase[str(row["phase"])] += 1
        per_coordinate_frame[str(row.get("coordinate_frame", "unknown"))] += 1
    summary = summarize_match_rows(rows)
    summary.update({
        "num_predictions": num_predictions, "num_gt": num_gt, "num_matches": len(rows),
        "match_rate": None if num_predictions <= 0 else float(len(rows)) / float(num_predictions),
        "ambiguous_matches_rejected": ambiguous, "samples_per_class": dict(sorted(per_class.items(), key=lambda item: int(item[0]))),
        "samples_per_scene": dict(sorted(per_scene.items())), "samples_per_camera": dict(sorted(per_camera.items())),
        "samples_per_phase": dict(sorted(per_phase.items())), "source_files": len(files),
        "samples_per_coordinate_frame": dict(sorted(per_coordinate_frame.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    })
    return summary


def _group_summaries(rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, ""))].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        summary = summarize_match_rows(values)
        summary[field] = key
        output.append(summary)
    return output


def _prediction_internal_class(prediction: Dict[str, Any]) -> Optional[int]:
    value = prediction.get("class_id")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return CLASS_NAME_TO_INTERNAL.get(str(prediction.get("class_name", "")))


def _int_value(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float_value(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None
