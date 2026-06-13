"""Build leakage-free fit/holdout/official-val calibration matches."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.v51_geometry_calibration_refit.fit_train_source_builder import ensure_fit_train_sources
from deep_oc_sort_3d.v51_geometry_calibration_refit.gt_prediction_matcher import match_prediction_to_gt
from deep_oc_sort_3d.v51_geometry_calibration_refit.source_availability_audit import discover_source_files
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import internal_to_official, output_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import iter_jsonl, progress_iter, vector3, write_csv, write_json
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_metrics import axis_aligned_iou3d, summarize_match_rows


CLASS_NAME_TO_INTERNAL = {
    "Person": 0, "Forklift": 1, "PalletTruck": 2, "Transporter": 3,
    "FourierGR1T2": 4, "AgilityDigit": 5, "NovaCarter": 6,
}


def build_v51_calibration_dataset(
    config: Dict[str, Any], progress: bool = True, overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate missing fit sources if needed, then match predictions to train/val GT."""
    generation = ensure_fit_train_sources(config, progress=progress, overwrite=overwrite)
    root = output_root(config)
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    files = discover_source_files(config)
    rows_by_phase: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    counts: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "num_predictions": 0, "num_gt": 0, "num_matches": 0,
        "ambiguous_matches_rejected": 0, "gt_derived_predictions_rejected": 0,
        "rejection_counts": defaultdict(int),
    })
    gt_cache: Dict[Tuple[str, str], List[GroundTruthObject]] = {}
    gt_counted = set()
    for phase, split, scene, camera, path in progress_iter(files, progress, "V5.1 calibration sources"):
        key = (split, scene)
        if key not in gt_cache:
            gt_path = dataset_root / split / scene / "ground_truth.json"
            gt_cache[key] = load_ground_truth_json(gt_path) if gt_path.is_file() else []
        gt_objects = gt_cache[key]
        indexed = _gt_index(gt_objects)
        if (phase, scene, camera) not in gt_counted:
            counts[phase]["num_gt"] += sum(1 for item in gt_objects if camera in item.visible_bboxes_2d)
            gt_counted.add((phase, scene, camera))
        for prediction in iter_jsonl(path):
            counts[phase]["num_predictions"] += 1
            if _gt_derived_prediction(prediction):
                counts[phase]["gt_derived_predictions_rejected"] += 1
                counts[phase]["rejection_counts"]["gt_derived_prediction"] += 1
                continue
            frame_id = _int_value(prediction.get("frame_id"), -1)
            class_id = _prediction_internal_class(prediction)
            if frame_id < 0 or class_id is None:
                counts[phase]["rejection_counts"]["invalid_frame_or_class"] += 1
                continue
            gt, iou, method = match_prediction_to_gt(prediction, indexed.get((frame_id, class_id), []), camera, config)
            if gt is None:
                counts[phase]["rejection_counts"][method] += 1
                if method.startswith("ambiguous"):
                    counts[phase]["ambiguous_matches_rejected"] += 1
                continue
            row = _match_row(prediction, gt, phase, split, scene, camera, class_id, iou, method, path, config)
            if row is None:
                counts[phase]["rejection_counts"]["missing_required_geometry"] += 1
                continue
            rows_by_phase[phase].append(row)
            counts[phase]["num_matches"] += 1
    all_rows = [row for phase in ["fit_train", "internal_holdout", "official_val"] for row in rows_by_phase.get(phase, [])]
    directory = root / "calibration_dataset"
    write_csv(directory / "calibration_matches.csv", all_rows)
    filenames = {
        "fit_train": "fit_train_matches.csv",
        "internal_holdout": "internal_holdout_matches.csv",
        "official_val": "official_val_matches.csv",
    }
    for phase, filename in filenames.items():
        write_csv(directory / filename, rows_by_phase.get(phase, []))
    summary = _summary(rows_by_phase, counts, files, generation)
    write_json(directory / "match_rate_summary.json", summary)
    write_json(directory / "calibration_matches_summary.json", summary)
    write_csv(directory / "samples_per_class.csv", _count_rows(all_rows, "official_class_id"))
    write_csv(directory / "samples_per_scene.csv", _count_rows(all_rows, "scene_name"))
    write_json(directory / "rejected_ambiguous_matches_summary.json", {
        phase: {
            "ambiguous_matches_rejected": int(values["ambiguous_matches_rejected"]),
            "rejection_counts": dict(values["rejection_counts"]),
        }
        for phase, values in counts.items()
    })
    return summary


def _gt_index(objects: List[GroundTruthObject]) -> Dict[Tuple[int, int], List[GroundTruthObject]]:
    result: Dict[Tuple[int, int], List[GroundTruthObject]] = defaultdict(list)
    for item in objects:
        class_id = CLASS_NAME_TO_INTERNAL.get(item.object_type)
        if class_id is not None:
            result[(int(item.frame_id), class_id)].append(item)
    return result


def _match_row(
    prediction: Dict[str, Any], gt: GroundTruthObject, phase: str, split: str, scene: str,
    camera: str, internal_class: int, matched_iou: Optional[float], match_method: str,
    source_path: Path, config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    center = vector3(prediction.get("center_3d"))
    dims = vector3(prediction.get("dimensions_3d"))
    yaw = _float_value(prediction.get("yaw"))
    if center is None or dims is None or yaw is None or np.any(dims <= 0.0):
        return None
    gt_center = np.asarray(gt.location_3d, dtype=float)
    gt_dims = np.asarray(gt.bbox3d_scale, dtype=float)
    gt_yaw = float(gt.bbox3d_rotation[-1])
    official_class = internal_to_official(config, internal_class)
    if official_class is None or np.any(gt_dims <= 0.0):
        return None
    pred_distance = float(np.linalg.norm(center))
    gt_distance = float(np.linalg.norm(gt_center))
    yaw_error = abs(float((gt_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi))
    return {
        "phase": phase, "split": split, "scene_name": scene, "camera_id": camera,
        "source_path": str(source_path), "frame_id": int(gt.frame_id),
        "internal_class_id": internal_class, "official_class_id": official_class,
        "class_name": gt.object_type, "gt_object_id": int(gt.object_id),
        "match_method": match_method, "matched_iou": matched_iou,
        "coordinate_frame": str(prediction.get("coordinate_frame") or "unknown").lower(),
        "bbox_xyxy": prediction.get("bbox_xyxy"),
        "confidence": prediction.get("confidence", prediction.get("confidence_2d")),
        "pseudo3d_method": prediction.get("pseudo3d_method"),
        "center_3d_source": prediction.get("center_3d_source"),
        "dimensions_3d_source": prediction.get("dimensions_3d_source"),
        "yaw_source": prediction.get("yaw_source"),
        "pred_x": float(center[0]), "pred_y": float(center[1]), "pred_z": float(center[2]),
        "gt_x": float(gt_center[0]), "gt_y": float(gt_center[1]), "gt_z": float(gt_center[2]),
        "pred_width": float(dims[0]), "pred_length": float(dims[1]), "pred_height": float(dims[2]),
        "gt_width": float(gt_dims[0]), "gt_length": float(gt_dims[1]), "gt_height": float(gt_dims[2]),
        "pred_yaw": yaw, "gt_yaw": gt_yaw, "pred_distance": pred_distance, "gt_distance": gt_distance,
        "center_error_before": float(np.linalg.norm(center - gt_center)),
        "dimension_error_before": float(np.mean(np.abs(dims - gt_dims))),
        "yaw_error_before": yaw_error, "depth_error_before": abs(pred_distance - gt_distance),
        "iou3d_proxy_before": axis_aligned_iou3d(center, dims, gt_center, gt_dims),
    }


def _summary(
    rows_by_phase: Dict[str, List[Dict[str, Any]]], counts: Dict[str, Dict[str, Any]],
    files: List[Tuple[str, str, str, str, Path]], generation: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"source_files": len(files), "source_generation": generation}
    all_rows: List[Dict[str, Any]] = []
    for phase in ["fit_train", "internal_holdout", "official_val"]:
        rows = rows_by_phase.get(phase, [])
        all_rows.extend(rows)
        values = counts.get(phase, {})
        predictions = int(values.get("num_predictions", 0))
        matches = int(values.get("num_matches", 0))
        result[phase] = {
            "num_predictions": predictions, "num_gt": int(values.get("num_gt", 0)),
            "num_matches": matches, "match_rate": None if predictions == 0 else float(matches) / float(predictions),
            "ambiguous_matches_rejected": int(values.get("ambiguous_matches_rejected", 0)),
            "gt_derived_predictions_rejected": int(values.get("gt_derived_predictions_rejected", 0)),
            "rejection_counts": dict(values.get("rejection_counts", {})),
            "metrics": summarize_match_rows(rows),
        }
        result["%s_num_predictions" % phase] = predictions
        result["%s_num_gt" % phase] = int(values.get("num_gt", 0))
        result["%s_num_matches" % phase] = matches
        result["%s_match_rate" % phase] = result[phase]["match_rate"]
    result["ambiguous_matches_rejected"] = sum(int(counts.get(phase, {}).get("ambiguous_matches_rejected", 0)) for phase in counts)
    result["samples_per_class"] = _count_dict(all_rows, "official_class_id")
    result["samples_per_scene"] = _count_dict(all_rows, "scene_name")
    result["samples_per_camera"] = _count_dict(all_rows, "camera_id")
    return result


def _gt_derived_prediction(row: Dict[str, Any]) -> bool:
    if row.get("is_gt_derived") is True or str(row.get("is_gt_derived", "")).lower() in ("1", "true", "yes"):
        return True
    matched = row.get("matched_gt") is True or str(row.get("matched_gt", "")).lower() in ("1", "true", "yes")
    return matched and row.get("is_gt_derived") is not False


def _prediction_internal_class(row: Dict[str, Any]) -> Optional[int]:
    try:
        return int(float(row.get("class_id")))
    except (TypeError, ValueError):
        return CLASS_NAME_TO_INTERNAL.get(str(row.get("class_name", "")))


def _count_dict(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    values: Dict[str, int] = defaultdict(int)
    for row in rows:
        values[str(row.get(field, ""))] += 1
    return dict(sorted(values.items()))


def _count_rows(rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    return [{field: key, "samples": value} for key, value in _count_dict(rows, field).items()]


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
