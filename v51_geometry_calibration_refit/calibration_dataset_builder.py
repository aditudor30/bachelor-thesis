"""Build leakage-free fit/holdout/official-val calibration matches."""

import csv
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.v51_geometry_calibration_refit.fit_train_source_builder import ensure_fit_train_sources
from deep_oc_sort_3d.v51_geometry_calibration_refit.gt_prediction_matcher import match_prediction_to_gt
from deep_oc_sort_3d.v51_geometry_calibration_refit.source_availability_audit import discover_source_files
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import internal_to_official, output_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import iter_jsonl, progress_iter, vector3, write_csv
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_metrics import axis_aligned_iou3d


CLASS_NAME_TO_INTERNAL = {
    "Person": 0, "Forklift": 1, "PalletTruck": 2, "Transporter": 3,
    "FourierGR1T2": 4, "AgilityDigit": 5, "NovaCarter": 6,
}

PHASES = ["fit_train", "internal_holdout", "official_val"]
MATCH_FIELDS = [
    "phase", "split", "scene_name", "camera_id", "source_path", "frame_id",
    "internal_class_id", "official_class_id", "class_name", "gt_object_id",
    "match_method", "matched_iou", "coordinate_frame", "bbox_xyxy", "confidence",
    "pseudo3d_method", "center_3d_source", "dimensions_3d_source", "yaw_source",
    "pred_x", "pred_y", "pred_z", "gt_x", "gt_y", "gt_z",
    "pred_width", "pred_length", "pred_height", "gt_width", "gt_length", "gt_height",
    "pred_yaw", "gt_yaw", "pred_distance", "gt_distance", "center_error_before",
    "dimension_error_before", "yaw_error_before", "depth_error_before", "iou3d_proxy_before",
]
METRIC_FIELDS = [
    "center_error_before", "dimension_error_before", "yaw_error_before",
    "depth_error_before", "iou3d_proxy_before",
]
CHECKPOINT_VERSION = 1


def build_v51_calibration_dataset(
    config: Dict[str, Any], progress: bool = True, overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate sources, checkpoint per-source matching, then aggregate on disk."""
    generation = ensure_fit_train_sources(config, progress=progress, overwrite=overwrite)
    root = output_root(config)
    directory = root / "calibration_dataset"
    partial_root = directory / "partial_sources"
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    files = discover_source_files(config)
    source_outputs = []
    total = len(files)

    for index, item in enumerate(progress_iter(files, progress, "V5.1 calibration sources"), 1):
        phase, split, scene, camera, path = item
        shard_path, checkpoint_path = _partial_paths(partial_root, phase, scene, camera)
        gt_path = dataset_root / split / scene / "ground_truth.json"
        signature = _source_signature(path, gt_path, config)
        checkpoint = _read_checkpoint(checkpoint_path)
        source_outputs.append((item, shard_path, checkpoint_path))

        if not overwrite and _checkpoint_is_current(checkpoint, shard_path, signature):
            print(
                "V5.1 calibration source %d/%d scene=%s camera=%s output=%s "
                "matches=%d memory-safe checkpoint=skipped_existing"
                % (index, total, scene, camera, shard_path, int(checkpoint.get("num_matches", 0)))
            )
            continue

        print(
            "V5.1 calibration source %d/%d scene=%s camera=%s input=%s output=%s"
            % (index, total, scene, camera, path, shard_path)
        )
        try:
            checkpoint = _process_source(
                config, phase, split, scene, camera, path, gt_path, shard_path, signature,
            )
            _write_json_atomic(checkpoint_path, checkpoint)
        except Exception as exc:
            print(
                "V5.1 calibration source FAILED %d/%d scene=%s camera=%s input=%s "
                "output=%s error=%s"
                % (index, total, scene, camera, path, shard_path, exc)
            )
            raise
        finally:
            gc.collect()
        print(
            "V5.1 calibration source %d/%d scene=%s camera=%s output=%s "
            "matches=%d memory-safe checkpoint=written"
            % (index, total, scene, camera, shard_path, int(checkpoint.get("num_matches", 0)))
        )

    return _aggregate_outputs(directory, source_outputs, files, generation)


def _process_source(
    config: Dict[str, Any], phase: str, split: str, scene: str, camera: str,
    source_path: Path, gt_path: Path, shard_path: Path, signature: Dict[str, Any],
) -> Dict[str, Any]:
    gt_objects = load_ground_truth_json(gt_path) if gt_path.is_file() else []
    indexed = _gt_index(gt_objects)
    counts: Dict[str, Any] = {
        "num_predictions": 0,
        "num_gt": sum(1 for item in gt_objects if camera in item.visible_bboxes_2d),
        "num_matches": 0,
        "ambiguous_matches_rejected": 0,
        "gt_derived_predictions_rejected": 0,
        "rejection_counts": defaultdict(int),
    }
    temporary = _temporary_path(shard_path)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for prediction in iter_jsonl(source_path):
                counts["num_predictions"] += 1
                if _gt_derived_prediction(prediction):
                    counts["gt_derived_predictions_rejected"] += 1
                    counts["rejection_counts"]["gt_derived_prediction"] += 1
                    continue
                frame_id = _int_value(prediction.get("frame_id"), -1)
                class_id = _prediction_internal_class(prediction)
                if frame_id < 0 or class_id is None:
                    counts["rejection_counts"]["invalid_frame_or_class"] += 1
                    continue
                gt, iou, method = match_prediction_to_gt(
                    prediction, indexed.get((frame_id, class_id), []), camera, config,
                )
                if gt is None:
                    counts["rejection_counts"][method] += 1
                    if method.startswith("ambiguous"):
                        counts["ambiguous_matches_rejected"] += 1
                    continue
                row = _match_row(
                    prediction, gt, phase, split, scene, camera, class_id, iou, method,
                    source_path, config,
                )
                if row is None:
                    counts["rejection_counts"]["missing_required_geometry"] += 1
                    continue
                writer.writerow({key: _csv_value(row.get(key)) for key in MATCH_FIELDS})
                counts["num_matches"] += 1
        temporary.replace(shard_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        indexed.clear()
        gt_objects.clear()
        del indexed
        del gt_objects

    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "status": "complete",
        "phase": phase,
        "split": split,
        "scene": scene,
        "camera": camera,
        "source_path": str(source_path),
        "output_path": str(shard_path),
        "source_signature": signature,
        "num_predictions": int(counts["num_predictions"]),
        "num_gt": int(counts["num_gt"]),
        "num_matches": int(counts["num_matches"]),
        "ambiguous_matches_rejected": int(counts["ambiguous_matches_rejected"]),
        "gt_derived_predictions_rejected": int(counts["gt_derived_predictions_rejected"]),
        "rejection_counts": dict(counts["rejection_counts"]),
    }


def _aggregate_outputs(
    directory: Path,
    source_outputs: Sequence[Tuple[Tuple[str, str, str, str, Path], Path, Path]],
    files: List[Tuple[str, str, str, str, Path]],
    generation: Dict[str, Any],
) -> Dict[str, Any]:
    print("V5.1 calibration aggregation: begin sources=%d" % len(source_outputs))
    counts = _aggregate_counts(source_outputs)
    samples_per_class: Dict[str, int] = defaultdict(int)
    samples_per_scene: Dict[str, int] = defaultdict(int)
    samples_per_camera: Dict[str, int] = defaultdict(int)
    metric_root = directory / ".aggregation_metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    metric_paths = {
        phase: {field: metric_root / (phase + "_" + field + ".float64") for field in METRIC_FIELDS}
        for phase in PHASES
    }
    metric_handles = {
        phase: {field: path.open("wb") for field, path in paths.items()}
        for phase, paths in metric_paths.items()
    }
    filenames = {
        "fit_train": "fit_train_matches.csv",
        "internal_holdout": "internal_holdout_matches.csv",
        "official_val": "official_val_matches.csv",
    }
    final_paths = {phase: directory / filename for phase, filename in filenames.items()}
    final_paths["all"] = directory / "calibration_matches.csv"
    temporary_paths = {key: _temporary_path(path) for key, path in final_paths.items()}
    output_handles: Dict[str, Any] = {}
    output_writers: Dict[str, csv.DictWriter] = {}
    try:
        for key, path in temporary_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS, extrasaction="ignore")
            writer.writeheader()
            output_handles[key] = handle
            output_writers[key] = writer

        for item, shard_path, checkpoint_path in source_outputs:
            phase, _split, scene, camera, _source_path = item
            checkpoint = _read_checkpoint(checkpoint_path)
            if checkpoint.get("status") != "complete" or not shard_path.is_file():
                raise RuntimeError("Missing completed calibration shard: %s" % shard_path)
            buffers = {field: [] for field in METRIC_FIELDS}
            with shard_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    output_writers[phase].writerow(row)
                    output_writers["all"].writerow(row)
                    samples_per_class[str(row.get("official_class_id", ""))] += 1
                    samples_per_scene[str(row.get("scene_name", scene))] += 1
                    samples_per_camera[str(row.get("camera_id", camera))] += 1
                    for field in METRIC_FIELDS:
                        value = _finite_float(row.get(field))
                        if value is not None:
                            buffers[field].append(value)
                    if len(buffers[METRIC_FIELDS[0]]) >= 4096:
                        _flush_metric_buffers(buffers, metric_handles[phase])
            _flush_metric_buffers(buffers, metric_handles[phase])
            buffers.clear()
    except Exception:
        for handle in output_handles.values():
            handle.close()
        for handles in metric_handles.values():
            for handle in handles.values():
                handle.close()
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        raise
    else:
        for handle in output_handles.values():
            handle.close()
        for handles in metric_handles.values():
            for handle in handles.values():
                handle.close()
        for key, path in final_paths.items():
            if key == "all":
                has_rows = sum(int(counts.get(phase, {}).get("num_matches", 0)) for phase in PHASES) > 0
            else:
                has_rows = int(counts.get(key, {}).get("num_matches", 0)) > 0
            if has_rows:
                temporary_paths[key].replace(path)
            else:
                temporary_paths[key].unlink()
                _write_csv_atomic(path, [])

    total_matches = sum(int(counts.get(phase, {}).get("num_matches", 0)) for phase in PHASES)
    print("V5.1 calibration aggregation: CSV complete matches=%d" % total_matches)
    phase_metrics = {}
    for phase in PHASES:
        phase_matches = int(counts.get(phase, {}).get("num_matches", 0))
        print("V5.1 calibration aggregation: metrics phase=%s matches=%d" % (phase, phase_matches))
        phase_metrics[phase] = _summarize_metric_files(metric_paths[phase], phase_matches)
    summary = _summary(
        counts, files, generation, phase_metrics,
        samples_per_class, samples_per_scene, samples_per_camera,
    )
    _write_json_atomic(directory / "match_rate_summary.json", summary)
    _write_json_atomic(directory / "calibration_matches_summary.json", summary)
    _write_csv_atomic(directory / "samples_per_class.csv", _counter_rows(samples_per_class, "official_class_id"))
    _write_csv_atomic(directory / "samples_per_scene.csv", _counter_rows(samples_per_scene, "scene_name"))
    _write_json_atomic(directory / "rejected_ambiguous_matches_summary.json", {
        phase: {
            "ambiguous_matches_rejected": int(values["ambiguous_matches_rejected"]),
            "rejection_counts": dict(values["rejection_counts"]),
        }
        for phase, values in counts.items()
    })
    for paths in metric_paths.values():
        for path in paths.values():
            if path.exists():
                path.unlink()
    if metric_root.exists():
        metric_root.rmdir()
    print("V5.1 calibration aggregation: complete matches=%d" % total_matches)
    return summary


def _aggregate_counts(
    source_outputs: Sequence[Tuple[Tuple[str, str, str, str, Path], Path, Path]],
) -> Dict[str, Dict[str, Any]]:
    counts: Dict[str, Dict[str, Any]] = defaultdict(_empty_counts)
    for item, shard_path, checkpoint_path in source_outputs:
        phase = item[0]
        checkpoint = _read_checkpoint(checkpoint_path)
        if checkpoint.get("status") != "complete" or not shard_path.is_file():
            raise RuntimeError("Missing completed calibration checkpoint: %s" % checkpoint_path)
        values = counts[phase]
        for key in [
            "num_predictions", "num_gt", "num_matches", "ambiguous_matches_rejected",
            "gt_derived_predictions_rejected",
        ]:
            values[key] += int(checkpoint.get(key, 0))
        for reason, count in checkpoint.get("rejection_counts", {}).items():
            values["rejection_counts"][str(reason)] += int(count)
    return counts


def _empty_counts() -> Dict[str, Any]:
    return {
        "num_predictions": 0, "num_gt": 0, "num_matches": 0,
        "ambiguous_matches_rejected": 0, "gt_derived_predictions_rejected": 0,
        "rejection_counts": defaultdict(int),
    }


def _summary(
    counts: Dict[str, Dict[str, Any]], files: List[Tuple[str, str, str, str, Path]],
    generation: Dict[str, Any], phase_metrics: Dict[str, Dict[str, Any]],
    samples_per_class: Dict[str, int], samples_per_scene: Dict[str, int],
    samples_per_camera: Dict[str, int],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"source_files": len(files), "source_generation": generation}
    for phase in PHASES:
        values = counts.get(phase, {})
        predictions = int(values.get("num_predictions", 0))
        matches = int(values.get("num_matches", 0))
        result[phase] = {
            "num_predictions": predictions, "num_gt": int(values.get("num_gt", 0)),
            "num_matches": matches, "match_rate": None if predictions == 0 else float(matches) / float(predictions),
            "ambiguous_matches_rejected": int(values.get("ambiguous_matches_rejected", 0)),
            "gt_derived_predictions_rejected": int(values.get("gt_derived_predictions_rejected", 0)),
            "rejection_counts": dict(values.get("rejection_counts", {})),
            "metrics": phase_metrics.get(phase, {}),
        }
        result["%s_num_predictions" % phase] = predictions
        result["%s_num_gt" % phase] = int(values.get("num_gt", 0))
        result["%s_num_matches" % phase] = matches
        result["%s_match_rate" % phase] = result[phase]["match_rate"]
    result["ambiguous_matches_rejected"] = sum(
        int(counts.get(phase, {}).get("ambiguous_matches_rejected", 0)) for phase in counts
    )
    result["samples_per_class"] = _sorted_counter(samples_per_class)
    result["samples_per_scene"] = _sorted_counter(samples_per_scene)
    result["samples_per_camera"] = _sorted_counter(samples_per_camera)
    return result


def _summarize_metric_files(paths: Dict[str, Path], samples: int) -> Dict[str, Any]:
    center_mean, center_pct = _metric_stats(paths["center_error_before"], [50, 75, 90, 95])
    dimension_mean, dimension_pct = _metric_stats(paths["dimension_error_before"], [50, 90])
    yaw_mean, yaw_pct = _metric_stats(paths["yaw_error_before"], [50, 90])
    depth_mean, depth_pct = _metric_stats(paths["depth_error_before"], [50])
    iou_mean, iou_pct = _metric_stats(paths["iou3d_proxy_before"], [50])
    return {
        "samples": samples,
        "center_error_mean": center_mean, "center_error_median": center_pct.get(50),
        "center_error_p75": center_pct.get(75), "center_error_p90": center_pct.get(90),
        "center_error_p95": center_pct.get(95),
        "dimension_error_mean": dimension_mean, "dimension_error_median": dimension_pct.get(50),
        "dimension_error_p90": dimension_pct.get(90),
        "yaw_error_mean": yaw_mean, "yaw_error_median": yaw_pct.get(50),
        "yaw_error_p90": yaw_pct.get(90),
        "depth_error_mean": depth_mean, "depth_error_median": depth_pct.get(50),
        "3d_iou_proxy_mean": iou_mean, "3d_iou_proxy_median": iou_pct.get(50),
    }


def _metric_stats(path: Path, percentiles: Sequence[int]) -> Tuple[Optional[float], Dict[int, float]]:
    size = path.stat().st_size // np.dtype(np.float64).itemsize if path.is_file() else 0
    if size == 0:
        return None, {}
    values = np.memmap(str(path), dtype=np.float64, mode="r+", shape=(size,))
    try:
        mean = float(np.mean(values))
        # NumPy percentile may copy a memmap. Sorting the temporary metric file
        # in place keeps peak RAM bounded independently of the match count.
        values.sort(kind="heapsort")
        result = {}
        for percentile in percentiles:
            rank = (float(percentile) / 100.0) * float(size - 1)
            lower = int(np.floor(rank))
            upper = int(np.ceil(rank))
            fraction = rank - float(lower)
            lower_value = float(values[lower])
            upper_value = float(values[upper])
            result[int(percentile)] = lower_value + (upper_value - lower_value) * fraction
        values.flush()
        return mean, result
    finally:
        del values
        gc.collect()


def _flush_metric_buffers(buffers: Dict[str, List[float]], handles: Dict[str, Any]) -> None:
    for field, buffer in buffers.items():
        if buffer:
            np.asarray(buffer, dtype=np.float64).tofile(handles[field])
            buffer.clear()


def _partial_paths(partial_root: Path, phase: str, scene: str, camera: str) -> Tuple[Path, Path]:
    directory = partial_root / _safe_name(phase) / _safe_name(scene)
    identity = "%s|%s|%s" % (phase, scene, camera)
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    stem = "%s_%s" % (_safe_name(camera), digest)
    return directory / (stem + ".csv"), directory / (stem + ".checkpoint.json")


def _safe_name(value: str) -> str:
    result = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in str(value))
    return result or "unknown"


def _source_signature(source_path: Path, gt_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "source": _path_signature(source_path),
        "ground_truth": _path_signature(gt_path),
        "matching": config.get("matching", {}),
        "class_mapping": config.get("class_mapping", {}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _path_signature(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path), "exists": True, "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))),
    }


def _checkpoint_is_current(checkpoint: Dict[str, Any], shard_path: Path, signature: Dict[str, Any]) -> bool:
    return (
        checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION
        and checkpoint.get("status") == "complete"
        and checkpoint.get("source_signature", {}).get("digest") == signature.get("digest")
        and shard_path.is_file()
    )


def _read_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = _temporary_path(path)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    temporary = _temporary_path(path)
    write_csv(temporary, rows)
    temporary.replace(path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else value


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _counter_rows(values: Dict[str, int], field: str) -> List[Dict[str, Any]]:
    return [{field: key, "samples": count} for key, count in _sorted_counter(values).items()]


def _sorted_counter(values: Dict[str, int]) -> Dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in values.items()))


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
