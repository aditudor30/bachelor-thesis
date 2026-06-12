"""GT object-frame retention audit for train/validation scenes only."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.bytetrack_audit.audit_config import audit_scenes, output_root, variant_paths
from deep_oc_sort_3d.bytetrack_audit.audit_io import iter_csv, iter_jsonl, progress_iter, safe_int, write_csv, write_json
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.tracking.association import bbox_iou_xyxy


CLASS_MAPPING = {
    "Person": 0,
    "Forklift": 1,
    "PalletTruck": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "NovaCarter": 6,
}


def run_gt_object_frame_audit(
    config: Dict[str, Any],
    progress: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Match visible GT boxes to each local output and trace the 21C match downstream."""
    if not bool(config.get("gt_audit", {}).get("enabled", True)):
        return {"status": "disabled", "summary_rows": [], "drop_samples": []}
    min_iou = float(config.get("gt_audit", {}).get("min_iou_for_gt_match", 0.3))
    sample_limit = int(max_samples or config.get("samples", {}).get("max_samples_per_category", 50))
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    scenes = [item for item in audit_scenes(config, include_test=False) if item[1] in ("train", "val")]
    trace_counts = {}
    bucket_counts = {}
    scene_counts = {}
    class_counts = {}
    camera_counts = {}
    drop_samples = []
    warnings = []
    for subset, split, scene_name in progress_iter(scenes, progress, "GT object-frame scenes"):
        gt_path = dataset_root / split / scene_name / "ground_truth.json"
        if not gt_path.exists():
            warnings.append("missing GT: %s" % gt_path)
            continue
        objects = load_ground_truth_json(gt_path)
        durations = _object_durations(objects)
        cameras = sorted(set(camera for obj in objects for camera in obj.visible_bboxes_2d.keys()))
        for camera_id in progress_iter(cameras, progress, "%s GT cameras" % scene_name):
            local_by_variant = {
                variant: _load_local_rows(variant_paths(config, variant).get("local_tracks_root", Path("")), subset, scene_name, camera_id)
                for variant in ("v2_current", "bytetrack_21b", "bytetrack_21c_best")
            }
            downstream = _load_downstream_indices(
                variant_paths(config, "bytetrack_21c_best"), subset, scene_name, camera_id
            )
            visible = [obj for obj in objects if camera_id in obj.visible_bboxes_2d]
            for obj in visible:
                class_id = CLASS_MAPPING.get(obj.object_type, -1)
                matches = {
                    variant: _best_match(obj, camera_id, class_id, local_rows, min_iou)
                    for variant, local_rows in local_by_variant.items()
                }
                selected = matches.get("bytetrack_21c_best")
                local_track_id = None if selected is None else safe_int(selected.get("local_track_id"), -1)
                stage_flags = _downstream_flags(obj.frame_id, local_track_id, downstream)
                values = {
                    "matched_in_v2_current_local": matches.get("v2_current") is not None,
                    "matched_in_bytetrack_21b_local": matches.get("bytetrack_21b") is not None,
                    "matched_in_bytetrack_21c_local": selected is not None,
                    "present_in_tracklets": stage_flags["present_in_tracklets"],
                    "present_in_candidates": stage_flags["present_in_candidates"],
                    "present_in_motion_clean": stage_flags["present_in_motion_clean"],
                    "present_in_global": stage_flags["present_in_global"],
                    "present_in_final_export": stage_flags["present_in_final_export"],
                }
                bucket = _duration_bucket(durations.get(obj.object_id, 1), config)
                _accumulate(trace_counts, "all", values)
                _accumulate(bucket_counts, bucket, values)
                _accumulate(scene_counts, scene_name, values)
                _accumulate(class_counts, obj.object_type, values)
                _accumulate(camera_counts, camera_id, values)
                drop_stage = _drop_stage(values)
                if drop_stage and len(drop_samples) < sample_limit * 6:
                    bbox = obj.visible_bboxes_2d.get(camera_id)
                    drop_samples.append(
                        {
                            "gt_scene": scene_name,
                            "gt_camera": camera_id,
                            "gt_frame": obj.frame_id,
                            "gt_class_id": class_id,
                            "gt_class_name": obj.object_type,
                            "gt_object_id": obj.object_id,
                            "bbox": json.dumps(list(bbox) if bbox is not None else []),
                            "gt_duration_frames": durations.get(obj.object_id, 1),
                            "duration_bucket": bucket,
                            "local_track_id": "" if local_track_id is None else local_track_id,
                            "drop_stage": drop_stage,
                            "drop_reason_proxy": _drop_reason(values),
                            **values,
                        }
                    )
    summary_rows = _summary_rows(trace_counts, "scope")
    bucket_rows = _summary_rows(bucket_counts, "duration_bucket")
    scene_rows = _summary_rows(scene_counts, "scene_name")
    class_rows = _summary_rows(class_counts, "class_name")
    camera_rows = _summary_rows(camera_counts, "camera_id")
    root = output_root(config) / "gt_matched_audit"
    write_csv(root / "gt_object_frame_retention_summary.csv", summary_rows)
    write_csv(root / "gt_object_frame_retention_by_duration_bucket.csv", bucket_rows)
    write_csv(root / "gt_object_frame_retention_by_scene.csv", scene_rows)
    write_csv(root / "gt_object_frame_retention_by_class.csv", class_rows)
    write_csv(root / "gt_object_frame_retention_by_camera.csv", camera_rows)
    write_csv(root / "gt_object_frame_drop_samples.csv", drop_samples)
    write_json(root / "gt_match_warnings.json", {"warnings": warnings})
    return {
        "status": "ok",
        "summary_rows": summary_rows,
        "duration_rows": bucket_rows,
        "scene_rows": scene_rows,
        "class_rows": class_rows,
        "camera_rows": camera_rows,
        "drop_samples": drop_samples,
        "warnings": warnings,
    }


def _load_local_rows(root: Path, subset: str, scene: str, camera: str) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    path = root / subset / scene / (camera + ".csv")
    output = {}
    for row in iter_csv(path):
        key = (safe_int(row.get("frame_id"), -1), safe_int(row.get("class_id"), -1))
        output.setdefault(key, []).append(row)
    return output


def _best_match(
    obj: GroundTruthObject,
    camera_id: str,
    class_id: int,
    local_rows: Dict[Tuple[int, int], List[Dict[str, Any]]],
    min_iou: float,
) -> Optional[Dict[str, Any]]:
    bbox = obj.visible_bboxes_2d.get(camera_id)
    if bbox is None:
        return None
    best = None
    best_iou = 0.0
    for row in local_rows.get((obj.frame_id, class_id), []):
        try:
            candidate = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        except (KeyError, TypeError, ValueError):
            continue
        iou = bbox_iou_xyxy(bbox, candidate)
        if iou > best_iou:
            best_iou = iou
            best = row
    return best if best_iou >= min_iou else None


def _load_downstream_indices(paths: Dict[str, Path], subset: str, scene: str, camera: str) -> Dict[str, Any]:
    tracklet_ids = _local_ids(paths.get("tracklets_root", Path("")), subset, scene, camera, "*_tracklets")
    candidate_ids = _local_ids(paths.get("candidates_root", Path("")), subset, scene, camera, "*_candidates")
    motion_ids = _local_ids(paths.get("motion_clean_root", Path("")), subset, scene, camera, "*_clean_candidates")
    global_ids = set()
    global_path = paths.get("global_root", Path("")) / subset / scene / "candidates_with_global_ids.jsonl"
    if not global_path.exists():
        global_path = global_path.with_suffix(".csv")
    iterator = iter_jsonl(global_path) if global_path.suffix == ".jsonl" else iter_csv(global_path)
    for row in iterator:
        if str(row.get("camera_id", "")) == camera and row.get("global_track_id") not in (None, ""):
            global_ids.add(safe_int(row.get("local_track_id"), -1))
    final_keys = set()
    frame_root = paths.get("final_export_root", Path("")) / "frame_global_records" / subset / scene
    final_path = frame_root / (camera + ".csv")
    for row in iter_csv(final_path):
        if row.get("global_track_id") not in (None, ""):
            final_keys.add((safe_int(row.get("frame_id"), -1), safe_int(row.get("local_track_id"), -1)))
    return {
        "tracklet_ids": tracklet_ids,
        "candidate_ids": candidate_ids,
        "motion_ids": motion_ids,
        "global_ids": global_ids,
        "final_keys": final_keys,
    }


def _local_ids(root: Path, subset: str, scene: str, camera: str, stem: str) -> set:
    base = root / subset / scene
    jsonl = base / ((stem.replace("*", camera)) + ".jsonl")
    path = jsonl if jsonl.exists() else jsonl.with_suffix(".csv")
    iterator = iter_jsonl(path) if path.suffix == ".jsonl" else iter_csv(path)
    return set(safe_int(row.get("local_track_id"), -1) for row in iterator)


def _downstream_flags(frame_id: int, local_track_id: Optional[int], values: Dict[str, Any]) -> Dict[str, bool]:
    if local_track_id is None or local_track_id < 0:
        return {key: False for key in [
            "present_in_tracklets", "present_in_candidates", "present_in_motion_clean",
            "present_in_global", "present_in_final_export",
        ]}
    return {
        "present_in_tracklets": local_track_id in values["tracklet_ids"],
        "present_in_candidates": local_track_id in values["candidate_ids"],
        "present_in_motion_clean": local_track_id in values["motion_ids"],
        "present_in_global": local_track_id in values["global_ids"],
        "present_in_final_export": (frame_id, local_track_id) in values["final_keys"],
    }


def _object_durations(objects: List[GroundTruthObject]) -> Dict[int, int]:
    frames = {}
    for obj in objects:
        frames.setdefault(obj.object_id, set()).add(obj.frame_id)
    return {object_id: len(values) for object_id, values in frames.items()}


def _duration_bucket(duration: int, config: Dict[str, Any]) -> str:
    values = config.get("gt_audit", {}).get("duration_buckets", {})
    short_max = int(values.get("short_max_frames", 10))
    medium_max = int(values.get("medium_max_frames", 50))
    if duration < short_max:
        return "short"
    if duration <= medium_max:
        return "medium"
    return "long"


def _accumulate(output: Dict[str, Any], key: str, values: Dict[str, bool]) -> None:
    target = output.setdefault(key, {"total": 0, "counts": {}})
    target["total"] += 1
    for field, value in values.items():
        target["counts"][field] = target["counts"].get(field, 0) + int(bool(value))


def _summary_rows(values: Dict[str, Any], key_name: str) -> List[Dict[str, Any]]:
    output = []
    for key, payload in sorted(values.items()):
        total = int(payload.get("total", 0) or 0)
        for stage, count in sorted(payload.get("counts", {}).items()):
            output.append(
                {
                    key_name: key,
                    "stage": stage,
                    "gt_object_frames": total,
                    "retained_gt_object_frames": count,
                    "gt_object_frame_retention": None if total <= 0 else float(count) / float(total),
                }
            )
    return output


def _drop_stage(values: Dict[str, bool]) -> str:
    order = [
        "matched_in_bytetrack_21c_local", "present_in_tracklets", "present_in_candidates",
        "present_in_motion_clean", "present_in_global", "present_in_final_export",
    ]
    labels = ["local_export", "tracklet_builder", "candidate_builder", "motion_filter", "global_association", "final_export"]
    for field, label in zip(order, labels):
        if not values.get(field, False):
            return label
    return ""


def _drop_reason(values: Dict[str, bool]) -> str:
    stage = _drop_stage(values)
    return {
        "local_export": "no_same_class_iou_match_in_bytetrack_local",
        "tracklet_builder": "matched_local_track_missing_from_tracklets",
        "candidate_builder": "tracklet_missing_from_candidates",
        "motion_filter": "candidate_not_motion_clean",
        "global_association": "motion_clean_candidate_has_no_global_id",
        "final_export": "global_candidate_frame_missing_from_final_export",
    }.get(stage, "")

