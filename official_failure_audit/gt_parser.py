"""Ground-truth structure audit and normalized Track1-like conversion."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from deep_oc_sort_3d.data.ground_truth import load_ground_truth_json
from deep_oc_sort_3d.official_failure_audit.failure_audit_config import dataset_root, official_class_names, scene_id, val_scenes
from deep_oc_sort_3d.official_failure_audit.failure_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row


def parse_val_ground_truth(
    config: Dict[str, Any], output_root: Path, progress: bool = True,
) -> Tuple[List[AuditTrack1Row], Dict[str, Any]]:
    rows: List[AuditTrack1Row] = []
    field_counts: Dict[str, int] = defaultdict(int)
    camera_ids = set()
    raw_examples: Dict[str, Any] = {}
    scene_summaries: List[Dict[str, Any]] = []
    calibration_summaries: List[Dict[str, Any]] = []
    name_to_official = official_class_names(config)
    for scene in progress_iter(val_scenes(config), progress, "23A GT scenes"):
        path = dataset_root(config) / "val" / scene / "ground_truth.json"
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if isinstance(raw, dict):
            for frame_objects in raw.values():
                if not isinstance(frame_objects, list):
                    continue
                for item in frame_objects:
                    if not isinstance(item, dict):
                        continue
                    for key in item.keys():
                        field_counts[str(key)] += 1
                    if not raw_examples:
                        raw_examples = dict(item)
        objects = load_ground_truth_json(path) if path.is_file() else []
        for item in objects:
            camera_ids.update(item.visible_bboxes_2d.keys())
            class_id = name_to_official.get(item.object_type)
            if class_id is None:
                continue
            rows.append(AuditTrack1Row(
                scene_id=scene_id(scene), class_id=class_id, object_id=int(item.object_id),
                frame_id=int(item.frame_id), x=float(item.location_3d[0]), y=float(item.location_3d[1]),
                z=float(item.location_3d[2]), width=float(item.bbox3d_scale[0]),
                length=float(item.bbox3d_scale[1]), height=float(item.bbox3d_scale[2]),
                yaw=float(item.bbox3d_rotation[-1]), raw_class_id=class_id,
                source_class_space="official", source_path=str(path), source_kind="ground_truth",
            ))
        scene_rows = [row for row in rows if row.scene_id == scene_id(scene)]
        scene_summaries.append(_frame_summary(scene, scene_rows, path))
        calibration_summaries.append(_calibration_summary(dataset_root(config) / "val" / scene / "calibration.json", scene))

    summary = _structure_summary(
        rows, field_counts, camera_ids, raw_examples, scene_summaries, calibration_summaries,
    )
    audit_root = output_root / "gt_audit"
    write_json(audit_root / "gt_structure_summary.json", summary)
    write_csv(audit_root / "gt_range_summary.csv", _range_rows(rows), ["field", "min", "max", "mean", "median"])
    write_csv(audit_root / "gt_class_distribution.csv", _class_rows(rows), ["class_id", "class_name", "rows", "tracks"])
    write_csv(audit_root / "gt_frame_summary.csv", scene_summaries)
    return rows, summary


def _structure_summary(
    rows: Sequence[AuditTrack1Row], field_counts: Dict[str, int], camera_ids: Set[str],
    raw_example: Dict[str, Any], scene_summaries: Sequence[Dict[str, Any]],
    calibration_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    frames = [row.frame_id for row in rows]
    return {
        "status": "ok" if rows else "missing_or_unparsed_gt",
        "rows": len(rows), "tracks": len(set((row.scene_id, row.class_id, row.object_id) for row in rows)),
        "raw_object_fields": dict(sorted(field_counts.items())), "raw_object_example": raw_example,
        "normalized_center_fields": ["3d_location", "3d location"],
        "normalized_dimension_fields": ["3d_bounding_box_scale", "3d bounding box scale"],
        "normalized_yaw_field": "last value of 3d_bounding_box_rotation",
        "has_yaw": bool(rows), "has_camera_id": bool(camera_ids), "camera_ids": sorted(camera_ids),
        "has_bbox2d": bool(camera_ids), "has_object_id": bool(rows),
        "frame_min": min(frames) if frames else None, "frame_max": max(frames) if frames else None,
        "frame_indexing_inference": _frame_indexing(frames), "scenes": list(scene_summaries),
        "calibration_scenes": list(calibration_summaries),
    }


def _range_rows(rows: Sequence[AuditTrack1Row]) -> List[Dict[str, Any]]:
    fields = ["x", "y", "z", "width", "length", "height", "yaw"]
    output = []
    for field in fields:
        values = np.asarray([float(getattr(row, field)) for row in rows], dtype=float)
        output.append({
            "field": field, "min": float(np.min(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
        })
    return output


def _class_rows(rows: Sequence[AuditTrack1Row]) -> List[Dict[str, Any]]:
    names = {0: "Person", 1: "Forklift", 2: "NovaCarter", 3: "Transporter", 4: "FourierGR1T2", 5: "AgilityDigit", 6: "PalletTruck"}
    output = []
    for class_id in sorted(set(row.class_id for row in rows)):
        selected = [row for row in rows if row.class_id == class_id]
        output.append({
            "class_id": class_id, "class_name": names.get(class_id, "unknown"), "rows": len(selected),
            "tracks": len(set((row.scene_id, row.object_id) for row in selected)),
        })
    return output


def _frame_summary(scene: str, rows: Sequence[AuditTrack1Row], path: Path) -> Dict[str, Any]:
    frames = [row.frame_id for row in rows]
    return {
        "scene_name": scene, "scene_id": scene_id(scene), "path": str(path), "exists": path.is_file(),
        "rows": len(rows), "unique_frames": len(set(frames)), "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None, "frame_indexing_inference": _frame_indexing(frames),
    }


def _frame_indexing(frames: Sequence[int]) -> str:
    if not frames:
        return "not_available"
    if min(frames) == 0:
        return "likely_zero_based"
    if min(frames) == 1:
        return "likely_one_based"
    return "unknown_nonzero_start"


def _calibration_summary(path: Path, scene: str) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    cameras = raw if isinstance(raw, dict) else {}
    field_names = set()
    for value in cameras.values():
        if isinstance(value, dict):
            field_names.update(str(key) for key in value.keys())
    return {
        "scene_name": scene, "path": str(path), "exists": path.is_file(),
        "camera_ids": sorted(str(key) for key in cameras.keys()),
        "camera_count": len(cameras), "camera_calibration_fields": sorted(field_names),
    }
