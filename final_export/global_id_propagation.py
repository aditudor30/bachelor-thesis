"""Propagate global track ids back to frame-level local track records."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from deep_oc_sort_3d.final_export.global_frame_types import GlobalFrameRecord
from deep_oc_sort_3d.final_export.generic_export import (
    is_valid_bbox,
    write_global_frame_records_csv,
    write_global_frame_records_jsonl,
)
from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.data.dataset_structure import scene_name_to_id
from deep_oc_sort_3d.tracking.track_io import read_local_tracks_csv
from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord


CandidateMapping = Dict[Tuple[Any, ...], Tuple[str, int]]


def load_candidate_global_id_mapping(
    candidates_with_global_ids_path: Union[str, Path],
    namespace_global_ids: bool = False,
    global_id_stride: int = 100000,
) -> CandidateMapping:
    """Load mapping from local track identity to candidate/global ids."""
    path = Path(candidates_with_global_ids_path)
    candidates = read_candidates_file(path)
    mapping = {}
    ambiguous_legacy_keys = set()
    for candidate in candidates:
        if candidate.global_track_id is None:
            continue
        global_track_id = int(candidate.global_track_id)
        if namespace_global_ids:
            global_track_id = namespace_global_track_id(
                candidate.subset,
                candidate.scene_name,
                global_track_id,
                global_id_stride=global_id_stride,
            )
        class_key = (
            str(candidate.subset),
            str(candidate.scene_name),
            str(candidate.camera_id),
            int(candidate.local_track_id),
            int(candidate.class_id),
        )
        legacy_key = (
            str(candidate.subset),
            str(candidate.scene_name),
            str(candidate.camera_id),
            int(candidate.local_track_id),
        )
        value = (str(candidate.candidate_id), int(global_track_id))
        mapping[class_key] = value
        if legacy_key in mapping and mapping[legacy_key] != value:
            ambiguous_legacy_keys.add(legacy_key)
        else:
            mapping[legacy_key] = value
    for key in ambiguous_legacy_keys:
        mapping.pop(key, None)
    return mapping


def propagate_global_ids_to_local_records(
    local_records: List[LocalTrackRecord],
    subset: str,
    candidate_mapping: CandidateMapping,
    include_unassigned: bool = True,
    drop_invalid_bbox: bool = False,
) -> List[GlobalFrameRecord]:
    """Propagate global ids from candidate mapping to local track records."""
    output = []
    for record in local_records:
        if drop_invalid_bbox and not is_valid_bbox(record.bbox_xyxy):
            continue
        class_key = (
            str(subset),
            str(record.scene_name),
            str(record.camera_id),
            int(record.local_track_id),
            int(record.class_id),
        )
        legacy_key = (str(subset), str(record.scene_name), str(record.camera_id), int(record.local_track_id))
        value = candidate_mapping.get(class_key)
        candidate_id = None
        global_track_id = None
        if value is not None:
            candidate_id, global_track_id = value
        elif not include_unassigned:
            continue
        output.append(
            GlobalFrameRecord(
                scene_id=int(record.scene_id),
                scene_name=str(record.scene_name),
                split=str(record.split),
                subset=str(subset),
                camera_id=str(record.camera_id),
                frame_id=int(record.frame_id),
                global_track_id=global_track_id,
                local_track_id=int(record.local_track_id),
                candidate_id=candidate_id,
                detection_id=int(record.detection_id),
                class_id=int(record.class_id),
                class_name=str(record.class_name),
                confidence=float(record.confidence),
                bbox_xyxy=record.bbox_xyxy,
                bbox_xywh=record.bbox_xywh,
                center_3d=record.center_3d,
                dimensions_3d=record.dimensions_3d,
                yaw=record.yaw,
                matched_gt_object_id=record.matched_gt_object_id,
                matched_gt=bool(record.matched_gt),
                source="local_track_record",
            )
        )
    return sorted(output, key=lambda item: (item.scene_name, item.camera_id, item.frame_id, _sort_global_id(item), item.detection_id))


def propagate_for_camera_file(
    local_tracks_csv: Path,
    candidates_with_global_ids_path: Path,
    subset: str,
    output_csv: Path,
    output_jsonl: Optional[Path],
    include_unassigned: bool,
    show_progress: bool = True,
    namespace_global_ids: bool = True,
    global_id_stride: int = 100000,
    drop_invalid_bbox: bool = False,
) -> Dict[str, Any]:
    """Propagate global ids for one camera local-track file."""
    try:
        mapping = load_candidate_global_id_mapping(
            candidates_with_global_ids_path,
            namespace_global_ids=namespace_global_ids,
            global_id_stride=global_id_stride,
        )
        local_records = read_local_tracks_csv(local_tracks_csv)
        records = propagate_global_ids_to_local_records(
            local_records,
            subset=subset,
            candidate_mapping=mapping,
            include_unassigned=include_unassigned,
            drop_invalid_bbox=drop_invalid_bbox,
        )
        write_global_frame_records_csv(records, output_csv)
        if output_jsonl is not None:
            write_global_frame_records_jsonl(records, output_jsonl)
        assigned = [record for record in records if record.global_track_id is not None]
        unassigned = [record for record in records if record.global_track_id is None]
        scene_name = local_records[0].scene_name if local_records else _scene_from_path(local_tracks_csv)
        camera_id = local_records[0].camera_id if local_records else local_tracks_csv.stem
        return {
            "subset": subset,
            "scene_name": scene_name,
            "camera_id": camera_id,
            "input_records": len(local_records),
            "output_records": len(records),
            "assigned_records": len(assigned),
            "unassigned_records": len(unassigned),
            "unique_local_tracks": len(set([record.local_track_id for record in local_records])),
            "unique_global_tracks": len(set([record.global_track_id for record in assigned])),
            "namespace_global_ids": bool(namespace_global_ids),
            "global_id_stride": int(global_id_stride),
            "drop_invalid_bbox": bool(drop_invalid_bbox),
            "output_csv": str(output_csv),
            "status": "ok",
            "error_message": "",
        }
    except Exception as exc:
        return {
            "subset": subset,
            "scene_name": _scene_from_path(local_tracks_csv),
            "camera_id": local_tracks_csv.stem,
            "input_records": 0,
            "output_records": 0,
            "assigned_records": 0,
            "unassigned_records": 0,
            "unique_local_tracks": 0,
            "unique_global_tracks": 0,
            "namespace_global_ids": bool(namespace_global_ids),
            "global_id_stride": int(global_id_stride),
            "drop_invalid_bbox": bool(drop_invalid_bbox),
            "output_csv": str(output_csv),
            "status": "error",
            "error_message": str(exc),
        }


def _sort_global_id(record: GlobalFrameRecord) -> int:
    if record.global_track_id is None:
        return 10**12
    return int(record.global_track_id)


def namespace_global_track_id(
    subset: str,
    scene_name: str,
    global_track_id: int,
    global_id_stride: int = 100000,
) -> int:
    """Return deterministic export-global id for a scene-local global track id."""
    scene_id = scene_name_to_id(str(scene_name))
    if scene_id is None:
        scene_id = _stable_small_hash(str(scene_name))
    subset_offset = _subset_offset(str(subset), int(global_id_stride))
    return int(subset_offset + int(scene_id) * int(global_id_stride) + int(global_track_id))


def _subset_offset(subset: str, global_id_stride: int) -> int:
    offsets = {
        "train": 0,
        "internal_holdout": 1000,
        "official_val": 2000,
        "val": 2000,
        "test": 3000,
    }
    base = offsets.get(subset, 4000 + _stable_small_hash(subset))
    return int(base) * int(global_id_stride)


def _stable_small_hash(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 131 + ord(char)) % 100000
    return total


def _scene_from_path(path: Path) -> str:
    parts = list(path.parts)
    if len(parts) >= 2:
        return parts[-2]
    return ""
