"""Dummy tests for final export global-id propagation."""

import numpy as np

from deep_oc_sort_3d.final_export.global_id_propagation import (
    load_candidate_global_id_mapping,
    namespace_global_track_id,
    propagate_global_ids_to_local_records,
)
from deep_oc_sort_3d.mtmc.candidate_io import write_candidates_jsonl
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord


def make_candidate(global_track_id=7):
    return MTMCTrackletCandidate(
        candidate_id="Warehouse_020__Camera_0000__track_1",
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        subset="official_val",
        camera_id="Camera_0000",
        local_track_id=1,
        class_id=0,
        class_name="Person",
        start_frame=0,
        end_frame=10,
        length=11,
        duration=11,
        mean_confidence=0.8,
        median_confidence=0.8,
        max_confidence=0.9,
        quality_score=1.0,
        quality_flag="good",
        source_tracklet_valid_for_mtmc=True,
        is_candidate=True,
        reject_reason=None,
        bbox_start=None,
        bbox_end=None,
        bbox_mean=None,
        center_3d_start=None,
        center_3d_end=None,
        center_3d_mean=np.asarray([1.0, 2.0, 3.0], dtype=float),
        center_3d_median=None,
        trajectory_2d_sampled=[],
        trajectory_3d_sampled=[],
        trajectory_3d_length=0,
        has_3d=True,
        entry_frame=0,
        exit_frame=10,
        entry_center_3d=None,
        exit_center_3d=None,
        mean_velocity_3d=None,
        travel_distance_3d=None,
        majority_gt_object_id=5,
        gt_purity=1.0,
        num_gt_ids=1,
        gt_id_counts={"5": 11},
        reid_embedding_path=None,
        reid_embedding=None,
        global_track_id=global_track_id,
    )


def make_local_record(local_track_id):
    return LocalTrackRecord(
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        camera_id="Camera_0000",
        frame_id=0,
        local_track_id=local_track_id,
        detection_id=3,
        class_id=0,
        class_name="Person",
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 11.0, 22.0),
        bbox_xywh=(1.0, 2.0, 10.0, 20.0),
        center_3d=np.asarray([1.0, 2.0, 3.0], dtype=float),
        dimensions_3d=np.asarray([0.5, 1.0, 1.8], dtype=float),
        yaw=0.1,
        matched_gt_object_id=5,
        matched_gt=True,
        track_age=1,
        track_hits=1,
        track_misses=0,
        track_state="confirmed",
    )


def test_load_mapping_and_propagate_include_drop_unassigned(tmp_path):
    candidates_path = tmp_path / "candidates_with_global_ids.jsonl"
    write_candidates_jsonl([make_candidate()], candidates_path)
    mapping = load_candidate_global_id_mapping(candidates_path)
    records = [make_local_record(1), make_local_record(2)]
    included = propagate_global_ids_to_local_records(records, "official_val", mapping, include_unassigned=True)
    dropped = propagate_global_ids_to_local_records(records, "official_val", mapping, include_unassigned=False)
    assert len(included) == 2
    assert len(dropped) == 1
    assert included[0].global_track_id == 7
    assert included[1].global_track_id is None


def test_namespace_global_track_id_separates_scenes_and_subsets():
    a = namespace_global_track_id("official_val", "Warehouse_020", 7)
    b = namespace_global_track_id("internal_holdout", "Warehouse_020", 7)
    c = namespace_global_track_id("official_val", "Warehouse_021", 7)
    assert a != b
    assert a != c
