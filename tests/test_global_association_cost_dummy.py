"""Dummy tests for global association costs."""

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_association_cost import compute_global_association_cost


def make_candidate(candidate_id, camera_id, class_id=0, class_name="Person", offset=0.0):
    return MTMCTrackletCandidate(
        candidate_id=candidate_id,
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        subset="official_val",
        camera_id=camera_id,
        local_track_id=1,
        class_id=class_id,
        class_name=class_name,
        start_frame=0,
        end_frame=4,
        length=5,
        duration=5,
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
        center_3d_start=np.asarray([offset, 0.0, 0.0], dtype=float),
        center_3d_end=np.asarray([offset + 0.4, 0.0, 0.0], dtype=float),
        center_3d_mean=np.asarray([offset + 0.2, 0.0, 0.0], dtype=float),
        center_3d_median=np.asarray([offset + 0.2, 0.0, 0.0], dtype=float),
        trajectory_2d_sampled=[],
        trajectory_3d_sampled=[
            (0, offset, 0.0, 0.0),
            (1, offset + 0.1, 0.0, 0.0),
            (2, offset + 0.2, 0.0, 0.0),
        ],
        trajectory_3d_length=3,
        has_3d=True,
        entry_frame=0,
        exit_frame=4,
        entry_center_3d=np.asarray([offset, 0.0, 0.0], dtype=float),
        exit_center_3d=np.asarray([offset + 0.4, 0.0, 0.0], dtype=float),
        mean_velocity_3d=np.asarray([1.0, 0.0, 0.0], dtype=float),
        travel_distance_3d=0.4,
        majority_gt_object_id=1,
        gt_purity=1.0,
        num_gt_ids=1,
        gt_id_counts={"1": 5},
        reid_embedding_path=None,
        reid_embedding=None,
        global_track_id=None,
    )


def test_close_overlap_candidates_are_accepted():
    a = make_candidate("a", "Camera_0000", offset=0.0)
    b = make_candidate("b", "Camera_0001", offset=0.1)
    edge = compute_global_association_cost(a, b, {})
    assert edge.accepted
    assert edge.reject_reason == "ok"
    assert edge.mean_3d_distance is not None


def test_class_same_camera_and_far_rejections():
    a = make_candidate("a", "Camera_0000", class_id=0, class_name="Person", offset=0.0)
    different_class = make_candidate("b", "Camera_0001", class_id=1, class_name="Forklift", offset=0.0)
    same_camera = make_candidate("c", "Camera_0000", class_id=0, class_name="Person", offset=0.0)
    far = make_candidate("d", "Camera_0002", class_id=0, class_name="Person", offset=100.0)
    assert compute_global_association_cost(a, different_class, {}).reject_reason == "class_mismatch"
    assert compute_global_association_cost(a, same_camera, {}).reject_reason == "same_camera_not_allowed"
    assert compute_global_association_cost(a, far, {}).reject_reason == "overlap_distance_too_large"
