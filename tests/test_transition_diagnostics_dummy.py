"""Dummy tests for MTMC transition diagnostics."""

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.transition_diagnostics import (
    build_transition_candidate_pairs,
    compute_transition_pair_metrics,
)


def make_transition_candidate(
    candidate_id,
    camera_id,
    start_frame,
    end_frame,
    entry_x,
    exit_x,
    gt_id=1,
):
    return MTMCTrackletCandidate(
        candidate_id=candidate_id,
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        subset="official_val",
        camera_id=camera_id,
        local_track_id=1,
        class_id=0,
        class_name="Person",
        start_frame=start_frame,
        end_frame=end_frame,
        length=end_frame - start_frame + 1,
        duration=end_frame - start_frame + 1,
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
        center_3d_start=np.asarray([entry_x, 0.0, 0.0], dtype=float),
        center_3d_end=np.asarray([exit_x, 0.0, 0.0], dtype=float),
        center_3d_mean=np.asarray([(entry_x + exit_x) * 0.5, 0.0, 0.0], dtype=float),
        center_3d_median=np.asarray([(entry_x + exit_x) * 0.5, 0.0, 0.0], dtype=float),
        trajectory_2d_sampled=[],
        trajectory_3d_sampled=[(start_frame, entry_x, 0.0, 0.0), (end_frame, exit_x, 0.0, 0.0)],
        trajectory_3d_length=2,
        has_3d=True,
        entry_frame=start_frame,
        exit_frame=end_frame,
        entry_center_3d=np.asarray([entry_x, 0.0, 0.0], dtype=float),
        exit_center_3d=np.asarray([exit_x, 0.0, 0.0], dtype=float),
        mean_velocity_3d=np.asarray([0.1, 0.0, 0.0], dtype=float),
        travel_distance_3d=abs(exit_x - entry_x),
        majority_gt_object_id=gt_id,
        gt_purity=1.0 if gt_id is not None else None,
        num_gt_ids=1 if gt_id is not None else 0,
        gt_id_counts={} if gt_id is None else {str(gt_id): end_frame - start_frame + 1},
        reid_embedding_path=None,
        reid_embedding=None,
        global_track_id=None,
    )


def test_transition_pair_metrics_and_true_label():
    a = make_transition_candidate("a", "Camera_0000", 0, 10, 0.0, 1.0, gt_id=5)
    b = make_transition_candidate("b", "Camera_0001", 20, 30, 1.2, 2.0, gt_id=5)
    pair = compute_transition_pair_metrics(a, b, {"max_entry_exit_distance": 5.0})
    assert pair.temporal_relation == "a_before_b"
    assert pair.temporal_gap == 10
    assert pair.entry_exit_distance is not None
    assert pair.diagnostic_label == "true_transition"


def test_build_transition_pairs_rejects_same_camera_by_blocking():
    a = make_transition_candidate("a", "Camera_0000", 0, 10, 0.0, 1.0, gt_id=5)
    b = make_transition_candidate("b", "Camera_0000", 20, 30, 1.2, 2.0, gt_id=5)
    pairs = build_transition_candidate_pairs([a, b], {}, show_progress=False)
    assert pairs == []


def test_transition_false_and_unknown_gt_labels():
    a = make_transition_candidate("a", "Camera_0000", 0, 10, 0.0, 1.0, gt_id=5)
    b = make_transition_candidate("b", "Camera_0001", 20, 30, 1.2, 2.0, gt_id=6)
    c = make_transition_candidate("c", "Camera_0002", 40, 50, 2.1, 3.0, gt_id=None)
    assert compute_transition_pair_metrics(a, b, {}).diagnostic_label == "false_transition"
    assert compute_transition_pair_metrics(b, c, {}).diagnostic_label == "unknown_gt"
