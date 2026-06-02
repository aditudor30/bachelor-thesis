import numpy as np

from deep_oc_sort_3d.mtmc.candidate_builder import MTMCCandidateBuilder
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


def _tracklet(length=6, quality_flag="good", is_valid=True):
    trajectory_3d = [(0, 0.0, 0.0, 0.0), (5, 1.0, 0.0, 0.0), (10, 2.0, 0.0, 0.0)]
    trajectory_2d = [(0, 1.0, 2.0, 11.0, 22.0), (5, 2.0, 2.0, 12.0, 22.0), (10, 3.0, 2.0, 13.0, 22.0)]
    return LocalTracklet(
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        camera_id="Camera_0000",
        local_track_id=7,
        class_id=0,
        class_name="Person",
        start_frame=0,
        end_frame=10,
        length=length,
        frame_ids=[0, 5, 10],
        detection_ids=[1, 2, 3],
        mean_confidence=0.8,
        median_confidence=0.8,
        max_confidence=0.9,
        bbox_start=(1.0, 2.0, 11.0, 22.0),
        bbox_end=(3.0, 2.0, 13.0, 22.0),
        bbox_mean=(2.0, 2.0, 12.0, 22.0),
        center_3d_start=np.asarray([0.0, 0.0, 0.0], dtype=float),
        center_3d_end=np.asarray([2.0, 0.0, 0.0], dtype=float),
        center_3d_mean=np.asarray([1.0, 0.0, 0.0], dtype=float),
        center_3d_median=np.asarray([1.0, 0.0, 0.0], dtype=float),
        dimensions_3d_mean=np.asarray([0.6, 0.8, 1.8], dtype=float),
        yaw_mean=0.0,
        trajectory_2d=trajectory_2d,
        trajectory_3d=trajectory_3d,
        majority_gt_object_id=42,
        gt_purity=1.0,
        num_gt_ids=1,
        gt_id_counts={"42": length},
        quality_score=0.9,
        quality_flag=quality_flag,
        is_valid_for_mtmc=is_valid,
        notes="",
    )


def test_candidate_builder_geometry_and_sampling():
    builder = MTMCCandidateBuilder(trajectory_sample_rate=2, max_trajectory_points=3)

    candidate = builder.build_from_tracklet(_tracklet(), subset="internal_holdout")

    assert candidate.is_candidate
    assert candidate.reject_reason == "ok"
    assert candidate.entry_frame == 0
    assert candidate.exit_frame == 10
    np.testing.assert_allclose(candidate.entry_center_3d, np.asarray([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(candidate.exit_center_3d, np.asarray([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(candidate.mean_velocity_3d, np.asarray([0.2, 0.0, 0.0]))
    assert candidate.travel_distance_3d == 2.0
    assert len(candidate.trajectory_3d_sampled) <= 3


def test_candidate_builder_keeps_rejected_records_as_non_candidates():
    builder = MTMCCandidateBuilder(min_length=3)

    candidate = builder.build_from_tracklet(_tracklet(length=1), subset="official_val")

    assert not candidate.is_candidate
    assert candidate.reject_reason == "too_short"
