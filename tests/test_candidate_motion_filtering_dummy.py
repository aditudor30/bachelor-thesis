import numpy as np

from deep_oc_sort_3d.mtmc.candidate_motion_filtering import split_candidates_by_motion_quality
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate, make_candidate_id


def _candidate(points):
    return MTMCTrackletCandidate(
        candidate_id=make_candidate_id("Warehouse_000", "Camera_0000", 1),
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        subset="internal_holdout",
        camera_id="Camera_0000",
        local_track_id=1,
        class_id=0,
        class_name="Person",
        start_frame=points[0][0],
        end_frame=points[-1][0],
        length=len(points),
        duration=points[-1][0] - points[0][0] + 1,
        mean_confidence=0.9,
        median_confidence=0.9,
        max_confidence=0.95,
        quality_score=0.8,
        quality_flag="good",
        source_tracklet_valid_for_mtmc=True,
        is_candidate=True,
        reject_reason="ok",
        bbox_start=None,
        bbox_end=None,
        bbox_mean=None,
        center_3d_start=np.asarray([points[0][1], points[0][2], points[0][3]], dtype=float),
        center_3d_end=np.asarray([points[-1][1], points[-1][2], points[-1][3]], dtype=float),
        center_3d_mean=None,
        center_3d_median=None,
        trajectory_2d_sampled=[],
        trajectory_3d_sampled=points,
        trajectory_3d_length=len(points),
        has_3d=True,
        entry_frame=points[0][0],
        exit_frame=points[-1][0],
        entry_center_3d=np.asarray([points[0][1], points[0][2], points[0][3]], dtype=float),
        exit_center_3d=np.asarray([points[-1][1], points[-1][2], points[-1][3]], dtype=float),
        mean_velocity_3d=None,
        travel_distance_3d=None,
        majority_gt_object_id=None,
        gt_purity=None,
        num_gt_ids=0,
        gt_id_counts={},
        reid_embedding_path=None,
        reid_embedding=None,
        global_track_id=None,
    )


def test_split_candidates_by_motion_quality():
    good = _candidate([(0, 0.0, 0.0, 0.0), (1, 0.5, 0.0, 0.0), (2, 1.0, 0.0, 0.0)])
    suspicious = _candidate([
        (0, 0.0, 0.0, 0.0),
        (1, 4.0, 0.0, 0.0),
        (2, 4.2, 0.0, 0.0),
        (3, 4.4, 0.0, 0.0),
        (4, 4.6, 0.0, 0.0),
        (5, 4.8, 0.0, 0.0),
        (6, 5.0, 0.0, 0.0),
        (7, 5.2, 0.0, 0.0),
        (8, 5.4, 0.0, 0.0),
        (9, 5.6, 0.0, 0.0),
        (10, 5.8, 0.0, 0.0),
    ])
    suspicious.candidate_id = "suspicious"
    invalid = _candidate([(0, 0.0, 0.0, 0.0), (1, 10.0, 0.0, 0.0), (2, 10.5, 0.0, 0.0)])
    invalid.candidate_id = "invalid"

    buckets = split_candidates_by_motion_quality([good, suspicious, invalid], {}, show_progress=False)

    assert len(buckets["clean"]) == 1
    assert len(buckets["suspicious"]) == 1
    assert len(buckets["invalid"]) == 1


def test_allow_suspicious_as_clean():
    suspicious = _candidate([
        (0, 0.0, 0.0, 0.0),
        (1, 4.0, 0.0, 0.0),
        (2, 4.2, 0.0, 0.0),
        (3, 4.4, 0.0, 0.0),
        (4, 4.6, 0.0, 0.0),
        (5, 4.8, 0.0, 0.0),
        (6, 5.0, 0.0, 0.0),
        (7, 5.2, 0.0, 0.0),
        (8, 5.4, 0.0, 0.0),
        (9, 5.6, 0.0, 0.0),
        (10, 5.8, 0.0, 0.0),
    ])

    buckets = split_candidates_by_motion_quality(
        [suspicious],
        {"allow_suspicious_as_clean": True},
        show_progress=False,
    )

    assert len(buckets["clean"]) == 1
    assert len(buckets["suspicious"]) == 0
