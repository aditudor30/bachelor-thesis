import numpy as np

from deep_oc_sort_3d.mtmc.candidate_summary import summarize_candidates
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate, make_candidate_id


def _candidate():
    return MTMCTrackletCandidate(
        candidate_id=make_candidate_id("Warehouse_000", "Camera_0000", 7),
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        subset="internal_holdout",
        camera_id="Camera_0000",
        local_track_id=7,
        class_id=0,
        class_name="Person",
        start_frame=0,
        end_frame=10,
        length=6,
        duration=11,
        mean_confidence=0.8,
        median_confidence=0.8,
        max_confidence=0.9,
        quality_score=0.7,
        quality_flag="good",
        source_tracklet_valid_for_mtmc=True,
        is_candidate=True,
        reject_reason="ok",
        bbox_start=(1.0, 2.0, 11.0, 22.0),
        bbox_end=(3.0, 2.0, 13.0, 22.0),
        bbox_mean=(2.0, 2.0, 12.0, 22.0),
        center_3d_start=np.asarray([0.0, 0.0, 0.0], dtype=float),
        center_3d_end=np.asarray([2.0, 0.0, 0.0], dtype=float),
        center_3d_mean=np.asarray([1.0, 0.0, 0.0], dtype=float),
        center_3d_median=np.asarray([1.0, 0.0, 0.0], dtype=float),
        trajectory_2d_sampled=[(0, 1.0, 2.0, 11.0, 22.0)],
        trajectory_3d_sampled=[(0, 0.0, 0.0, 0.0), (10, 2.0, 0.0, 0.0)],
        trajectory_3d_length=2,
        has_3d=True,
        entry_frame=0,
        exit_frame=10,
        entry_center_3d=np.asarray([0.0, 0.0, 0.0], dtype=float),
        exit_center_3d=np.asarray([2.0, 0.0, 0.0], dtype=float),
        mean_velocity_3d=np.asarray([0.2, 0.0, 0.0], dtype=float),
        travel_distance_3d=2.0,
        majority_gt_object_id=42,
        gt_purity=1.0,
        num_gt_ids=1,
        gt_id_counts={"42": 6},
        reid_embedding_path=None,
        reid_embedding=None,
        global_track_id=None,
    )


def test_candidate_summary_counts_kept_rejected_and_3d():
    kept = _candidate()
    rejected = _candidate()
    rejected.candidate_id = "Warehouse_000__Camera_0000__track_8"
    rejected.local_track_id = 8
    rejected.is_candidate = False
    rejected.reject_reason = "too_short"
    rejected.has_3d = False
    rejected.trajectory_3d_length = 0

    summary = summarize_candidates([kept, rejected])

    assert summary["total_candidates_including_rejected"] == 2
    assert summary["kept_candidates"] == 1
    assert summary["rejected_candidates"] == 1
    assert summary["reject_reason_counts"]["too_short"] == 1
    assert summary["per_class_kept_counts"]["Person"] == 1
    assert summary["has_3d_count"] == 1
    assert summary["no_3d_count"] == 1
