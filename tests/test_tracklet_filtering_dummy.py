import numpy as np

from deep_oc_sort_3d.tracklets.tracklet_filtering import (
    classify_tracklet_quality,
    compute_tracklet_quality_score,
)
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


def _tracklet():
    return LocalTracklet(
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        camera_id="Camera_0000",
        local_track_id=5,
        class_id=0,
        class_name="Person",
        start_frame=0,
        end_frame=2,
        length=3,
        frame_ids=[0, 1, 2],
        detection_ids=[10, 11, 12],
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
        yaw_mean=0.1,
        trajectory_2d=[(0, 1.0, 2.0, 11.0, 22.0), (1, 2.0, 2.0, 12.0, 22.0)],
        trajectory_3d=[(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)],
        majority_gt_object_id=42,
        gt_purity=1.0,
        num_gt_ids=1,
        gt_id_counts={"42": 3},
        quality_score=0.9,
        quality_flag="good",
        is_valid_for_mtmc=True,
        notes="",
    )


def test_quality_good_tracklet():
    tracklet = _tracklet()

    flag, is_valid, notes = classify_tracklet_quality(tracklet)

    assert flag == "good"
    assert is_valid
    assert notes == ""
    assert compute_tracklet_quality_score(tracklet) > 0.0


def test_quality_short_tracklet():
    tracklet = _tracklet()
    tracklet.length = 1

    flag, is_valid, _notes = classify_tracklet_quality(tracklet, min_length=3)

    assert flag == "short"
    assert not is_valid


def test_quality_low_confidence_tracklet():
    tracklet = _tracklet()
    tracklet.mean_confidence = 0.001

    flag, is_valid, _notes = classify_tracklet_quality(tracklet, min_mean_confidence=0.01)

    assert flag == "low_confidence"
    assert not is_valid


def test_quality_no_3d_tracklet_is_valid_but_flagged():
    tracklet = _tracklet()
    tracklet.trajectory_3d = []

    flag, is_valid, notes = classify_tracklet_quality(tracklet)

    assert flag == "no_3d"
    assert is_valid
    assert "missing 3d trajectory" in notes
