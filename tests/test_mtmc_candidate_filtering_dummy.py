import numpy as np

from deep_oc_sort_3d.mtmc.candidate_filtering import should_keep_tracklet
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


def _tracklet(length=6, quality_flag="good", is_valid=True):
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
        trajectory_2d=[(0, 1.0, 2.0, 11.0, 22.0)],
        trajectory_3d=[(0, 0.0, 0.0, 0.0), (10, 2.0, 0.0, 0.0)],
        majority_gt_object_id=42,
        gt_purity=1.0,
        num_gt_ids=1,
        gt_id_counts={"42": length},
        quality_score=0.9,
        quality_flag=quality_flag,
        is_valid_for_mtmc=is_valid,
        notes="",
    )


def test_filter_ok():
    keep, reason = should_keep_tracklet(_tracklet())
    assert keep
    assert reason == "ok"


def test_filter_too_short():
    keep, reason = should_keep_tracklet(_tracklet(length=1), min_length=3)
    assert not keep
    assert reason == "too_short"


def test_filter_low_confidence():
    tracklet = _tracklet()
    tracklet.mean_confidence = 0.001
    keep, reason = should_keep_tracklet(tracklet, min_mean_confidence=0.01)
    assert not keep
    assert reason == "low_confidence"


def test_filter_quality_flag_not_allowed():
    keep, reason = should_keep_tracklet(_tracklet(quality_flag="no_3d"))
    assert not keep
    assert reason == "quality_flag_not_allowed"


def test_filter_invalid_for_mtmc():
    keep, reason = should_keep_tracklet(_tracklet(is_valid=False), require_valid_for_mtmc=True)
    assert not keep
    assert reason == "invalid_for_mtmc"


def test_filter_missing_3d():
    tracklet = _tracklet()
    tracklet.trajectory_3d = []
    keep, reason = should_keep_tracklet(tracklet, require_3d=True)
    assert not keep
    assert reason == "missing_3d"
