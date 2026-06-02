import numpy as np

from deep_oc_sort_3d.tracklets.tracklet_eval import evaluate_tracklets
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


def test_evaluate_tracklets_reports_metrics():
    first = _tracklet()
    second = _tracklet()
    second.local_track_id = 6
    second.gt_purity = 0.5
    second.num_gt_ids = 2
    second.quality_flag = "short"
    second.is_valid_for_mtmc = False

    metrics = evaluate_tracklets([first, second])

    assert metrics["num_tracklets"] == 2
    assert metrics["valid_for_mtmc"] == 1
    assert metrics["mixed_gt_tracklets"] == 1
    assert metrics["short_tracklets"] == 1
    assert metrics["purity_mean"] == 0.75
