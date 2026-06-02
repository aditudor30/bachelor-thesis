import numpy as np

from deep_oc_sort_3d.observations.observation_types import Observation3D
from deep_oc_sort_3d.tracking.local_tracker import LocalObservationTracker


def _observation(frame_id, detection_id, center, object_id=10):
    return Observation3D(
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        camera_id="Camera_0000",
        frame_id=frame_id,
        detection_id=detection_id,
        class_id=0,
        class_name="Person",
        confidence=0.9,
        bbox_xyxy=(10.0 + frame_id, 20.0, 30.0 + frame_id, 60.0),
        bbox_xywh=(10.0 + frame_id, 20.0, 20.0, 40.0),
        center_3d=np.asarray(center, dtype=float),
        dimensions_3d=np.asarray([0.6, 0.8, 1.8], dtype=float),
        yaw=0.0,
        object_id=object_id,
        matched_gt=object_id is not None,
        matched_iou=0.9 if object_id is not None else None,
        depth_value=3.0,
        depth_sampling_method="center_median",
        source="dummy",
    )


def test_tracker_keeps_single_track_for_close_observations():
    observations = [
        _observation(0, 0, [0.0, 0.0, 0.0]),
        _observation(1, 1, [0.1, 0.0, 0.0]),
        _observation(2, 2, [0.2, 0.0, 0.0]),
    ]
    tracker = LocalObservationTracker(mode="oracle_3d", min_hits=1, max_3d_distance=1.0)

    records = tracker.run(observations, show_progress=False)

    assert len(records) == 3
    assert sorted(set(record.local_track_id for record in records)) == [1]
    assert tracker.summary()["num_kept_detections"] == 3


def test_tracker_handles_test_like_observations_without_gt():
    observations = [
        _observation(0, 0, [0.0, 0.0, 0.0], object_id=None),
        _observation(1, 1, [0.1, 0.0, 0.0], object_id=None),
    ]
    tracker = LocalObservationTracker(mode="oracle_3d", min_hits=1, max_3d_distance=1.0)

    records = tracker.run(observations, show_progress=False)

    assert len(records) == 2
    assert records[0].matched_gt_object_id is None
    assert sorted(set(record.local_track_id for record in records)) == [1]
