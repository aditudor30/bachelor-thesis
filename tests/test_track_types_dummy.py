import numpy as np

from deep_oc_sort_3d.observations.observation_types import Observation3D
from deep_oc_sort_3d.tracking.track_state import LocalTrack
from deep_oc_sort_3d.tracking.track_types import (
    array_to_list,
    detection_from_observation,
    list_to_array,
    record_from_track_detection,
)


def _observation(frame_id=0, detection_id=1):
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
        bbox_xyxy=(10.0, 20.0, 30.0, 60.0),
        bbox_xywh=(10.0, 20.0, 20.0, 40.0),
        center_3d=np.asarray([1.0, 2.0, 0.9], dtype=float),
        dimensions_3d=np.asarray([0.6, 0.8, 1.8], dtype=float),
        yaw=0.1,
        object_id=42,
        matched_gt=True,
        matched_iou=0.8,
        depth_value=4.0,
        depth_sampling_method="center_median",
        source="dummy",
    )


def test_detection_from_observation_preserves_fields():
    det = detection_from_observation(_observation())

    assert det.scene_name == "Warehouse_000"
    assert det.camera_id == "Camera_0000"
    assert det.frame_id == 0
    assert det.class_name == "Person"
    assert det.object_id == 42
    assert det.bbox_xyxy == (10.0, 20.0, 30.0, 60.0)
    np.testing.assert_allclose(det.center_3d, np.asarray([1.0, 2.0, 0.9], dtype=float))


def test_record_from_track_detection_and_array_helpers():
    det = detection_from_observation(_observation())
    track = LocalTrack(7, det)
    record = record_from_track_detection(track, det)

    assert record.local_track_id == 7
    assert record.matched_gt_object_id == 42
    assert array_to_list(record.center_3d) == [1.0, 2.0, 0.9]
    np.testing.assert_allclose(list_to_array([1, 2, 3]), np.asarray([1.0, 2.0, 3.0]))
