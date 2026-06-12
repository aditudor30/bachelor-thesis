import numpy as np

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_local_runner import local_record_from_benchmark
from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import BenchmarkTrackRecord
from deep_oc_sort_3d.observations.observation_types import Observation3D


def test_bytetrack_record_preserves_observation_3d_fields():
    tracked = BenchmarkTrackRecord(
        scene_id=20,
        scene_name="Warehouse_020",
        subset="official_val",
        split="val",
        camera_id="Camera_0000",
        frame_id=3,
        track_id=17,
        detection_id=4,
        class_id=0,
        class_name="Person",
        confidence=0.9,
        bbox_xyxy=(10.0, 20.0, 30.0, 60.0),
        matched_gt_object_id=99,
        track_age=4,
        track_hits=4,
        track_misses=0,
        track_state="confirmed",
        source_detection_id=4,
    )
    observation = Observation3D(
        scene_id=20,
        scene_name="Warehouse_020",
        split="val",
        camera_id="Camera_0000",
        frame_id=3,
        detection_id=4,
        class_id=0,
        class_name="Person",
        confidence=0.9,
        bbox_xyxy=(10.0, 20.0, 30.0, 60.0),
        bbox_xywh=(10.0, 20.0, 20.0, 40.0),
        center_3d=np.asarray([1.0, 2.0, 0.875]),
        dimensions_3d=np.asarray([0.6, 0.8, 1.75]),
        yaw=0.2,
        object_id=99,
        matched_gt=True,
        matched_iou=0.9,
        depth_value=None,
        depth_sampling_method=None,
        source="pseudo3d",
    )

    record = local_record_from_benchmark(tracked, observation)

    assert record.local_track_id == 17
    assert record.detection_id == 4
    assert np.allclose(record.center_3d, [1.0, 2.0, 0.875])
    assert np.allclose(record.dimensions_3d, [0.6, 0.8, 1.75])
    assert record.matched_gt_object_id == 99
