import numpy as np

from deep_oc_sort_3d.tracking.track_eval import evaluate_local_tracks
from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord


def _record(track_id, frame_id, object_id):
    return LocalTrackRecord(
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        camera_id="Camera_0000",
        frame_id=frame_id,
        local_track_id=track_id,
        detection_id=frame_id,
        class_id=0,
        class_name="Person",
        confidence=0.8,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        bbox_xywh=(0.0, 0.0, 10.0, 10.0),
        center_3d=np.asarray([float(frame_id), 0.0, 0.0], dtype=float),
        dimensions_3d=np.asarray([0.6, 0.8, 1.8], dtype=float),
        yaw=0.0,
        matched_gt_object_id=object_id,
        matched_gt=object_id is not None,
        track_age=frame_id + 1,
        track_hits=frame_id + 1,
        track_misses=0,
        track_state="confirmed",
    )


def test_evaluate_local_tracks_reports_basic_metrics():
    records = [_record(1, 0, 10), _record(1, 1, 10), _record(1, 2, 10)]

    metrics = evaluate_local_tracks(records)

    assert metrics["num_records"] == 3
    assert metrics["num_tracks"] == 1
    assert metrics["mean_track_length"] == 3.0
    assert metrics["id_switches_approx"] == 0
    assert metrics["purity_mean"] == 1.0


def test_evaluate_local_tracks_detects_approx_fragmentation():
    records = [_record(1, 0, 10), _record(2, 1, 10)]

    metrics = evaluate_local_tracks(records)

    assert metrics["id_switches_approx"] == 1
    assert metrics["fragmentations_approx"] == 1
