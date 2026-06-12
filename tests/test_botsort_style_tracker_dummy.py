import numpy as np

from deep_oc_sort_3d.local_tracker_benchmark.botsort_style_tracker import BoTSORTStyleTracker
from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import BenchmarkDetection


def _detection(frame_id, x1, embedding=None):
    return BenchmarkDetection(
        scene_id=20,
        scene_name="Warehouse_020",
        subset="official_val",
        split="val",
        camera_id="Camera_0000",
        frame_id=frame_id,
        detection_id=frame_id,
        class_id=0,
        class_name="Person",
        confidence=0.9,
        bbox_xyxy=(x1, 10.0, x1 + 10.0, 20.0),
        embedding=embedding,
    )


def test_botsort_motion_prediction_continues_track():
    tracker = BoTSORTStyleTracker(
        {
            "track_high_thresh": 0.5,
            "new_track_thresh": 0.5,
            "match_thresh": 0.5,
            "track_buffer": 5,
        },
        use_reid=False,
    )

    first = tracker.update(0, [_detection(0, 0.0)])
    second = tracker.update(1, [_detection(1, 3.0)])
    predicted = tracker.update(2, [_detection(2, 6.0)])

    assert first[0].track_id == second[0].track_id
    assert second[0].track_id == predicted[0].track_id


def test_botsort_accepts_optional_person_embeddings():
    tracker = BoTSORTStyleTracker(
        {
            "track_high_thresh": 0.5,
            "new_track_thresh": 0.5,
            "match_thresh": 0.8,
            "appearance_weight": 0.5,
            "appearance_thresh": 0.5,
        },
        use_reid=True,
    )
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)

    first = tracker.update(0, [_detection(0, 0.0, embedding=embedding)])
    second = tracker.update(1, [_detection(1, 1.0, embedding=embedding)])

    assert first[0].track_id == second[0].track_id
