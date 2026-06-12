from deep_oc_sort_3d.local_tracker_benchmark.bytetrack_style_tracker import ByteTrackStyleTracker
from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import BenchmarkDetection


def _detection(frame_id, confidence=0.9, class_id=0, x1=10.0):
    return BenchmarkDetection(
        scene_id=20,
        scene_name="Warehouse_020",
        subset="official_val",
        split="val",
        camera_id="Camera_0000",
        frame_id=frame_id,
        detection_id=frame_id,
        class_id=class_id,
        class_name="Person" if class_id == 0 else "Forklift",
        confidence=confidence,
        bbox_xyxy=(x1, 10.0, x1 + 10.0, 20.0),
    )


def test_bytetrack_second_stage_keeps_low_confidence_detection():
    tracker = ByteTrackStyleTracker(
        {
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.5,
            "match_thresh": 0.8,
            "second_stage_match_thresh": 0.5,
        }
    )

    first = tracker.update(0, [_detection(0, confidence=0.9)])
    second = tracker.update(1, [_detection(1, confidence=0.2)])

    assert first[0].track_id == second[0].track_id
    assert second[0].confidence == 0.2


def test_bytetrack_buffer_and_class_safe_matching():
    tracker = ByteTrackStyleTracker(
        {
            "track_high_thresh": 0.5,
            "new_track_thresh": 0.5,
            "match_thresh": 0.8,
            "track_buffer": 2,
        }
    )

    first = tracker.update(0, [_detection(0)])
    tracker.update(1, [])
    resumed = tracker.update(2, [_detection(2)])
    other_class = tracker.update(3, [_detection(3, class_id=1)])

    assert resumed[0].track_id == first[0].track_id
    assert other_class[0].track_id != first[0].track_id
    assert other_class[0].class_id == 1


def test_bytetrack_expires_track_after_real_frame_gap():
    tracker = ByteTrackStyleTracker(
        {
            "track_high_thresh": 0.5,
            "new_track_thresh": 0.5,
            "match_thresh": 0.8,
            "track_buffer": 2,
        }
    )

    first = tracker.update(0, [_detection(0)])
    after_gap = tracker.update(4, [_detection(4)])

    assert after_gap[0].track_id != first[0].track_id
