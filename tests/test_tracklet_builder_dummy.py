import numpy as np

from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord
from deep_oc_sort_3d.tracklets.tracklet_builder import LocalTrackletBuilder


def _record(frame_id, confidence=0.8, object_id=100):
    return LocalTrackRecord(
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        camera_id="Camera_0000",
        frame_id=frame_id,
        local_track_id=1,
        detection_id=frame_id + 10,
        class_id=0,
        class_name="Person",
        confidence=confidence,
        bbox_xyxy=(float(frame_id), 0.0, float(frame_id + 10), 20.0),
        bbox_xywh=(float(frame_id), 0.0, 10.0, 20.0),
        center_3d=np.asarray([float(frame_id), 0.0, 1.0], dtype=float),
        dimensions_3d=np.asarray([0.6, 0.8, 1.8], dtype=float),
        yaw=0.1,
        matched_gt_object_id=object_id,
        matched_gt=True,
        track_age=frame_id + 1,
        track_hits=frame_id + 1,
        track_misses=0,
        track_state="confirmed",
    )


def test_build_one_tracklet_stats():
    records = [_record(0, 0.7), _record(1, 0.8), _record(2, 0.9)]
    builder = LocalTrackletBuilder(smooth_trajectory=False)

    tracklet = builder.build_one_tracklet(records)

    assert tracklet.start_frame == 0
    assert tracklet.end_frame == 2
    assert tracklet.length == 3
    assert abs(tracklet.mean_confidence - 0.8) < 1e-6
    assert tracklet.bbox_mean == (1.0, 0.0, 11.0, 20.0)
    np.testing.assert_allclose(tracklet.center_3d_mean, np.asarray([1.0, 0.0, 1.0]))
    assert tracklet.majority_gt_object_id == 100
    assert tracklet.gt_purity == 1.0
    assert tracklet.is_valid_for_mtmc


def test_build_from_records_groups_tracks():
    records = [_record(0), _record(1)]
    other = _record(0)
    other.local_track_id = 2
    records.append(other)

    tracklets = LocalTrackletBuilder(smooth_trajectory=False).build_from_records(records)

    assert len(tracklets) == 2
