import numpy as np

from deep_oc_sort_3d.tracking.track_io import (
    read_local_tracks_csv,
    read_local_tracks_jsonl,
    write_local_tracks_csv,
    write_local_tracks_jsonl,
)
from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord


def _record():
    return LocalTrackRecord(
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        camera_id="Camera_0000",
        frame_id=5,
        local_track_id=3,
        detection_id=9,
        class_id=0,
        class_name="Person",
        confidence=0.75,
        bbox_xyxy=(1.0, 2.0, 11.0, 22.0),
        bbox_xywh=(1.0, 2.0, 10.0, 20.0),
        center_3d=np.asarray([1.0, 2.0, 3.0], dtype=float),
        dimensions_3d=np.asarray([0.5, 0.6, 1.7], dtype=float),
        yaw=0.2,
        matched_gt_object_id=42,
        matched_gt=True,
        track_age=5,
        track_hits=4,
        track_misses=0,
        track_state="confirmed",
    )


def test_local_tracks_csv_roundtrip(tmp_path):
    path = tmp_path / "tracks.csv"
    write_local_tracks_csv([_record()], path)

    records = read_local_tracks_csv(path)

    assert len(records) == 1
    assert records[0].local_track_id == 3
    assert records[0].bbox_xyxy == (1.0, 2.0, 11.0, 22.0)
    np.testing.assert_allclose(records[0].center_3d, np.asarray([1.0, 2.0, 3.0]))


def test_local_tracks_jsonl_roundtrip(tmp_path):
    path = tmp_path / "tracks.jsonl"
    write_local_tracks_jsonl([_record()], path)

    records = read_local_tracks_jsonl(path)

    assert len(records) == 1
    assert records[0].matched_gt_object_id == 42
    assert records[0].track_state == "confirmed"
