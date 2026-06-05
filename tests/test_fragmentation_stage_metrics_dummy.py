"""Dummy tests for fragmentation stage metric helpers."""

from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.stage_metric_loaders import length_distribution, summarize_records_by_id


def test_length_distribution_counts_short_singleton_and_long():
    thresholds = FragmentationThresholds(singleton_length=1, short_track_length=3, very_short_track_length=5, long_track_length=10)
    stats = length_distribution([1, 2, 3, 5, 10], thresholds)

    assert stats["count"] == 5
    assert stats["singleton_count"] == 1
    assert stats["short_count"] == 3
    assert stats["long_count"] == 1


def test_summarize_records_by_id_groups_tracks():
    rows = [
        {"subset": "official_val", "scene_name": "Warehouse_020", "camera_id": "Camera_0000", "class_name": "Person", "local_track_id": 1, "frame_id": 0},
        {"subset": "official_val", "scene_name": "Warehouse_020", "camera_id": "Camera_0000", "class_name": "Person", "local_track_id": 1, "frame_id": 1},
        {"subset": "official_val", "scene_name": "Warehouse_020", "camera_id": "Camera_0000", "class_name": "Person", "local_track_id": 2, "frame_id": 2},
    ]
    stats = summarize_records_by_id(rows, "local_track_id")

    assert stats["total_records"] == 3
    assert stats["num_tracks"] == 2
    assert stats["singleton_count"] == 1

