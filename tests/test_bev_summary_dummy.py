from deep_oc_sort_3d.visualization3d.bev_summary import compute_bev_coordinate_summary
from deep_oc_sort_3d.visualization3d.bev_track_selection import BEVTrack


def test_compute_bev_coordinate_summary_with_outlier():
    tracks = [
        BEVTrack(
            global_track_id=1,
            class_id=0,
            class_name="Person",
            frames=[0, 1, 2, 3],
            x=[0.0, 1.0, 2.0, 1000.0],
            y=[0.0, 1.0, 2.0, 3.0],
            confidence=[0.9, 0.9, 0.9, 0.9],
            length=4,
            start_frame=0,
            end_frame=3,
            duration=4,
            mean_confidence=0.9,
        )
    ]
    summary = compute_bev_coordinate_summary(tracks, lower_percentile=25.0, upper_percentile=75.0)
    assert summary["num_tracks"] == 1
    assert summary["num_points"] == 4
    assert summary["x_min"] == 0.0
    assert summary["x_max"] == 1000.0
    assert summary["x_p_low"] > 0.0
    assert summary["x_p_high"] < 1000.0
    assert summary["outlier_points_count"] > 0

