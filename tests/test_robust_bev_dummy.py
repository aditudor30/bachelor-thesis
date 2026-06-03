from deep_oc_sort_3d.visualization3d.bev_track_selection import BEVTrack
from deep_oc_sort_3d.visualization3d.robust_bev import compute_percentile_axis_limits, plot_robust_bev_tracks


def test_plot_robust_bev_tracks_creates_png(tmp_path):
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
    output = tmp_path / "bev.png"
    summary = plot_robust_bev_tracks(
        tracks,
        output,
        lower_percentile=25.0,
        upper_percentile=75.0,
        use_percentile_clipping=True,
    )
    assert output.exists()
    assert summary["plotted_tracks"] == 1
    assert "axis_limits" in summary


def test_compute_percentile_axis_limits_returns_tuple():
    tracks = [
        BEVTrack(
            global_track_id=1,
            class_id=None,
            class_name=None,
            frames=[0, 1, 2],
            x=[0.0, 1.0, 2.0],
            y=[10.0, 11.0, 12.0],
            confidence=None,
            length=3,
            start_frame=0,
            end_frame=2,
            duration=3,
            mean_confidence=None,
        )
    ]
    limits = compute_percentile_axis_limits(tracks)
    assert len(limits) == 4
    assert limits[0] < limits[1]
    assert limits[2] < limits[3]
