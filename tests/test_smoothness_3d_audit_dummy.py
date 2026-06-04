from deep_oc_sort_3d.audit3d.smoothness_3d_audit import compute_track_smoothness, find_worst_jumps


def _row(frame_id, x, y, z):
    return {
        "scene_id": 23,
        "class_id": 0,
        "object_id": 1,
        "frame_id": frame_id,
        "x": x,
        "y": y,
        "z": z,
        "width": 1.0,
        "length": 2.0,
        "height": 1.5,
        "yaw": 0.0,
    }


def test_smoothness_audit_smooth_track_has_expected_distance():
    rows = [_row(0, 0.0, 0.0, 0.0), _row(1, 1.0, 0.0, 0.0), _row(2, 2.0, 0.0, 0.0)]

    summary = compute_track_smoothness(rows)

    assert summary["step_distance_max"] == 1.0
    assert summary["travel_distance"] == 2.0
    assert summary["straight_line_distance"] == 2.0
    assert summary["path_efficiency"] == 1.0
    assert summary["jump_count"] == 0


def test_smoothness_audit_detects_jump():
    rows = [_row(0, 0.0, 0.0, 0.0), _row(1, 10.0, 0.0, 0.0)]

    summary = compute_track_smoothness(rows)
    worst = find_worst_jumps(rows, top_k=1)

    assert summary["jump_count"] == 1
    assert worst[0]["step_distance_3d"] == 10.0

