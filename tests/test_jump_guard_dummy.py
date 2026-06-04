import numpy as np
import pytest

from deep_oc_sort_3d.pseudo3d.jump_guard import apply_jump_guard, detect_jump_outliers


def test_detect_jump_outliers() -> None:
    centers = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    report = detect_jump_outliers(centers, [1, 2], max_step_m=6.0)
    assert report["num_jumps"] == 1
    assert report["jump_indices"] == [1]


def test_hold_previous_corrects_jump() -> None:
    centers = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    corrected, report = apply_jump_guard(centers, [1, 2], max_step_m=6.0, strategy="hold_previous")
    assert report["num_jumps"] == 1
    assert corrected[1, 0] == pytest.approx(0.0)


def test_interpolate_corrects_simple_jump() -> None:
    centers = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    corrected, report = apply_jump_guard(centers, [1, 2, 3], max_step_m=6.0, strategy="interpolate")
    assert report["num_jumps"] >= 1
    assert corrected[1, 0] == pytest.approx(1.0)
