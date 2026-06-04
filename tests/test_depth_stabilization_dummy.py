import numpy as np
import pytest

from deep_oc_sort_3d.pseudo3d.depth_stabilization import detect_depth_outliers, stabilize_depth_sequence


def test_detect_depth_outlier() -> None:
    depths = np.asarray([10.0, 10.2, 50.0, 10.1, 9.9])
    report = detect_depth_outliers(depths, threshold=3.5)
    assert report["num_outliers"] == 1
    assert report["outlier_indices"] == [2]


def test_stabilize_depth_sequence() -> None:
    depths = np.asarray([10.0, 10.0, 50.0, 10.0, 10.0])
    smoothed, report = stabilize_depth_sequence(depths, window=3, max_relative_change=0.5)
    assert smoothed[2] == pytest.approx(10.0)
    assert report["median_depth"] == pytest.approx(10.0)
