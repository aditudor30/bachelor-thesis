import numpy as np
import pytest

from deep_oc_sort_3d.pseudo3d.temporal_smoothing import ema_filter_sequence, median_filter_sequence


def test_median_filter_reduces_outlier() -> None:
    values = np.asarray([1.0, 1.0, 100.0, 1.0, 1.0])
    smoothed = median_filter_sequence(values, window=3)
    assert smoothed[2] == pytest.approx(1.0)


def test_ema_filter_sequence() -> None:
    values = np.asarray([0.0, 10.0])
    smoothed = ema_filter_sequence(values, alpha=0.5)
    assert smoothed[0] == pytest.approx(0.0)
    assert smoothed[1] == pytest.approx(5.0)


def test_median_filter_handles_nan() -> None:
    values = np.asarray([np.nan, 2.0, np.nan])
    smoothed = median_filter_sequence(values, window=3)
    assert smoothed[1] == pytest.approx(2.0)
    assert smoothed[0] == pytest.approx(2.0)
