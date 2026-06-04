"""Temporal smoothing helpers for isolated pseudo-3D predictions."""

from typing import Tuple

import numpy as np


def median_filter_sequence(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a NaN-aware median filter to a 1D or 2D sequence."""
    array, original_shape = _to_2d(values)
    if window <= 1 or array.shape[0] == 0:
        return _restore_shape(array.copy(), original_shape)
    radius = int(window) // 2
    output = np.full(array.shape, np.nan, dtype=float)
    for index in range(array.shape[0]):
        start = max(0, index - radius)
        end = min(array.shape[0], index + radius + 1)
        chunk = array[start:end]
        for column in range(array.shape[1]):
            valid = chunk[:, column][np.isfinite(chunk[:, column])]
            if valid.size:
                output[index, column] = float(np.median(valid))
    return _restore_shape(output, original_shape)


def ema_filter_sequence(values: np.ndarray, alpha: float) -> np.ndarray:
    """Apply an exponential moving average to a 1D or 2D sequence."""
    array, original_shape = _to_2d(values)
    clipped_alpha = max(0.0, min(1.0, float(alpha)))
    output = np.full(array.shape, np.nan, dtype=float)
    previous = np.full((array.shape[1],), np.nan, dtype=float)
    for index in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[index, column]
            prev_value = previous[column]
            if np.isfinite(value):
                if np.isfinite(prev_value):
                    output[index, column] = clipped_alpha * value + (1.0 - clipped_alpha) * prev_value
                else:
                    output[index, column] = value
                previous[column] = output[index, column]
            elif np.isfinite(prev_value):
                output[index, column] = prev_value
    return _restore_shape(output, original_shape)


def smooth_center_sequence(centers: np.ndarray, method: str, window: int, alpha: float) -> np.ndarray:
    """Smooth an Nx3 center sequence using the configured method."""
    if method == "none":
        return np.asarray(centers, dtype=float).copy()
    if method == "ema":
        return ema_filter_sequence(centers, alpha)
    return median_filter_sequence(centers, window)


def smooth_depth_sequence(depths: np.ndarray, method: str, window: int, alpha: float) -> np.ndarray:
    """Smooth a 1D depth sequence using the configured method."""
    if method == "none":
        return np.asarray(depths, dtype=float).copy()
    if method == "ema":
        return ema_filter_sequence(depths, alpha)
    return median_filter_sequence(depths, window)


def _to_2d(values: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    array = np.asarray(values, dtype=float)
    original_shape = tuple(array.shape)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array, original_shape


def _restore_shape(values: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
    if len(original_shape) == 0:
        return values.reshape(-1)[0]
    if len(original_shape) == 1:
        return values.reshape(original_shape)
    return values.reshape(original_shape)
