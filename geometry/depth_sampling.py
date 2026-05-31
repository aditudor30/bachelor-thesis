"""Robust depth sampling strategies inside 2D bounding boxes.

These helpers operate on one already-loaded depth frame. They do not open HDF5
files and never load more data than the caller passes in.
"""

from typing import Any, Optional, Tuple

import numpy as np


def clip_bbox_to_image(
    bbox_xyxy: Tuple[float, float, float, float],
    image_shape: Tuple[int, ...],
) -> Optional[Tuple[int, int, int, int]]:
    """Clip an xyxy bbox to image bounds and return integer slice limits.

    The returned box is ``(left, top, right, bottom)`` where right/bottom are
    exclusive slice limits. Returns None when the clipped box is empty.
    """
    if len(image_shape) < 2:
        return None
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, int(np.floor(min(x1, x2))))
    right = min(width, int(np.ceil(max(x1, x2))))
    top = max(0, int(np.floor(min(y1, y2))))
    bottom = min(height, int(np.ceil(max(y1, y2))))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def extract_depth_patch(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    margin_ratio: float = 0.0,
) -> Optional[np.ndarray]:
    """Extract a clipped depth patch for a bbox, optionally expanded by margin."""
    if depth is None or depth.ndim < 2:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    width = abs(float(x2) - float(x1))
    height = abs(float(y2) - float(y1))
    margin_x = width * float(margin_ratio)
    margin_y = height * float(margin_ratio)
    expanded = (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)
    clipped = clip_bbox_to_image(expanded, depth.shape)
    if clipped is None:
        return None
    left, top, right, bottom = clipped
    return depth[top:bottom, left:right]


def valid_depth_values(
    patch: np.ndarray,
    min_valid_depth: float = 1e-6,
    max_valid_depth: Optional[float] = None,
) -> np.ndarray:
    """Return flattened valid depth values from a patch."""
    if patch is None:
        return np.asarray([], dtype=float)
    values = np.asarray(patch, dtype=float).reshape(-1)
    mask = np.isfinite(values) & (values > float(min_valid_depth))
    if max_valid_depth is not None:
        mask = mask & (values <= float(max_valid_depth))
    return values[mask]


def sample_depth_center(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    window: int = 5,
) -> Optional[float]:
    """Sample median depth in a small window around bbox center."""
    x1, y1, x2, y2 = bbox_xyxy
    u = (float(x1) + float(x2)) * 0.5
    v = (float(y1) + float(y2)) * 0.5
    return _sample_window(depth, u, v, window)


def sample_depth_bottom_center(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    window: int = 5,
) -> Optional[float]:
    """Sample median depth in a small window around bbox bottom-center."""
    x1, _y1, x2, y2 = bbox_xyxy
    u = (float(x1) + float(x2)) * 0.5
    v = float(y2)
    return _sample_window(depth, u, v, window)


def sample_depth_lower_region(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    lower_ratio: float = 0.35,
    statistic: str = "median",
) -> Optional[float]:
    """Sample depth from the lower portion of the bbox.

    This is often more relevant for objects standing on the floor than a median
    over the full bbox, which can include shelves, occluders, or far background.
    """
    if depth is None or depth.ndim < 2:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    ratio = min(max(float(lower_ratio), 0.0), 1.0)
    top_lower = float(y1) + (float(y2) - float(y1)) * (1.0 - ratio)
    patch = extract_depth_patch(depth, (x1, top_lower, x2, y2), margin_ratio=0.0)
    return _statistic(valid_depth_values(patch), statistic)


def sample_depth_center_region(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    center_ratio: float = 0.5,
    statistic: str = "median",
) -> Optional[float]:
    """Sample depth from the central crop of a bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    ratio = min(max(float(center_ratio), 0.0), 1.0)
    if ratio <= 0.0:
        return None
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    half_w = abs(float(x2) - float(x1)) * ratio * 0.5
    half_h = abs(float(y2) - float(y1)) * ratio * 0.5
    patch = extract_depth_patch(depth, (cx - half_w, cy - half_h, cx + half_w, cy + half_h), margin_ratio=0.0)
    return _statistic(valid_depth_values(patch), statistic)


def sample_depth_percentile(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    percentile: float = 30,
) -> Optional[float]:
    """Sample a depth percentile over the full bbox."""
    patch = extract_depth_patch(depth, bbox_xyxy, margin_ratio=0.0)
    values = valid_depth_values(patch)
    if values.size == 0:
        return None
    return float(np.percentile(values, float(percentile)))


def sample_depth_trimmed_median(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    lower_percentile: float = 10,
    upper_percentile: float = 60,
) -> Optional[float]:
    """Sample median after trimming depth outliers by percentiles."""
    patch = extract_depth_patch(depth, bbox_xyxy, margin_ratio=0.0)
    values = valid_depth_values(patch)
    if values.size == 0:
        return None
    low = float(np.percentile(values, float(lower_percentile)))
    high = float(np.percentile(values, float(upper_percentile)))
    trimmed = values[(values >= low) & (values <= high)]
    if trimmed.size == 0:
        return None
    return float(np.median(trimmed))


def sample_depth_histogram_mode(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    bins: int = 64,
) -> Optional[float]:
    """Approximate the dominant depth mode inside the bbox."""
    patch = extract_depth_patch(depth, bbox_xyxy, margin_ratio=0.0)
    values = valid_depth_values(patch)
    if values.size == 0:
        return None
    hist, edges = np.histogram(values, bins=int(bins))
    if hist.size == 0:
        return None
    index = int(np.argmax(hist))
    left = edges[index]
    right = edges[index + 1]
    in_bin = values[(values >= left) & (values <= right)]
    if in_bin.size == 0:
        return float((left + right) * 0.5)
    return float(np.median(in_bin))


def sample_depth_robust(
    depth: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    method: str = "lower_median",
    **kwargs: Any
) -> Optional[float]:
    """Sample depth from a bbox using a named robust strategy."""
    if method == "center":
        return sample_depth_center(depth, bbox_xyxy, window=int(kwargs.get("window", 5)))
    if method == "bottom_center":
        return sample_depth_bottom_center(depth, bbox_xyxy, window=int(kwargs.get("window", 5)))
    if method == "bbox_median":
        return sample_depth_percentile(depth, bbox_xyxy, percentile=50)
    if method == "bbox_percentile_30":
        return sample_depth_percentile(depth, bbox_xyxy, percentile=30)
    if method == "lower_median":
        return sample_depth_lower_region(depth, bbox_xyxy, lower_ratio=float(kwargs.get("lower_ratio", 0.35)))
    if method == "lower_percentile_30":
        return sample_depth_lower_region(
            depth,
            bbox_xyxy,
            lower_ratio=float(kwargs.get("lower_ratio", 0.35)),
            statistic="percentile_30",
        )
    if method == "center_median":
        return sample_depth_center_region(depth, bbox_xyxy, center_ratio=float(kwargs.get("center_ratio", 0.5)))
    if method == "trimmed_median":
        return sample_depth_trimmed_median(
            depth,
            bbox_xyxy,
            lower_percentile=float(kwargs.get("lower_percentile", 10)),
            upper_percentile=float(kwargs.get("upper_percentile", 60)),
        )
    if method == "histogram_mode":
        return sample_depth_histogram_mode(depth, bbox_xyxy, bins=int(kwargs.get("bins", 64)))
    raise ValueError("Unsupported depth sampling method: %s" % method)


def _sample_window(depth: np.ndarray, u: float, v: float, window: int) -> Optional[float]:
    if depth is None or depth.ndim < 2:
        return None
    height, width = depth.shape[:2]
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    radius = max(int(window), 1) // 2
    left = max(0, x - radius)
    right = min(width, x + radius + 1)
    top = max(0, y - radius)
    bottom = min(height, y + radius + 1)
    return _statistic(valid_depth_values(depth[top:bottom, left:right]), "median")


def _statistic(values: np.ndarray, statistic: str) -> Optional[float]:
    if values is None or values.size == 0:
        return None
    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "min":
        return float(np.min(values))
    if statistic == "max":
        return float(np.max(values))
    if statistic.startswith("percentile_"):
        percentile = float(statistic.split("_", 1)[1])
        return float(np.percentile(values, percentile))
    raise ValueError("Unsupported statistic: %s" % statistic)

