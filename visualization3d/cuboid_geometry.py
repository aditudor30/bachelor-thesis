"""3D cuboid geometry helpers for visualization.

Convention:
- ``center`` is the geometric center of the cuboid in world coordinates.
- ``dimensions`` are ordered as ``[width, length, height]``.
- width is local x, length is local y, height is local z.
- yaw rotates around the vertical z axis in the x-y plane.
"""

from typing import List, Tuple

import numpy as np

from deep_oc_sort_3d.geometry.box_projection import create_3d_box_corners


def validate_cuboid_inputs(center, dimensions, yaw) -> bool:
    """Return True if center, dimensions, and yaw can define a finite cuboid."""
    try:
        center_arr = np.asarray(center, dtype=float).reshape(3)
        dims = np.asarray(dimensions, dtype=float).reshape(3)
        yaw_value = float(yaw)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(center_arr)):
        return False
    if not np.all(np.isfinite(dims)):
        return False
    if not np.isfinite(yaw_value):
        return False
    return bool(np.all(dims > 0.0))


def compute_3d_cuboid_corners(center: np.ndarray, dimensions: np.ndarray, yaw: float) -> np.ndarray:
    """Compute 8 cuboid corners in world coordinates.

    The returned order is bottom face corners 0-3 followed by top face corners
    4-7. This matches ``get_cuboid_edges`` for drawing.
    """
    if not validate_cuboid_inputs(center, dimensions, yaw):
        raise ValueError("Invalid cuboid inputs: center, dimensions, and yaw must be finite; dimensions must be positive")
    return create_3d_box_corners(center, dimensions, yaw)


def get_cuboid_edges() -> List[Tuple[int, int]]:
    """Return edge index pairs for the 8-corner cuboid convention."""
    return [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

