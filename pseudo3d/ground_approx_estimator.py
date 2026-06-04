"""Ground-contact approximation scaffold for pseudo-3D.

This module keeps the method explicit, but Step 15C falls back to bbox-height
depth unless a future ground-plane implementation is provided.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.pseudo3d.camera_model_adapter import CameraModel


def estimate_ground_contact_point(
    bbox_xyxy: Tuple[float, float, float, float],
    camera_model: CameraModel,
    config: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return no ground estimate unless an explicit ground plane is configured."""
    if not bool(config.get("enabled", False)):
        return None, "ground_approx_disabled"
    if bool(config.get("require_ground_plane", True)):
        return None, "missing_ground_plane"
    return None, "ground_approx_not_implemented"

