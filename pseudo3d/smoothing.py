"""Track-level smoothing helpers for pseudo-3D outputs."""

from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


def smooth_track_outputs(outputs: List[Pseudo3DOutput], config: Dict[str, Any]) -> List[Pseudo3DOutput]:
    """Apply simple median smoothing to center_3d when enabled."""
    if not bool(config.get("enabled", False)):
        return outputs
    window = max(1, int(config.get("window", 5)))
    radius = window // 2
    result = []
    for index, output in enumerate(outputs):
        if output.center_3d is None:
            result.append(output)
            continue
        values = []
        for other in outputs[max(0, index - radius) : min(len(outputs), index + radius + 1)]:
            if other.center_3d is not None:
                values.append(other.center_3d)
        if values:
            output.center_3d = np.median(np.asarray(values, dtype=float), axis=0)
            output.center_3d_source = "pseudo3d_motion_smoothed"
            output.source_notes = _append_note(output.source_notes, "center smoothed with median filter")
        result.append(output)
    return result


def _append_note(existing: str, note: str) -> str:
    if existing:
        return "%s; %s" % (existing, note)
    return note

