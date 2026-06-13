"""Track-level robust dimension stabilization for V4."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import clone_row, dimensions, group_tracks, progress_iter, read_json


def stabilize_track_dimensions(
    rows: Sequence[OfficialTrack1Row],
    config: Dict[str, Any],
    progress: bool = True,
) -> Tuple[List[OfficialTrack1Row], List[Dict[str, Any]]]:
    """Replace varying dimensions with a robust per-track median."""
    rules = config.get("dimension_consistency", {})
    if not bool(rules.get("enabled", True)):
        return list(rows), []
    priors = load_class_dimension_priors(config) if bool(rules.get("use_class_prior_fallback", True)) else {}
    minimum = int(rules.get("min_track_length_for_track_median", 3))
    max_ratio = float(rules.get("max_dimension_change_ratio_apply", 0.75))
    output = []
    changes = []
    for key, track in progress_iter(list(group_tracks(rows).items()), progress, "V4 dimension consistency"):
        values = np.asarray([dimensions(row) for row in track], dtype=float)
        target = np.median(values, axis=0) if len(track) >= minimum else priors.get(int(key[1]))
        if target is None or not np.all(np.isfinite(target)) or not np.all(np.asarray(target) > 0.0):
            output.extend(track)
            continue
        target = np.asarray(target, dtype=float)
        for index, row in enumerate(track):
            old = values[index]
            ratios = np.abs(target - old) / np.maximum(np.abs(old), 1e-6)
            if np.any(ratios > max_ratio):
                output.append(row)
                continue
            new_row = clone_row(row, width=float(target[0]), length=float(target[1]), height=float(target[2]))
            output.append(new_row)
            if float(np.max(np.abs(target - old))) > 1e-9:
                changes.append({
                    "scene_id": key[0], "class_id": key[1], "object_id": key[2], "frame_id": row.frame_id,
                    "field_group": "dimensions", "reason": "track_median" if len(track) >= minimum else "class_prior_fallback",
                    "old_width": float(old[0]), "old_length": float(old[1]), "old_height": float(old[2]),
                    "new_width": float(target[0]), "new_length": float(target[1]), "new_height": float(target[2]),
                    "max_dimension_change_ratio": float(np.max(ratios)),
                })
    return sorted(output, key=lambda row: row.key()), changes


def load_class_dimension_priors(config: Dict[str, Any]) -> Dict[int, np.ndarray]:
    """Load optional internal priors and map them explicitly to official classes."""
    path = Path(str(config.get("paths", {}).get("class_priors_path", "")))
    if not path.is_file():
        return {}
    data = read_json(path)
    mapping_values = config.get("class_mapping", {}).get("official_to_internal", {})
    mapping = {int(key): int(value) for key, value in mapping_values.items()}
    output = {}
    for official_id, internal_id in mapping.items():
        item = _find_prior_item(data, internal_id)
        dims = _extract_prior_dimensions(item)
        if dims is not None:
            output[official_id] = dims
    return output


def _find_prior_item(data: Dict[str, Any], class_id: int) -> Dict[str, Any]:
    for container_key in ["classes", "class_priors", "priors"]:
        container = data.get(container_key, {})
        if isinstance(container, dict):
            item = container.get(str(class_id), container.get(class_id, {}))
            if isinstance(item, dict):
                return item
    item = data.get(str(class_id), data.get(class_id, {}))
    return item if isinstance(item, dict) else {}


def _extract_prior_dimensions(item: Dict[str, Any]) -> Optional[np.ndarray]:
    direct = item.get("dimensions")
    if isinstance(direct, (list, tuple)) and len(direct) >= 3:
        values = np.asarray(direct[:3], dtype=float)
        return values if np.all(np.isfinite(values)) and np.all(values > 0.0) else None
    dimensions_data = item.get("dimensions", item)
    values = []
    for name in ["width", "length", "height"]:
        value = dimensions_data.get(name) if isinstance(dimensions_data, dict) else None
        if isinstance(value, dict):
            value = value.get("median", value.get("mean"))
        if value is None:
            return None
        values.append(float(value))
    array = np.asarray(values, dtype=float)
    return array if np.all(np.isfinite(array)) and np.all(array > 0.0) else None
