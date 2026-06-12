"""Feature computation for Person fragment pairs."""

import math
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.pair_dataset_io import safe_float


def build_pair_features(
    candidate_pair: Dict[str, Any],
    train_camera_pairs: Optional[Set[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute identity, ReID, temporal, camera, geometry and quality features."""
    fragment_a = candidate_pair.get("_fragment_a") or {}
    fragment_b = candidate_pair.get("_fragment_b") or {}
    row = {key: value for key, value in candidate_pair.items() if not key.startswith("_")}

    embedding_a = fragment_a.get("_embedding")
    embedding_b = fragment_b.get("_embedding")
    similarity = cosine_similarity(embedding_a, embedding_b)
    row.update(
        {
            "reid_similarity": similarity,
            "reid_distance": 1.0 - similarity if similarity is not None else None,
            "embedding_available_a": int(bool(fragment_a.get("embedding_available"))),
            "embedding_available_b": int(bool(fragment_b.get("embedding_available"))),
            "embedding_valid_pair": int(similarity is not None),
        }
    )

    temporal = temporal_features(fragment_a, fragment_b)
    camera = camera_features(fragment_a, fragment_b, train_camera_pairs)
    geometry = geometry_features(fragment_a, fragment_b, temporal)
    quality = quality_features(fragment_a, fragment_b)
    row.update(temporal)
    row.update(camera)
    row.update(geometry)
    row.update(quality)
    row.update(conflict_features(row, config or {}))
    return row


def cosine_similarity(vector_a: Any, vector_b: Any) -> Optional[float]:
    """Return cosine similarity, or None for missing/invalid vectors."""
    if vector_a is None or vector_b is None:
        return None
    array_a = np.asarray(vector_a, dtype=np.float64).reshape(-1)
    array_b = np.asarray(vector_b, dtype=np.float64).reshape(-1)
    if array_a.size == 0 or array_a.shape != array_b.shape:
        return None
    if not np.all(np.isfinite(array_a)) or not np.all(np.isfinite(array_b)):
        return None
    denominator = float(np.linalg.norm(array_a) * np.linalg.norm(array_b))
    if denominator <= 1e-12:
        return None
    return float(np.dot(array_a, array_b) / denominator)


def temporal_features(fragment_a: Dict[str, Any], fragment_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compute interval gap, overlap and ordering."""
    start_a = int(fragment_a.get("frame_start") or 0)
    end_a = int(fragment_a.get("frame_end") or start_a)
    start_b = int(fragment_b.get("frame_start") or 0)
    end_b = int(fragment_b.get("frame_end") or start_b)
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    if overlap > 0:
        gap = 0
        order = "overlap"
    elif end_a < start_b:
        gap = start_b - end_a - 1
        order = "a_before_b"
    else:
        gap = start_a - end_b - 1
        order = "b_before_a"
    duration_a = max(1, end_a - start_a + 1)
    duration_b = max(1, end_b - start_b + 1)
    return {
        "frame_start_a": start_a,
        "frame_end_a": end_a,
        "frame_start_b": start_b,
        "frame_end_b": end_b,
        "temporal_gap": gap,
        "temporal_overlap": overlap,
        "temporal_order": order,
        "duration_a": duration_a,
        "duration_b": duration_b,
        "duration_ratio": min(duration_a, duration_b) / float(max(duration_a, duration_b)),
    }


def camera_features(
    fragment_a: Dict[str, Any],
    fragment_b: Dict[str, Any],
    train_camera_pairs: Optional[Set[str]],
) -> Dict[str, Any]:
    """Compute stable camera-pair features."""
    camera_a = str(fragment_a.get("camera_id") or "")
    camera_b = str(fragment_b.get("camera_id") or "")
    camera_pair = "__".join(sorted((camera_a, camera_b)))
    same_camera = camera_a == camera_b
    seen = True if train_camera_pairs is None else camera_pair in train_camera_pairs
    return {
        "camera_a": camera_a,
        "camera_b": camera_b,
        "camera_pair": camera_pair,
        "same_camera": int(same_camera),
        "cross_camera": int(not same_camera),
        "camera_pair_seen_in_train": int(seen),
    }


def geometry_features(
    fragment_a: Dict[str, Any],
    fragment_b: Dict[str, Any],
    temporal: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute pseudo3D endpoint, center and motion compatibility features."""
    start_a = point_from_fragment(fragment_a, "start")
    end_a = point_from_fragment(fragment_a, "end")
    mean_a = point_from_fragment(fragment_a, "mean")
    start_b = point_from_fragment(fragment_b, "start")
    end_b = point_from_fragment(fragment_b, "end")
    mean_b = point_from_fragment(fragment_b, "mean")
    velocity_a = vector_from_fragment(fragment_a, "velocity")
    velocity_b = vector_from_fragment(fragment_b, "velocity")

    start_distance = distance(start_a, start_b)
    end_distance = distance(end_a, end_b)
    endpoint_values = [value for value in (distance(end_a, start_b), distance(end_b, start_a)) if value is not None]
    center_distance = distance(mean_a, mean_b)
    spatial_xy = None
    spatial_z = None
    if mean_a is not None and mean_b is not None:
        delta = mean_a - mean_b
        spatial_xy = float(np.linalg.norm(delta[:2]))
        spatial_z = float(abs(delta[2]))
    velocity_similarity = cosine_similarity(velocity_a, velocity_b)
    velocity_difference = distance(velocity_a, velocity_b)
    speed_a = safe_float(fragment_a.get("speed_mean"))
    speed_b = safe_float(fragment_b.get("speed_mean"))
    speed_difference = abs(speed_a - speed_b) if speed_a is not None and speed_b is not None else None
    expected_error = expected_position_error(fragment_a, fragment_b, temporal)
    components = []
    if velocity_similarity is not None:
        components.append((velocity_similarity + 1.0) / 2.0)
    if expected_error is not None:
        components.append(math.exp(-expected_error / 10.0))
    motion_score = float(sum(components) / len(components)) if components else None
    return {
        "start_distance_3d": start_distance,
        "end_distance_3d": end_distance,
        "min_endpoint_distance_3d": min(endpoint_values) if endpoint_values else None,
        "center_mean_distance_3d": center_distance,
        "spatial_distance_xy": spatial_xy,
        "spatial_distance_z": spatial_z,
        "velocity_cosine": velocity_similarity,
        "velocity_difference": velocity_difference,
        "speed_difference": speed_difference,
        "expected_position_error": expected_error,
        "motion_consistency_score": motion_score,
    }


def expected_position_error(
    fragment_a: Dict[str, Any], fragment_b: Dict[str, Any], temporal: Dict[str, Any]
) -> Optional[float]:
    """Extrapolate the earlier fragment to the later fragment start."""
    if temporal.get("temporal_order") == "overlap":
        return distance(point_from_fragment(fragment_a, "mean"), point_from_fragment(fragment_b, "mean"))
    if temporal.get("temporal_order") == "a_before_b":
        earlier, later = fragment_a, fragment_b
    else:
        earlier, later = fragment_b, fragment_a
    endpoint = point_from_fragment(earlier, "end")
    target = point_from_fragment(later, "start")
    velocity = vector_from_fragment(earlier, "velocity")
    if endpoint is None or target is None or velocity is None:
        return None
    gap = float(temporal.get("temporal_gap") or 0)
    predicted = endpoint + velocity * gap
    return float(np.linalg.norm(predicted - target))


def quality_features(fragment_a: Dict[str, Any], fragment_b: Dict[str, Any]) -> Dict[str, Any]:
    """Copy compact quality statistics from both fragments."""
    return {
        "num_obs_a": fragment_a.get("num_observations"),
        "num_obs_b": fragment_b.get("num_observations"),
        "mean_conf_a": fragment_a.get("mean_confidence"),
        "mean_conf_b": fragment_b.get("mean_confidence"),
        "min_conf_a": fragment_a.get("min_confidence"),
        "min_conf_b": fragment_b.get("min_confidence"),
        "bbox_area_mean_a": fragment_a.get("bbox_area_mean"),
        "bbox_area_mean_b": fragment_b.get("bbox_area_mean"),
        "bbox_height_mean_a": fragment_a.get("bbox_height_mean"),
        "bbox_height_mean_b": fragment_b.get("bbox_height_mean"),
        "fragment_quality_a": fragment_a.get("fragment_quality"),
        "fragment_quality_b": fragment_b.get("fragment_quality"),
        "gt_purity_a": fragment_a.get("gt_purity"),
        "gt_purity_b": fragment_b.get("gt_purity"),
    }


def conflict_features(row: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, int]:
    """Compute simple conservative conflict flags."""
    thresholds = config.get("conflict_thresholds", {})
    large_spatial = float(thresholds.get("large_spatial_gap", 50.0))
    large_temporal = int(thresholds.get("large_temporal_gap", 900))
    low_quality_conf = float(thresholds.get("low_quality_confidence", 0.1))
    overlap = int(row.get("temporal_overlap") or 0)
    same_camera = bool(row.get("same_camera"))
    spatial = safe_float(row.get("center_mean_distance_3d"))
    return {
        "same_camera_temporal_conflict": int(same_camera and overlap > 0),
        "temporal_overlap_conflict": int(overlap > 0 and not same_camera),
        "large_spatial_gap_flag": int(spatial is not None and spatial > large_spatial),
        "large_temporal_gap_flag": int(int(row.get("temporal_gap") or 0) > large_temporal),
        "low_quality_fragment_flag": int(
            float(row.get("mean_conf_a") or 0.0) < low_quality_conf
            or float(row.get("mean_conf_b") or 0.0) < low_quality_conf
        ),
        "missing_reid_flag": int(not bool(row.get("embedding_valid_pair"))),
        "missing_geometry_flag": int(spatial is None),
    }


def point_from_fragment(fragment: Dict[str, Any], position: str) -> Optional[np.ndarray]:
    """Read a 3D center vector from normalized fragment fields."""
    values = [safe_float(fragment.get("center_%s_%s" % (axis, position))) for axis in ("x", "y", "z")]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float64)


def vector_from_fragment(fragment: Dict[str, Any], prefix: str) -> Optional[np.ndarray]:
    """Read a 3D vector from normalized fragment fields."""
    values = [safe_float(fragment.get("%s_%s" % (prefix, axis))) for axis in ("x", "y", "z")]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float64)


def distance(point_a: Any, point_b: Any) -> Optional[float]:
    """Return Euclidean distance for compatible vectors."""
    if point_a is None or point_b is None:
        return None
    array_a = np.asarray(point_a, dtype=np.float64).reshape(-1)
    array_b = np.asarray(point_b, dtype=np.float64).reshape(-1)
    if array_a.shape != array_b.shape or not np.all(np.isfinite(array_a)) or not np.all(np.isfinite(array_b)):
        return None
    return float(np.linalg.norm(array_a - array_b))
