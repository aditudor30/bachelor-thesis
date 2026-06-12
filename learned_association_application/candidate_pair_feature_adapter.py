"""Adapt V2 candidate pairs to the exact raw feature schema used in Step 20B."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.pair_feature_builder import build_pair_features
from deep_oc_sort_3d.person_association.person_association_io import serialize_track_key
from deep_oc_sort_3d.person_association.person_pair_mining import (
    PersonTrackFragment,
    load_person_fragments_from_final_export,
    mine_person_candidate_pairs_with_summary,
)
from deep_oc_sort_3d.learned_association_application.scorer_association_io import (
    progress_iter,
    read_csv_rows,
    safe_float,
    safe_int,
)


def load_and_adapt_candidate_pairs(
    config: Dict[str, Any],
    progress: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load 18C pairs when present, rebuilding geometry pairs as a fallback."""
    paths = config.get("paths", {})
    candidate_path = Path(str(paths.get("candidate_pair_reid_scores_csv", "")))
    source = "existing_18c_candidate_pairs"
    use_existing = bool(config.get("candidate_scoring", {}).get("use_existing_candidate_pairs", True))
    rows = read_csv_rows(candidate_path) if use_existing and candidate_path.exists() else []
    fragments = load_person_fragments_from_final_export(
        Path(str(paths.get("v2_final_export_root"))),
        {"class_id": int(config.get("candidate_scoring", {}).get("class_id", 0))},
        show_progress=progress,
    )
    if not rows:
        if not bool(config.get("candidate_scoring", {}).get("rebuild_if_missing", True)):
            raise FileNotFoundError("Candidate pair CSV is missing: %s" % candidate_path)
        source = "rebuilt_geometry_pairs"
        rows, _summary = mine_person_candidate_pairs_with_summary(
            fragments,
            _pair_mining_config(config),
            show_progress=progress,
        )
    lookup = _fragment_lookup(fragments)
    embeddings = _load_fragment_embeddings(config)
    adapted = []  # type: List[Dict[str, Any]]
    missing_fragments = 0
    for index, row in enumerate(progress_iter(rows, progress, "feature adaptation")):
        normalized = _normalize_pair_row(row, index)
        fragment_a = _find_fragment(normalized, "a", lookup)
        fragment_b = _find_fragment(normalized, "b", lookup)
        if fragment_a is None or fragment_b is None:
            missing_fragments += 1
        candidate = dict(normalized)
        candidate["_fragment_a"] = fragment_a or {}
        candidate["_fragment_b"] = fragment_b or {}
        features = build_pair_features(candidate, train_camera_pairs=None, config=_feature_config(config))
        similarity = safe_float(normalized.get("reid_similarity"), None)
        if similarity is None and fragment_a is not None and fragment_b is not None:
            similarity = _embedding_similarity(fragment_a, fragment_b, embeddings)
        features["reid_similarity"] = similarity
        features["reid_distance"] = None if similarity is None else 1.0 - similarity
        features["embedding_valid_pair"] = int(similarity is not None)
        features["missing_reid_flag"] = int(similarity is None)
        features["spatial_distance"] = _first_number(
            normalized,
            ("spatial_distance", "entry_exit_distance_3d", "min_endpoint_distance_3d", "center_mean_distance_3d"),
        )
        adapted.append(features)
    summary = {
        "source": source,
        "candidate_pairs": len(rows),
        "adapted_pairs": len(adapted),
        "missing_fragment_pairs": missing_fragments,
        "fragment_count": len(fragments),
        "embedding_count": len(embeddings),
    }
    return adapted, summary


def align_feature_columns(
    rows: Sequence[Dict[str, Any]],
    selected_features: Sequence[str],
    fallback_values: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Align already-encoded rows to selected model columns for dummy/debug use."""
    fallback = fallback_values or {}
    matrix = np.zeros((len(rows), len(selected_features)), dtype=np.float32)
    report = []  # type: List[Dict[str, Any]]
    for column, feature in enumerate(selected_features):
        missing = 0
        fill = float(fallback.get(feature, 0.0))
        for index, row in enumerate(rows):
            value = safe_float(row.get(feature), None)
            if value is None:
                value = fill
                missing += 1
            matrix[index, column] = float(value)
        report.append({"feature": feature, "missing_count": missing, "fallback_value": fill})
    return matrix, report


def missing_feature_report(
    rows: Sequence[Dict[str, Any]], raw_features: Sequence[str]
) -> List[Dict[str, Any]]:
    """Count unavailable raw features before Step 20B preprocessing."""
    result = []
    for feature in raw_features:
        missing = len([row for row in rows if row.get(feature) in (None, "")])
        result.append(
            {
                "feature": feature,
                "missing_count": missing,
                "missing_ratio": float(missing) / float(len(rows)) if rows else 0.0,
            }
        )
    return result


def _fragment_lookup(fragments: Sequence[PersonTrackFragment]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    result = {}
    for fragment in fragments:
        result[(fragment.subset, fragment.scene_name, fragment.global_track_id)] = _fragment_dict(fragment)
        result[("", fragment.scene_name, fragment.global_track_id)] = _fragment_dict(fragment)
    return result


def _fragment_dict(fragment: PersonTrackFragment) -> Dict[str, Any]:
    entry = fragment.entry_center_3d
    exit_point = fragment.exit_center_3d
    mean_point = None
    if entry is not None and exit_point is not None:
        mean_point = tuple((entry[index] + exit_point[index]) / 2.0 for index in range(3))
    velocity = fragment.velocity_3d
    result = {
        "fragment_id": serialize_track_key(fragment.key),
        "split": fragment.subset,
        "scene_name": fragment.scene_name,
        "class_id": fragment.class_id,
        "class_name": fragment.class_name,
        "camera_id": ";".join(fragment.cameras),
        "frame_start": fragment.start_frame,
        "frame_end": fragment.end_frame,
        "num_observations": fragment.row_count,
        "mean_confidence": fragment.mean_confidence,
        "min_confidence": fragment.mean_confidence,
        "fragment_quality": "unknown",
        "gt_purity": None,
        "global_track_id": fragment.global_track_id,
    }
    for suffix, point in (("start", entry), ("end", exit_point), ("mean", mean_point)):
        if point is not None:
            for axis, value in zip(("x", "y", "z"), point):
                result["center_%s_%s" % (axis, suffix)] = value
    if velocity is not None:
        for axis, value in zip(("x", "y", "z"), velocity):
            result["velocity_%s" % axis] = value
        result["speed_mean"] = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))
    return result


def _normalize_pair_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    scene_name = str(row.get("scene_name") or "")
    return {
        **row,
        "pair_id": str(row.get("pair_id") or "pair_%08d" % index),
        "subset": str(row.get("subset") or row.get("split") or ""),
        "scene_name": scene_name,
        "scene_id": safe_int(row.get("scene_id"), _scene_id(scene_name)),
        "class_id": safe_int(row.get("class_id"), 0),
        "class_name": str(row.get("class_name") or "Person"),
        "fragment_a_id": str(row.get("fragment_a_id") or row.get("fragment_a") or row.get("track_a") or ""),
        "fragment_b_id": str(row.get("fragment_b_id") or row.get("fragment_b") or row.get("track_b") or ""),
        "global_track_a": str(row.get("global_track_a") or row.get("global_track_id_a") or ""),
        "global_track_b": str(row.get("global_track_b") or row.get("global_track_id_b") or ""),
        "camera_a": str(row.get("camera_a") or row.get("cameras_a") or ""),
        "camera_b": str(row.get("camera_b") or row.get("cameras_b") or ""),
        "frame_start_a": safe_int(row.get("frame_start_a", row.get("start_a")), None),
        "frame_end_a": safe_int(row.get("frame_end_a", row.get("end_a")), None),
        "frame_start_b": safe_int(row.get("frame_start_b", row.get("start_b")), None),
        "frame_end_b": safe_int(row.get("frame_end_b", row.get("end_b")), None),
    }


def _scene_id(scene_name: str) -> Optional[int]:
    try:
        return int(scene_name.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return None


def _find_fragment(
    row: Dict[str, Any], suffix: str, lookup: Dict[Tuple[str, str, str], Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    key = (str(row.get("subset", "")), str(row.get("scene_name", "")), str(row.get("global_track_%s" % suffix, "")))
    value = lookup.get(key) or lookup.get(("", key[1], key[2]))
    if value is None:
        return None
    result = dict(value)
    for name in ("camera", "frame_start", "frame_end"):
        override = row.get("%s_%s" % (name, suffix))
        if override not in (None, ""):
            result["camera_id" if name == "camera" else name] = override
    return result


def _load_fragment_embeddings(config: Dict[str, Any]) -> Dict[Tuple[str, str, str], np.ndarray]:
    paths = config.get("paths", {})
    root = Path(str(paths.get("reid_finetuned_association_root", "")))
    matrix_path = root / "embeddings" / "finetuned_fragment_embeddings.npy"
    index_path = root / "embeddings" / "finetuned_fragment_embeddings_index.csv"
    if not matrix_path.exists() or not index_path.exists():
        return {}
    matrix = np.load(str(matrix_path), mmap_mode="r")
    result = {}
    for row in read_csv_rows(index_path):
        index = safe_int(row.get("embedding_index"), None)
        if index is None or index < 0 or index >= len(matrix):
            continue
        key = (str(row.get("subset", "")), str(row.get("scene_name", "")), str(row.get("global_track_id", "")))
        result[key] = np.asarray(matrix[index], dtype=np.float32)
    return result


def _embedding_similarity(
    fragment_a: Dict[str, Any], fragment_b: Dict[str, Any], embeddings: Dict[Tuple[str, str, str], np.ndarray]
) -> Optional[float]:
    key_a = (str(fragment_a.get("split", "")), str(fragment_a.get("scene_name", "")), str(fragment_a.get("global_track_id", "")))
    key_b = (str(fragment_b.get("split", "")), str(fragment_b.get("scene_name", "")), str(fragment_b.get("global_track_id", "")))
    vector_a = embeddings.get(key_a)
    vector_b = embeddings.get(key_b)
    if vector_a is None or vector_b is None:
        return None
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    return None if denominator <= 1e-12 else float(np.dot(vector_a, vector_b) / denominator)


def _first_number(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = safe_float(row.get(key), None)
        if value is not None:
            return value
    return None


def _pair_mining_config(config: Dict[str, Any]) -> Dict[str, Any]:
    constraints = config.get("constraints", {})
    return {
        "max_temporal_gap": int(constraints.get("max_temporal_gap_default", 300)),
        "max_entry_exit_distance": float(constraints.get("max_spatial_distance_default", 12.0)),
        "max_expected_position_error": float(constraints.get("max_spatial_distance_default", 12.0)),
        "forbid_same_camera_temporal_overlap": True,
        "include_rejected": False,
        "store_rejected_pairs": False,
    }


def _feature_config(config: Dict[str, Any]) -> Dict[str, Any]:
    constraints = config.get("constraints", {})
    return {
        "conflict_thresholds": {
            "large_spatial_gap": float(constraints.get("max_spatial_distance_default", 12.0)),
            "large_temporal_gap": int(constraints.get("max_temporal_gap_default", 300)),
        }
    }
