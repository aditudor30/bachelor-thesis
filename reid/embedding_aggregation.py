"""Aggregation helpers for crop-level ReID embeddings."""

from typing import List, Optional

import numpy as np

from deep_oc_sort_3d.reid.reid_types import CandidateReIDEmbedding, ReIDCropSample, TrackletReIDEmbedding, normalize_embedding_l2


def aggregate_crop_embeddings(
    embeddings: List[np.ndarray],
    method: str = "mean",
    crop_samples: Optional[List[ReIDCropSample]] = None,
) -> Optional[np.ndarray]:
    """Aggregate crop embeddings and L2-normalize the result."""
    if not embeddings:
        return None
    matrix = np.vstack([np.asarray(item, dtype=float).reshape(1, -1) for item in embeddings])
    if method == "median":
        vector = np.median(matrix, axis=0)
    elif method == "confidence_weighted_mean" and crop_samples is not None and len(crop_samples) == matrix.shape[0]:
        weights = np.asarray([max(float(sample.confidence), 0.0) for sample in crop_samples], dtype=float)
        if float(np.sum(weights)) <= 1e-12:
            vector = np.mean(matrix, axis=0)
        else:
            vector = np.average(matrix, axis=0, weights=weights)
    else:
        vector = np.mean(matrix, axis=0)
    return normalize_embedding_l2(vector)


def compute_tracklet_embedding(
    crop_embeddings: List[np.ndarray],
    crop_samples: List[ReIDCropSample],
    method: str,
    backend: str = "",
) -> Optional[TrackletReIDEmbedding]:
    """Aggregate crop embeddings into one local tracklet embedding."""
    if not crop_embeddings or not crop_samples:
        return None
    embedding = aggregate_crop_embeddings(crop_embeddings, method=method, crop_samples=crop_samples)
    if embedding is None:
        return None
    first = crop_samples[0]
    return TrackletReIDEmbedding(
        subset=first.subset,
        split=first.split,
        scene_name=first.scene_name,
        camera_id=first.camera_id,
        local_track_id=first.local_track_id,
        class_id=first.class_id,
        class_name=first.class_name,
        embedding=embedding,
        embedding_dim=int(embedding.size),
        backend=backend,
        num_crops=len(crop_samples),
        frame_ids=[sample.frame_id for sample in crop_samples],
        mean_confidence=_mean_confidence(crop_samples),
        candidate_id=first.candidate_id,
        global_track_id=first.global_track_id,
    )


def compute_candidate_embedding(
    crop_embeddings: List[np.ndarray],
    crop_samples: List[ReIDCropSample],
    candidate_id: str,
    method: str,
    backend: str = "",
) -> Optional[CandidateReIDEmbedding]:
    """Aggregate crop embeddings into one candidate embedding."""
    if not crop_embeddings or not crop_samples:
        return None
    embedding = aggregate_crop_embeddings(crop_embeddings, method=method, crop_samples=crop_samples)
    if embedding is None:
        return None
    first = crop_samples[0]
    return CandidateReIDEmbedding(
        subset=first.subset,
        split=first.split,
        scene_name=first.scene_name,
        camera_id=first.camera_id,
        local_track_id=first.local_track_id,
        candidate_id=str(candidate_id),
        class_id=first.class_id,
        class_name=first.class_name,
        embedding=embedding,
        embedding_dim=int(embedding.size),
        backend=backend,
        num_crops=len(crop_samples),
        frame_ids=[sample.frame_id for sample in crop_samples],
        mean_confidence=_mean_confidence(crop_samples),
        global_track_id=first.global_track_id,
    )


def _mean_confidence(samples: List[ReIDCropSample]) -> Optional[float]:
    if not samples:
        return None
    return float(np.mean(np.asarray([sample.confidence for sample in samples], dtype=float)))

