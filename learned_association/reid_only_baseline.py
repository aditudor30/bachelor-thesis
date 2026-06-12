"""ReID-similarity baseline for Person association."""

from typing import Any, Dict, Sequence

import numpy as np

from deep_oc_sort_3d.learned_association.pair_dataset_io import safe_float


def reid_scores_from_rows(
    rows: Sequence[Dict[str, Any]], missing_score: float = 0.0
) -> np.ndarray:
    """Return ReID similarity as a direct association score."""
    values = []
    for row in rows:
        value = safe_float(row.get("reid_similarity"), missing_score)
        values.append(missing_score if value is None else value)
    return np.asarray(values, dtype=np.float64)


def reid_predictions(
    rows: Sequence[Dict[str, Any]], threshold: float
) -> np.ndarray:
    """Threshold direct ReID similarities."""
    return (reid_scores_from_rows(rows) >= float(threshold)).astype(np.int64)


def reid_model_payload(thresholds: Sequence[float]) -> Dict[str, Any]:
    """Return a serializable baseline model description."""
    return {
        "model_name": "reid_only_baseline",
        "score_column": "reid_similarity",
        "missing_score": 0.0,
        "thresholds": [float(value) for value in thresholds],
        "trainable": False,
    }
