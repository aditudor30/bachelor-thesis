"""Optional matplotlib figures for association-pair diagnostics."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from deep_oc_sort_3d.learned_association.pair_dataset_io import safe_float


def generate_pair_dataset_figures(
    pairs: Sequence[Dict[str, Any]], output_dir: Path, enabled: bool = True
) -> List[str]:
    """Generate requested figures, returning warnings instead of failing."""
    if not enabled:
        return ["figures_disabled"]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib_unavailable_figures_skipped"]
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []  # type: List[str]
    try:
        _histogram(plt, pairs, "reid_similarity", output_dir / "reid_similarity_pos_neg.png")
        _histogram(plt, pairs, "temporal_gap", output_dir / "temporal_gap_pos_neg.png")
        _histogram(plt, pairs, "center_mean_distance_3d", output_dir / "spatial_distance_pos_neg.png")
        _camera_heatmap(plt, pairs, output_dir / "camera_pair_heatmap.png")
        _missingness(plt, pairs, output_dir / "feature_missingness.png")
    except Exception as exc:
        warnings.append("figure_generation_failed: %s" % exc)
    return warnings


def _histogram(plt: Any, pairs: Sequence[Dict[str, Any]], feature: str, path: Path) -> None:
    positives = _numeric_values(pairs, feature, 1)
    negatives = _numeric_values(pairs, feature, 0)
    figure, axis = plt.subplots(figsize=(8, 5))
    if positives:
        axis.hist(positives, bins=50, alpha=0.55, label="positive", density=True)
    if negatives:
        axis.hist(negatives, bins=50, alpha=0.55, label="negative", density=True)
    axis.set_title(feature.replace("_", " ").title())
    axis.set_xlabel(feature)
    axis.set_ylabel("density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _camera_heatmap(plt: Any, pairs: Sequence[Dict[str, Any]], path: Path) -> None:
    cameras = sorted(
        {str(row.get("camera_a") or "") for row in pairs}.union(
            {str(row.get("camera_b") or "") for row in pairs}
        )
    )
    cameras = [camera for camera in cameras if camera]
    index = {camera: position for position, camera in enumerate(cameras)}
    matrix = np.zeros((len(cameras), len(cameras)), dtype=np.int64)
    for row in pairs:
        camera_a = str(row.get("camera_a") or "")
        camera_b = str(row.get("camera_b") or "")
        if camera_a in index and camera_b in index:
            matrix[index[camera_a], index[camera_b]] += 1
            if camera_a != camera_b:
                matrix[index[camera_b], index[camera_a]] += 1
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    axis.set_xticks(range(len(cameras)))
    axis.set_yticks(range(len(cameras)))
    axis.set_xticklabels(cameras, rotation=90, fontsize=7)
    axis.set_yticklabels(cameras, fontsize=7)
    axis.set_title("Camera pair counts")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _missingness(plt: Any, pairs: Sequence[Dict[str, Any]], path: Path) -> None:
    features = (
        "reid_similarity",
        "center_mean_distance_3d",
        "velocity_cosine",
        "expected_position_error",
        "bbox_area_mean_a",
        "bbox_area_mean_b",
    )
    rates = []
    for feature in features:
        valid = sum(safe_float(row.get(feature)) is not None for row in pairs)
        rates.append(1.0 - valid / float(max(1, len(pairs))))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(range(len(features)), rates)
    axis.set_xticks(range(len(features)))
    axis.set_xticklabels(features, rotation=35, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("missing rate")
    axis.set_title("Feature missingness")
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def _numeric_values(
    pairs: Sequence[Dict[str, Any]], feature: str, label: int
) -> List[float]:
    values = []
    for row in pairs:
        if int(row.get("same_identity") or 0) != label:
            continue
        value = safe_float(row.get(feature))
        if value is not None:
            values.append(value)
    return values
