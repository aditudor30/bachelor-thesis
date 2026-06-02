"""Visualizations for MTMC transition diagnostics."""

from pathlib import Path
from typing import Dict, List

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.transition_types import TransitionCandidatePair


def plot_transition_distance_histogram(pairs: List[TransitionCandidatePair], output_path: Path) -> None:
    """Plot entry/exit distance histogram."""
    plt = _import_matplotlib()
    values = [pair.entry_exit_distance for pair in pairs if pair.entry_exit_distance is not None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if values:
        ax.hist(values, bins=50)
    ax.set_title("Transition entry-exit distance")
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_transition_gap_histogram(pairs: List[TransitionCandidatePair], output_path: Path) -> None:
    """Plot temporal gap histogram."""
    plt = _import_matplotlib()
    values = [pair.temporal_gap for pair in pairs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if values:
        ax.hist(values, bins=50)
    ax.set_title("Transition temporal gap")
    ax.set_xlabel("frames")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_transition_camera_pair_heatmap(pairs: List[TransitionCandidatePair], output_path: Path) -> None:
    """Plot camera-pair counts as a simple heatmap."""
    plt = _import_matplotlib()
    camera_ids = sorted(set(_all_cameras(pairs)))
    index = {camera_id: pos for pos, camera_id in enumerate(camera_ids)}
    matrix = np.zeros((len(camera_ids), len(camera_ids)), dtype=float)
    for pair in pairs:
        if pair.camera_id_a not in index or pair.camera_id_b not in index:
            continue
        i = index[pair.camera_id_a]
        j = index[pair.camera_id_b]
        matrix[i, j] += 1.0
        matrix[j, i] += 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    if matrix.size > 0:
        image = ax.imshow(matrix)
        fig.colorbar(image, ax=ax)
    ax.set_title("Transition camera-pair counts")
    ax.set_xticks(range(len(camera_ids)))
    ax.set_yticks(range(len(camera_ids)))
    ax.set_xticklabels(camera_ids, rotation=90, fontsize=6)
    ax.set_yticklabels(camera_ids, fontsize=6)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_transition_pairs_bev(
    candidates_by_id: Dict[str, MTMCTrackletCandidate],
    pairs: List[TransitionCandidatePair],
    output_path: Path,
    max_pairs: int = 100,
) -> None:
    """Plot accepted transition links in BEV."""
    plt = _import_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    selected = [pair for pair in pairs if pair.accepted_by_threshold][: int(max_pairs)]
    for pair in selected:
        if pair.candidate_id_a not in candidates_by_id or pair.candidate_id_b not in candidates_by_id:
            continue
        a = candidates_by_id[pair.candidate_id_a]
        b = candidates_by_id[pair.candidate_id_b]
        points = _transition_points(a, b, pair.temporal_relation)
        if points is None:
            continue
        ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], marker="o", linewidth=1.0)
    ax.set_title("Accepted transition links BEV")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def _transition_points(a: MTMCTrackletCandidate, b: MTMCTrackletCandidate, relation: str):
    if relation == "a_before_b":
        return _point_pair(a.exit_center_3d, b.entry_center_3d)
    if relation == "b_before_a":
        return _point_pair(b.exit_center_3d, a.entry_center_3d)
    return None


def _point_pair(left, right):
    if left is None or right is None:
        return None
    left_arr = np.asarray(left, dtype=float).reshape(-1)
    right_arr = np.asarray(right, dtype=float).reshape(-1)
    if left_arr.size < 2 or right_arr.size < 2:
        return None
    return left_arr[:2], right_arr[:2]


def _all_cameras(pairs: List[TransitionCandidatePair]) -> List[str]:
    cameras = []
    for pair in pairs:
        cameras.append(pair.camera_id_a)
        cameras.append(pair.camera_id_b)
    return cameras


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
