"""Matplotlib visualizations for global MTMC association."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, GlobalTrack


def plot_global_tracks_bev(
    global_tracks: List[GlobalTrack],
    output_path: Path,
    max_tracks: int = 100,
) -> None:
    """Plot sampled 3D trajectories for global tracks in BEV."""
    plt = _import_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for track in global_tracks[: int(max_tracks)]:
        points = _trajectory_array(track.trajectory_3d_sampled)
        if points.size == 0:
            continue
        label = "g%d %s" % (track.global_track_id, track.class_name)
        ax.plot(points[:, 0], points[:, 1], marker=".", linewidth=1.0, label=label)
    ax.set_title("Global MTMC BEV trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    if len(global_tracks) <= 15:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_association_graph_summary(edges: List[GlobalAssociationEdge], output_path: Path) -> None:
    """Plot accepted/rejected edge cost histograms."""
    plt = _import_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = [edge.cost for edge in edges if edge.accepted and edge.cost < 1e8]
    rejected = [edge.cost for edge in edges if not edge.accepted and edge.cost < 1e8]
    fig, ax = plt.subplots(figsize=(8, 5))
    if accepted:
        ax.hist(accepted, bins=40, alpha=0.7, label="accepted")
    if rejected:
        ax.hist(rejected, bins=40, alpha=0.5, label="rejected")
    ax.set_title("Global association edge costs")
    ax.set_xlabel("cost")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def visualize_global_track_bev(global_track: GlobalTrack, output_path: Path) -> None:
    """Plot one global track in BEV."""
    plt = _import_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = _trajectory_array(global_track.trajectory_3d_sampled)
    fig, ax = plt.subplots(figsize=(7, 7))
    if points.size > 0:
        ax.plot(points[:, 0], points[:, 1], marker="o", linewidth=1.0)
        ax.scatter(points[0, 0], points[0, 1], s=60, label="start")
        ax.scatter(points[-1, 0], points[-1, 1], s=60, label="end")
    ax.set_title("Global track %d %s" % (global_track.global_track_id, global_track.class_name))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def visualize_candidate_links_bev(
    candidates: List[MTMCTrackletCandidate],
    edges: List[GlobalAssociationEdge],
    output_path: Path,
    max_edges: int = 100,
) -> None:
    """Plot candidate centers and accepted links in BEV."""
    plt = _import_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    centers = _candidate_centers(by_id)
    fig, ax = plt.subplots(figsize=(8, 8))
    if centers:
        xs = [value[0] for value in centers.values()]
        ys = [value[1] for value in centers.values()]
        ax.scatter(xs, ys, s=12, alpha=0.6)
    accepted = [edge for edge in edges if edge.accepted]
    for edge in accepted[: int(max_edges)]:
        if edge.candidate_id_a not in centers or edge.candidate_id_b not in centers:
            continue
        point_a = centers[edge.candidate_id_a]
        point_b = centers[edge.candidate_id_b]
        ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], linewidth=0.8, alpha=0.7)
    ax.set_title("Accepted global candidate links")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def _candidate_centers(candidates_by_id: Dict[str, MTMCTrackletCandidate]) -> Dict[str, np.ndarray]:
    output = {}
    for candidate_id, candidate in candidates_by_id.items():
        if candidate.center_3d_mean is not None:
            arr = np.asarray(candidate.center_3d_mean, dtype=float).reshape(-1)
            if arr.size >= 2:
                output[candidate_id] = arr[:2]
    return output


def _trajectory_array(points: List[Tuple[int, float, float, float]]) -> np.ndarray:
    values = []
    for item in points:
        if len(item) < 4:
            continue
        values.append([float(item[1]), float(item[2]), float(item[3])])
    if not values:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(values, dtype=float)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
