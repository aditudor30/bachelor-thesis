"""Visualization helpers for MTMC candidate motion-quality diagnostics."""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from deep_oc_sort_3d.mtmc.candidate_motion_quality import CandidateMotionMetrics
from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate


def plot_step_distances(candidate: MTMCTrackletCandidate, metrics: CandidateMotionMetrics, output_path: Path) -> None:
    """Plot 3D step distance over candidate trajectory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_blank(output_path, "matplotlib unavailable")
        return
    frames = [item[1] for item in metrics.step_distances_3d]
    distances = [item[2] for item in metrics.step_distances_3d]
    plt.figure(figsize=(8, 4))
    plt.plot(frames, distances, marker="o")
    plt.title("%s step distances" % candidate.candidate_id)
    plt.xlabel("frame_b")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def plot_motion_outlier_bev(
    candidate: MTMCTrackletCandidate,
    metrics: CandidateMotionMetrics,
    output_path: Path,
) -> None:
    """Plot one candidate BEV trajectory and mark large motion steps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not candidate.trajectory_3d_sampled:
        _write_blank(output_path, "No 3D trajectory")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_blank(output_path, "matplotlib unavailable")
        return
    xs = [item[1] for item in candidate.trajectory_3d_sampled]
    ys = [item[2] for item in candidate.trajectory_3d_sampled]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o")
    for frame_a, frame_b, distance, _gap in metrics.step_distances_3d:
        if metrics.mean_step_distance_3d is not None and distance < max(metrics.mean_step_distance_3d * 3.0, 3.0):
            continue
        pa = _point_by_frame(candidate, frame_a)
        pb = _point_by_frame(candidate, frame_b)
        if pa is None or pb is None:
            continue
        plt.plot([pa[1], pb[1]], [pa[2], pb[2]], linewidth=3)
    plt.title("%s %s" % (candidate.candidate_id, metrics.motion_quality_flag))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def plot_motion_quality_distribution(metrics_list: List[CandidateMotionMetrics], output_path: Path) -> None:
    """Plot motion-quality flag counts."""
    counts = {}
    for metrics in metrics_list:
        key = str(metrics.motion_quality_flag)
        counts[key] = counts.get(key, 0) + 1
    _bar_plot(counts, "Motion quality distribution", output_path)


def plot_max_step_by_class(metrics_list: List[CandidateMotionMetrics], output_path: Path) -> None:
    """Plot mean max-step distance by class."""
    values = {}
    for metrics in metrics_list:
        if metrics.max_step_distance_3d is None:
            continue
        key = str(metrics.class_name)
        if key not in values:
            values[key] = []
        values[key].append(float(metrics.max_step_distance_3d))
    means = {}
    for key, items in values.items():
        means[key] = float(np.mean(np.asarray(items, dtype=float)))
    _bar_plot(means, "Mean max step by class", output_path)


def _point_by_frame(candidate: MTMCTrackletCandidate, frame_id: int):
    for item in candidate.trajectory_3d_sampled:
        if int(item[0]) == int(frame_id):
            return item
    return None


def _bar_plot(values, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_blank(output_path, "matplotlib unavailable")
        return
    names = sorted(values.keys())
    vals = [values[name] for name in names]
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def _write_blank(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), image)
