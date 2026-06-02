"""Visualization helpers for MTMC candidates."""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.tracking.track_visualization import color_from_track_id


def plot_candidate_counts_by_class(candidates: List[MTMCTrackletCandidate], output_path: Path) -> None:
    """Save a bar plot with candidate counts by class."""
    kept = [item for item in candidates if item.is_candidate]
    counts = _count_by_class(kept)
    _bar_plot(counts, "Candidate counts by class", output_path)


def plot_candidate_lengths_by_class(candidates: List[MTMCTrackletCandidate], output_path: Path) -> None:
    """Save a bar plot with mean candidate length by class."""
    kept = [item for item in candidates if item.is_candidate]
    values = {}
    for candidate in kept:
        key = str(candidate.class_name)
        if key not in values:
            values[key] = []
        values[key].append(float(candidate.length))
    means = {}
    for key, items in values.items():
        means[key] = sum(items) / float(len(items))
    _bar_plot(means, "Mean candidate length by class", output_path)


def plot_candidate_3d_trajectories_bev(
    candidates: List[MTMCTrackletCandidate],
    output_path: Path,
    max_candidates: int = 100,
) -> None:
    """Save a BEV plot of sampled 3D candidate trajectories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = [item for item in candidates if item.is_candidate and item.trajectory_3d_sampled][: int(max_candidates)]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _bev_cv2(selected, output_path)
        return
    plt.figure(figsize=(8, 8))
    for candidate in selected:
        xs = [point[1] for point in candidate.trajectory_3d_sampled]
        ys = [point[2] for point in candidate.trajectory_3d_sampled]
        if len(xs) < 2:
            continue
        plt.plot(xs, ys, marker="o", linewidth=1)
    plt.title("MTMC candidate BEV trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def visualize_candidate_on_frame(
    image: np.ndarray,
    candidates: List[MTMCTrackletCandidate],
    frame_id: int,
    include_rejected: bool = False,
) -> np.ndarray:
    """Draw candidates visible at frame_id on an RGB image."""
    out = image.copy()
    for candidate in candidates:
        if not include_rejected and not candidate.is_candidate:
            continue
        for point in candidate.trajectory_2d_sampled:
            if int(point[0]) != int(frame_id):
                continue
            color = color_from_track_id(candidate.local_track_id)
            label = "%s T%d" % (candidate.class_name, candidate.local_track_id)
            _draw_bbox(out, (point[1], point[2], point[3], point[4]), color, label)
    return out


def _bar_plot(values, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_blank(output_path, "matplotlib unavailable")
        return
    names = sorted(values.keys())
    counts = [values[name] for name in names]
    plt.figure(figsize=(10, 5))
    plt.bar(names, counts)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def _count_by_class(candidates: List[MTMCTrackletCandidate]):
    counts = {}
    for candidate in candidates:
        key = str(candidate.class_name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _draw_bbox(image: np.ndarray, bbox, color, label: str) -> None:
    x1, y1, x2, y2 = bbox
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _write_blank(path: Path, text: str) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), image)


def _bev_cv2(candidates: List[MTMCTrackletCandidate], output_path: Path) -> None:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    points_all = []
    for candidate in candidates:
        for point in candidate.trajectory_3d_sampled:
            points_all.append((float(point[1]), float(point[2])))
    if not points_all:
        _write_blank(output_path, "No 3D candidate trajectories")
        return
    xs = [point[0] for point in points_all]
    ys = [point[1] for point in points_all]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    for candidate in candidates:
        pts = []
        for _frame, x, y, _z in candidate.trajectory_3d_sampled:
            px = int(40 + (float(x) - x_min) / x_span * 720)
            py = int(760 - (float(y) - y_min) / y_span * 720)
            pts.append((px, py))
        color = color_from_track_id(candidate.local_track_id)
        for index in range(1, len(pts)):
            cv2.line(image, pts[index - 1], pts[index], color, 2)
    cv2.imwrite(str(output_path), image)
