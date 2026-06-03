"""Robust coordinate-space BEV plotting for paper/demo figures."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deep_oc_sort_3d.visualization3d.bev_track_selection import BEVTrack


def compute_percentile_axis_limits(
    tracks: List[BEVTrack],
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    padding_ratio: float = 0.05,
) -> Tuple[float, float, float, float]:
    """Compute robust x/y axis limits from percentile ranges."""
    x_values, y_values = _flatten_xy(tracks)
    if x_values.size == 0 or y_values.size == 0:
        return (0.0, 1.0, 0.0, 1.0)
    x_min = float(np.percentile(x_values, float(lower_percentile)))
    x_max = float(np.percentile(x_values, float(upper_percentile)))
    y_min = float(np.percentile(y_values, float(lower_percentile)))
    y_max = float(np.percentile(y_values, float(upper_percentile)))
    return _pad_limits(x_min, x_max, y_min, y_max, padding_ratio)


def plot_robust_bev_tracks(
    tracks: List[BEVTrack],
    output_path: Union[str, Path],
    title: str = "Coordinate-space BEV global trajectories",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    use_percentile_clipping: bool = True,
    show_start_end: bool = True,
    draw_points: bool = False,
    max_tracks: Optional[int] = 100,
    equal_aspect: bool = True,
    note_not_map_aligned: bool = True,
    figsize: Tuple[float, float] = (8.0, 8.0),
) -> Dict[str, Any]:
    """Save a robust BEV trajectory plot.

    Percentile clipping only changes axis limits for visualization. It does not
    modify the input tracks or any baseline/export files.
    """
    selected = list(tracks)
    if max_tracks is not None:
        selected = selected[: int(max_tracks)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    axis_limits = _axis_limits(selected, lower_percentile, upper_percentile, use_percentile_clipping)

    fig, ax = plt.subplots(figsize=figsize)
    for track in selected:
        x_values = np.asarray(track.x, dtype=float)
        y_values = np.asarray(track.y, dtype=float)
        valid = np.logical_and(np.isfinite(x_values), np.isfinite(y_values))
        x_values = x_values[valid]
        y_values = y_values[valid]
        if x_values.size == 0:
            continue
        color = _color_for_track(track.global_track_id)
        ax.plot(x_values, y_values, linewidth=1.4, alpha=0.8, color=color)
        if draw_points:
            ax.scatter(x_values, y_values, s=8, alpha=0.5, color=color)
        if show_start_end:
            ax.scatter([x_values[0]], [y_values[0]], s=24, marker="o", color=color, edgecolors="black", linewidths=0.4)
            ax.scatter([x_values[-1]], [y_values[-1]], s=30, marker="x", color=color)

    ax.set_xlim(axis_limits[0], axis_limits[1])
    ax.set_ylim(axis_limits[2], axis_limits[3])
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [coordinate units]")
    ax.set_ylabel("y [coordinate units]")
    plot_title = title
    if note_not_map_aligned:
        plot_title += "\nnot map-aligned; percentile clipping is visualization-only"
    ax.set_title(plot_title)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(str(output), dpi=180)
    plt.close(fig)
    plotted_points = int(sum([track.length for track in selected]))
    return {
        "output_path": str(output),
        "plotted_tracks": len(selected),
        "plotted_points": plotted_points,
        "axis_limits": {
            "x_min": axis_limits[0],
            "x_max": axis_limits[1],
            "y_min": axis_limits[2],
            "y_max": axis_limits[3],
        },
        "clipping_percentiles": {
            "enabled": bool(use_percentile_clipping),
            "lower": float(lower_percentile),
            "upper": float(upper_percentile),
        },
        "note": "Coordinate-space BEV; not map-aligned; clipping affects visualization only.",
        "figsize": [float(figsize[0]), float(figsize[1])],
    }


def _axis_limits(
    tracks: List[BEVTrack],
    lower_percentile: float,
    upper_percentile: float,
    use_percentile_clipping: bool,
) -> Tuple[float, float, float, float]:
    if use_percentile_clipping:
        return compute_percentile_axis_limits(tracks, lower_percentile, upper_percentile)
    x_values, y_values = _flatten_xy(tracks)
    if x_values.size == 0 or y_values.size == 0:
        return (0.0, 1.0, 0.0, 1.0)
    return _pad_limits(float(np.min(x_values)), float(np.max(x_values)), float(np.min(y_values)), float(np.max(y_values)), 0.05)


def _pad_limits(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    padding_ratio: float,
) -> Tuple[float, float, float, float]:
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    x_pad = (x_max - x_min) * float(padding_ratio)
    y_pad = (y_max - y_min) * float(padding_ratio)
    return (float(x_min - x_pad), float(x_max + x_pad), float(y_min - y_pad), float(y_max + y_pad))


def _flatten_xy(tracks: List[BEVTrack]) -> Any:
    x_values = []
    y_values = []
    for track in tracks:
        x_values.extend(track.x)
        y_values.extend(track.y)
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    valid = np.logical_and(np.isfinite(x_arr), np.isfinite(y_arr))
    return x_arr[valid], y_arr[valid]


def _color_for_track(track_id: int) -> Any:
    value = int(track_id) * 2654435761
    red = float((value >> 0) & 255) / 255.0
    green = float((value >> 8) & 255) / 255.0
    blue = float((value >> 16) & 255) / 255.0
    return (red, green, blue)
