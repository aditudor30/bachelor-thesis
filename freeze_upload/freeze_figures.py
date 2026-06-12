"""Diagnostic figures for the frozen candidate comparison."""

import csv
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.freeze_upload.freeze_config import output_root
from deep_oc_sort_3d.freeze_upload.freeze_io import write_json


def write_freeze_figures(config: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Write compact local-comparison figures without changing candidate files."""
    figures_root = output_root(config) / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    if config.get("figures", {}).get("enabled", True) is False:
        status = {"status": "skipped", "reason": "figures_disabled", "figures": []}
        write_json(figures_root / "figures_status.json", status)
        return status
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        status = {"status": "skipped", "reason": "matplotlib_unavailable", "figures": []}
        write_json(figures_root / "figures_status.json", status)
        return status

    candidates = comparison.get("candidates", {})
    v2 = candidates.get("v2_current", {})
    v3 = candidates.get("v3_gap_aware_soft", {})
    paths = []
    paths.append(_bar_breakdown(plt, figures_root, v2, v3, "per_scene_rows", "Track1 rows by scene", "scene_id", "track1_rows_by_scene.png"))
    paths.append(_bar_breakdown(plt, figures_root, v2, v3, "per_class_rows", "Track1 rows by class", "class_id", "track1_rows_by_class.png"))
    paths.append(_track_length_histogram(plt, output_root(config), figures_root))
    paths.append(_summary_bars(plt, figures_root, comparison))
    status = {"status": "ok", "figures": [str(path) for path in paths]}
    write_json(figures_root / "figures_status.json", status)
    return status


def _bar_breakdown(
    plt: Any,
    figures_root: Path,
    v2: Dict[str, Any],
    v3: Dict[str, Any],
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
) -> Path:
    left = v2.get(metric, {}) if isinstance(v2.get(metric), dict) else {}
    right = v3.get(metric, {}) if isinstance(v3.get(metric), dict) else {}
    keys = sorted(set(list(left.keys()) + list(right.keys())), key=_sort_key)
    x = list(range(len(keys)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(7.0, len(keys) * 1.1), 4.8))
    axis.bar([value - width / 2.0 for value in x], [left.get(key, 0) for key in keys], width, label="V2 current")
    axis.bar([value + width / 2.0 for value in x], [right.get(key, 0) for key in keys], width, label="V3 gap-aware soft")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("rows")
    axis.set_xticks(x)
    axis.set_xticklabels(keys, rotation=30, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = figures_root / filename
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return path


def _track_length_histogram(plt: Any, root: Path, figures_root: Path) -> Path:
    path = root / "comparison" / "per_track_statistics.csv"
    values = {"v2_current": [], "v3_gap_aware_soft": []}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("candidate_name"))
                try:
                    length = int(float(row.get("num_rows", 0)))
                except (TypeError, ValueError):
                    continue
                if name in values:
                    values[name].append(length)
    all_values = values["v2_current"] + values["v3_gap_aware_soft"]
    upper = _percentile(sorted(all_values), 0.99) if all_values else 1.0
    clipped = max(1.0, upper)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.hist(
        [[min(float(value), clipped) for value in values["v2_current"]], [min(float(value), clipped) for value in values["v3_gap_aware_soft"]]],
        bins=40,
        label=["V2 current", "V3 gap-aware soft"],
        alpha=0.72,
    )
    axis.set_title("Rows per global track (clipped at p99)")
    axis.set_xlabel("rows per track")
    axis.set_ylabel("tracks")
    if all_values:
        axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = figures_root / "rows_per_track_distribution.png"
    figure.savefig(str(output), dpi=160)
    plt.close(figure)
    return output


def _summary_bars(plt: Any, figures_root: Path, comparison: Dict[str, Any]) -> Path:
    rows = comparison.get("metrics", [])
    wanted = ["track1_rows", "unique_tracks", "multi_camera_tracks", "fragmentation_approx"]
    selected = {str(row.get("metric")): row for row in rows if str(row.get("metric")) in wanted}
    labels = [name for name in wanted if name in selected]
    x = list(range(len(labels)))
    width = 0.38
    left = [_number(selected[name].get("v2_current")) for name in labels]
    right = [_number(selected[name].get("v3_gap_aware_soft")) for name in labels]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar([value - width / 2.0 for value in x], left, width, label="V2 current")
    axis.bar([value + width / 2.0 for value in x], right, width, label="V3 gap-aware soft")
    axis.set_title("Frozen candidate local summary")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("count")
    if labels:
        axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = figures_root / "v2_vs_v3_summary_barplot.png"
    figure.savefig(str(output), dpi=160)
    plt.close(figure)
    return output


def _percentile(values: List[int], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = fraction * float(len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return float(values[low]) * (1.0 - weight) + float(values[high]) * weight


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sort_key(value: Any) -> Any:
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)
