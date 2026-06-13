"""Figures for the official 023-027 candidate comparison."""

import csv
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.official_023_027.official_config import output_root
from deep_oc_sort_3d.official_023_027.official_track1_io import write_json


def write_official_figures(config: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Write four compact diagnostic figures."""
    root = output_root(config) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        value = {"status": "skipped", "reason": "matplotlib_unavailable", "figures": []}
        write_json(root / "figures_status.json", value)
        return value
    candidates = comparison.get("candidates", {})
    v2 = candidates.get("v2_current", {})
    v3 = candidates.get("v3_gap_aware_soft", {})
    paths = [
        _bar(plt, root, v2.get("per_scene_rows", {}), v3.get("per_scene_rows", {}), "Rows by official scene", "rows_by_scene.png"),
        _bar(plt, root, v2.get("per_class_rows", {}), v3.get("per_class_rows", {}), "Rows by official class", "rows_by_class.png"),
        _track_hist(plt, root, output_root(config) / "comparison" / "per_track_statistics.csv"),
        _summary(plt, root, comparison.get("metrics", [])),
    ]
    value = {"status": "ok", "figures": [str(path) for path in paths]}
    write_json(root / "figures_status.json", value)
    return value


def _bar(plt: Any, root: Path, left: Dict[str, Any], right: Dict[str, Any], title: str, filename: str) -> Path:
    keys = sorted(set(list(left.keys()) + list(right.keys())), key=lambda value: int(value))
    x = list(range(len(keys)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar([value - width / 2.0 for value in x], [left.get(key, 0) for key in keys], width, label="V2 official")
    axis.bar([value + width / 2.0 for value in x], [right.get(key, 0) for key in keys], width, label="V3 official")
    axis.set_title(title)
    axis.set_xticks(x)
    axis.set_xticklabels(keys)
    axis.set_ylabel("rows")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = root / filename
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return path


def _track_hist(plt: Any, root: Path, csv_path: Path) -> Path:
    values = {"v2_current": [], "v3_gap_aware_soft": []}
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("variant") in values:
                    values[str(row.get("variant"))].append(int(float(row.get("num_rows", 0))))
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.hist([values["v2_current"], values["v3_gap_aware_soft"]], bins=40, alpha=0.7, label=["V2 official", "V3 official"])
    axis.set_title("Rows per global track")
    axis.set_xlabel("rows per track")
    axis.set_ylabel("tracks")
    if values["v2_current"] or values["v3_gap_aware_soft"]:
        axis.legend()
    figure.tight_layout()
    path = root / "rows_per_track_distribution.png"
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return path


def _summary(plt: Any, root: Path, rows: List[Dict[str, Any]]) -> Path:
    wanted = ["track1_rows", "unique_tracks"]
    selected = {str(row.get("metric")): row for row in rows if row.get("metric") in wanted}
    labels = [key for key in wanted if key in selected]
    x = list(range(len(labels)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.bar([value - width / 2.0 for value in x], [selected[key].get("v2_current", 0) for key in labels], width, label="V2 official")
    axis.bar([value + width / 2.0 for value in x], [selected[key].get("v3_gap_aware_soft", 0) for key in labels], width, label="V3 official")
    axis.set_title("Official 023-027 candidate summary")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.legend()
    figure.tight_layout()
    path = root / "v2_vs_v3_official_summary.png"
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return path
